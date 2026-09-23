"""MITRE ATT&CK technique lookup, derived from the indexed corpus.

:mod:`blackbook.knowledge.vocab` carries a small **curated** map from our own
technique terms to ATT&CK IDs (``TECHNIQUE_ATTACK_IDS``). Every row there is a
human judgement — "which ATT&CK technique does *our* term correspond to" — so
the map is deliberately short, and short is the right shape for it: a curated
table is only worth trusting while someone has actually decided each entry.

It is a poor lookup table in the other direction. The indexed ``attack`` source
holds all 697 current ATT&CK enterprise techniques, and a caller asking about
"Scheduled Task/Job", "Impair Defenses" or "T1566.001" is naming a real
technique the curated table has no row for. Transcribing those names into
Python would create a second, larger, hand-maintained table that begins going
stale the moment the source is re-ingested, and ATT&CK ships several times a
year.

This module is the alternative. :class:`AttackIndex` derives the name -> ID and
ID -> name lookups **from the indexed documents themselves**, so the corpus is
the single source of truth, there is no generation step to forget, and
re-ingesting ATT&CK refreshes the lookup with no code change.

Three properties are deliberate:

* **Nothing is fabricated.** Every entry comes from a row in the ``attack``
  source; a term that is not there resolves to ``None``.
* **An ambiguous name is not a lookup.** ATT&CK contains distinct techniques
  that share a name exactly — 23 such groups today, and they are not accidents
  of naming: "DNS" is both the reconnaissance technique T1590.002 and the
  command-and-control one T1071.004, and "Domains" is both acquiring
  infrastructure and compromising someone else's. Resolving one of those to a
  single ID would present a coin flip as a fact, so :meth:`resolve` returns
  ``None`` for them and :meth:`ids_for` exposes the ambiguity to a caller that
  wants to see it.
* **Staleness is bounded and checkable.** The lookup is rebuilt when the
  ``attack`` source changes, so a long-lived server process picks up a
  re-ingest without a restart. See :meth:`refresh` for the exact rule and its
  two stated limits.

A sub-technique is named two ways and the corpus only holds one of them. STIX
stores the leaf name (``"Kerberoasting"``, ``"JavaScript"``), while
attack.mitre.org displays ``"Parent: Child"`` — and the display form is the one
a reader copies out of the site. The parent is recoverable exactly, so
:meth:`refresh` derives those display paths as additional lookup keys. Both
halves of a derived path come from an indexed row, so the alias repeats
something the corpus already says rather than asserting anything new, and a
path whose parent is not indexed is not added at all.
"""

from __future__ import annotations

import json
import logging
import re

from blackbook.storage.database import Database

log = logging.getLogger(__name__)

#: Source id of the ingested MITRE ATT&CK techniques. Mirrors
#: ``MitreAttackAdapter.source_id`` and the ``attack`` entry in the source
#: config; kept here so this module has no import of the ingestion layer.
ATTACK_SOURCE_ID = "attack"

#: A well-formed ATT&CK technique ID: ``T1234`` or a sub-technique ``T1234.567``.
ATTACK_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")

#: The adapter writes the name into the title with the ID appended
#: (``"Kerberoasting (T1558.003)"``), so the name is the title without it.
_TITLE_SUFFIX_RE = re.compile(r"\s*\(T\d{4}(?:\.\d{3})?\)\s*$")

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str) -> str:
    """Fold a technique name onto its lookup key.

    ATT&CK punctuates inconsistently — "Command and Scripting Interpreter",
    "Scheduled Task/Job", "System Network Connections Discovery" — so every run
    of non-alphanumerics collapses to a single space before the key is
    lowercased and trimmed. That makes "Scheduled Task/Job",
    "scheduled task job" and "Scheduled  Task / Job" one key.

    A derived display path folds the same way, which is what leaves the colon
    between parent and child carrying no weight in the lookup: "Command and
    Scripting Interpreter: JavaScript" and a caller's "command and scripting
    interpreter  javascript" are the same key.
    """
    return " ".join(_NON_ALNUM_RE.sub(" ", (name or "").lower()).split())


class AttackIndex:
    """Name/ID lookup over the ATT&CK techniques in the indexed corpus.

    Cheap to hold: it is a dict per direction, built from one indexed query and
    rebuilt only when the source actually changes, so it is safe to keep for
    the life of a long-lived server process.
    """

    def __init__(self, db: Database) -> None:
        self.db = db
        #: ATT&CK ID -> technique name, for every indexed technique.
        self.by_id: dict[str, str] = {}
        #: Normalized name -> sorted ATT&CK IDs sharing it. A list, not a
        #: string, because ATT&CK reuses names across techniques. Carries each
        #: sub-technique's derived display path as well as its stored name.
        self._by_name: dict[str, list[str]] = {}
        self._built_for: tuple[int, str] | None = None
        self.refresh(force=True)

    def __len__(self) -> int:
        return len(self.by_id)

    # -- cache control -----------------------------------------------------

    def _read_signature(self) -> tuple[int, str]:
        """``(document count, newest updated_at)`` for the ATT&CK source.

        Both halves move on any real change: adding or removing techniques
        changes the count, and rewriting one changes the newest ``updated_at``
        (``upsert_document`` stamps it on every content change). It is an
        indexed read of the source's own rows rather than a hash of the corpus,
        so it costs a query and not a scan.
        """
        row = self.db.conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(updated_at), '') FROM documents "
            "WHERE source_id = ?",
            (ATTACK_SOURCE_ID,),
        ).fetchone()
        return (int(row[0]), str(row[1]))

    def refresh(self, *, force: bool = False) -> bool:
        """Rebuild the lookup if the ATT&CK source changed.

        Returns True when the index was rebuilt. Called at the top of every
        lookup, which is what keeps a running server honest about a re-ingest
        without needing a scheduler or an explicit invalidation call.

        Two limits are stated rather than hidden. A document rewritten within
        the same second as the last build is not noticed, because SQLite's
        ``datetime('now')`` has one-second resolution. And the check keys on
        the source's rows, so a bundle re-fetched with *identical* content
        changes nothing — which is the intended behaviour, since the derived
        lookup would be identical too.
        """
        signature = self._read_signature()
        if not force and signature == self._built_for:
            return False

        by_id: dict[str, str] = {}
        by_name: dict[str, set[str]] = {}
        for row in self.db.iter_documents([ATTACK_SOURCE_ID]):
            attack_id_value, name = self._parse(row)
            if attack_id_value is None or not name:
                continue
            by_id[attack_id_value] = name
            by_name.setdefault(normalize_name(name), set()).add(attack_id_value)

        # Display paths, derived rather than transcribed. A sub-technique's
        # parent is the ID with its final segment dropped, and both the parent
        # name and the child name are read from indexed rows, so a path is a
        # restatement of the corpus and never a new claim. Three consequences
        # are worth naming. A path whose parent is absent is not added, which
        # is what keeps this from inventing a parent. Two sub-techniques of
        # differently-parented techniques that share a name produce one key
        # with two IDs, so the ordinary ambiguity rule covers it and
        # :meth:`ids_for` shows both. And a derived path that happened to equal
        # some technique's own stored name would land in the same list rather
        # than overwrite it, so :meth:`resolve` declines instead of silently
        # preferring one. Neither of the last two occurs in the current corpus;
        # both are handled anyway, because a future ATT&CK release could.
        for attack_id_value, name in by_id.items():
            if "." not in attack_id_value:
                continue
            parent_name = by_id.get(attack_id_value.rsplit(".", 1)[0])
            if parent_name:
                by_name.setdefault(
                    normalize_name(f"{parent_name}: {name}"), set()
                ).add(attack_id_value)

        self.by_id = by_id
        self._by_name = {key: sorted(ids) for key, ids in by_name.items()}
        self._built_for = signature
        log.debug(
            "attack index: %d techniques, %d distinct names (%d ambiguous)",
            len(by_id),
            len(self._by_name),
            sum(1 for ids in self._by_name.values() if len(ids) > 1),
        )
        return True

    @staticmethod
    def _parse(row: dict) -> tuple[str | None, str]:
        """Recover ``(attack_id, name)`` from one ``attack`` document row.

        The ID is the document's ``external_id`` (the adapter mirrors it in
        ``metadata.attack_id``) and the name is its title minus the appended
        ID. A row that does not yield a well-formed ID and a non-empty name is
        skipped, never guessed at.
        """
        meta = row.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except ValueError:
                meta = None
        if not isinstance(meta, dict):
            meta = {}

        attack_id_value = str(
            row.get("external_id") or meta.get("attack_id") or ""
        ).strip().upper()
        if not ATTACK_ID_RE.match(attack_id_value):
            return None, ""
        name = _TITLE_SUFFIX_RE.sub("", str(row.get("title") or "")).strip()
        return attack_id_value, name

    # -- lookups -----------------------------------------------------------

    def resolve(self, term: str) -> str | None:
        """ATT&CK ID for a technique name or ID, or ``None``.

        Accepts whichever the caller has:

        * an ID (``"T1558.003"``, case-insensitive) resolves to itself when the
          technique is indexed;
        * a name (``"Kerberoasting"``, ``"kerberoasting"``, ``"steal or forge
          kerberos tickets"``) is folded by :func:`normalize_name` and looked up;
        * a sub-technique's display path (``"Command and Scripting
          Interpreter: JavaScript"``, ``"OS Credential Dumping: DCSync"``)
          resolves too, because :meth:`refresh` derives those from the indexed
          parent. That form is not any document's name, so without the
          derivation the very names attack.mitre.org prints would be the ones
          that failed to resolve.

        ``None`` means "not a technique this corpus can name" — an unknown term,
        and equally a name ATT&CK gives to more than one technique, which is
        asked about through :meth:`ids_for` instead. It is never a guess.
        """
        self.refresh()
        key = (term or "").strip().upper()
        if ATTACK_ID_RE.match(key):
            return key if key in self.by_id else None
        ids = self._by_name.get(normalize_name(term))
        if ids is not None and len(ids) == 1:
            return ids[0]
        return None

    def ids_for(self, name: str) -> list[str]:
        """Every indexed ATT&CK ID whose name folds to ``name`` (sorted).

        Empty when the name is unknown, and longer than one when ATT&CK reuses
        it — the case :meth:`resolve` refuses to guess at.
        """
        self.refresh()
        return list(self._by_name.get(normalize_name(name), ()))

    def name(self, attack_id_value: str) -> str | None:
        """The indexed technique name for an ATT&CK ID, or ``None``.

        The reverse of :meth:`resolve`, and exact in that direction: IDs are
        unique even where names are not.
        """
        self.refresh()
        return self.by_id.get((attack_id_value or "").strip().upper())
