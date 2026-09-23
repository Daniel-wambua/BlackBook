"""Phase 4 knowledge-graph construction.

The graph is a lightweight, **evidence-linked** layer over the already-indexed
corpus. It never invents facts: every entity is either a real source name, a
real document title, or a term from the controlled vocabulary that *literally
occurs* in a document; every relationship records the ``doc_id`` it was derived
from so a caller can cite the exact supporting text. Heuristic (keyword-derived)
edges are flagged ``inferred=True`` with a graded ``confidence``; structural
edges taken directly from a document's identity (a writeup belongs to its
source, runs on the OS named in its card) are ``inferred=False``.

The graph **enhances** retrieval — it is queried to enrich technique dossiers
and case results — but retrieval works with an empty graph, so a corpus that was
never graphed still answers every tool.

Nothing here executes anything or reaches the network. ``rebuild`` is a pure,
idempotent transform of indexed rows into entities/relationships, wrapped in a
single transaction so a partial build can never be observed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from blackbook.knowledge.vocab import extract_terms, is_writeup_category
from blackbook.storage.database import Database, sha256_text
from blackbook.storage.models import Entity, Relationship

log = logging.getLogger(__name__)

# Entity types.
E_TECHNIQUE = "technique"
E_TOOL = "tool"
E_SERVICE = "service"
E_OS = "os"
E_WRITEUP = "writeup"
E_SOURCE = "source"

# Predicates.
P_DEMONSTRATED_IN = "demonstrated_in"  # technique -> writeup
P_USED_IN = "used_in"                  # tool -> writeup
P_PRESENT_IN = "present_in"            # service -> writeup
P_DOCUMENTED_BY = "documented_by"      # technique|writeup -> source
P_RUNS_ON = "runs_on"                  # writeup -> os
P_USES = "uses"                        # technique -> tool  (co-occurrence)
P_TARGETS = "targets"                  # technique -> service (co-occurrence)

# Confidence tiers for inferred, keyword-derived edges.
_CONF_HEADING = 0.9   # term appears in the title / a section heading
_CONF_METADATA = 0.75  # term came from the adapter's explicit inferred signals
_CONF_BODY = 0.6      # term appears only in body text
_CONF_COOCCUR = 0.5   # two terms merely co-occur in the same document

# Bound how much body text we scan per document (controlled corpus; this is a
# safety cap against a pathologically large document, not an expected limit).
_BODY_SCAN_CAP = 500_000

# Bump when the term extraction itself changes in a way that would produce
# different terms for the same document: a new vocabulary list, a changed alias
# table, a different scan cap. It is part of every document fingerprint, so a
# bump invalidates the whole term cache at once and the next rebuild re-extracts
# every document. Without it, a cache written by an older version would keep
# serving terms the current code would no longer produce.
_TERMS_CACHE_VERSION = 1

_KIND_TO_ETYPE = {"service": E_SERVICE, "technique": E_TECHNIQUE, "tool": E_TOOL}
_KIND_TO_WRITEUP_PRED = {
    "technique": P_DEMONSTRATED_IN,
    "tool": P_USED_IN,
    "service": P_PRESENT_IN,
}

# Predicates whose edges are statistical rather than documentary.
#
# A documentary edge is a fact about one document ("this writeup was published by
# 0xdf"), so one row per document is the honest storage. A co-occurrence edge is
# a single claim — "this technique uses this tool" — for which any number of
# documents can be witnesses, and storing one row per witness says the same thing
# hundreds of times: on a real corpus ``targets`` averages ~43 rows per distinct
# pair, and every one of them is a duplicate of the claim. These predicates are
# therefore collapsed to one row per pair, with ``support`` counting how many
# distinct documents backed it and ``evidence_doc_id`` naming the lowest of them
# as the citation. The claim and its evidence survive; the repetition does not.
_COOCCURRENCE_PREDICATES = frozenset({P_USES, P_TARGETS})


@dataclass
class GraphStats:
    """Summary of a graph build."""

    entities: int = 0
    relationships: int = 0
    documents: int = 0
    writeups: int = 0
    #: Documents whose term extraction was served from the cache instead of
    #: re-run. Deliberately absent from :meth:`as_dict`: it describes the *run*,
    #: not the graph, so two builds of the same corpus would report different
    #: payloads (0 for the first, all for the second) and the graph itself would
    #: look like it had changed. Callers that want it read the attribute.
    reused: int = 0
    #: True when the stored graph already matched the corpus and the rebuild was
    #: skipped outright. Also a property of the run, not the graph.
    skipped: bool = False
    by_entity_type: dict[str, int] = field(default_factory=dict)
    by_predicate: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "entities": self.entities,
            "relationships": self.relationships,
            "documents": self.documents,
            "writeups": self.writeups,
            "by_entity_type": dict(self.by_entity_type),
            "by_predicate": dict(self.by_predicate),
        }


@dataclass
class WriteupCoverage:
    """Which sources contribute writeups, and which contribute none.

    Counted by re-applying :func:`is_writeup_document` to the indexed documents
    rather than by reading the graph's ``writeup`` entities, so it describes what
    the corpus supports whether or not the graph has been built. The built
    figure is carried alongside, in ``graph_writeups``, for comparison.
    """

    #: Every indexed document.
    documents: int = 0
    #: Documents the writeup rule accepts.
    eligible: int = 0
    #: source_id -> documents accepted as writeups (sources with none are 0).
    by_source: dict[str, int] = field(default_factory=dict)
    #: source_id -> every indexed document from that source.
    docs_by_source: dict[str, int] = field(default_factory=dict)
    #: One entry per source with documents but no writeups: source_id,
    #: documents, reason.
    gaps: list[dict] = field(default_factory=list)
    #: ``writeup`` entities actually present in the graph.
    graph_writeups: int = 0
    #: Whether a graph has been built at all.
    graph_built: bool = False

    @property
    def sources_with_writeups(self) -> int:
        """How many sources contribute at least one writeup."""
        return sum(1 for n in self.by_source.values() if n)

    def as_dict(self) -> dict:
        return {
            "documents": self.documents,
            "eligible": self.eligible,
            "sources": len(self.by_source),
            "sources_with_writeups": self.sources_with_writeups,
            "by_source": dict(self.by_source),
            "docs_by_source": dict(self.docs_by_source),
            "gaps": [dict(g) for g in self.gaps],
            "graph_writeups": self.graph_writeups,
            "graph_built": self.graph_built,
        }


def writeup_coverage(db: Database) -> WriteupCoverage:
    """Report writeup coverage across the corpus, source by source.

    Answers the question the graph's own totals hide. ``GraphStats.writeups`` is
    a single number, and a single number cannot show that every writeup in the
    graph comes from one source while the largest source in the corpus
    contributes none: the case-study layer looks populated while actually
    covering a fraction of what was ingested, and nothing in the tooling says so.

    Read-only, and deliberately independent of the graph. It re-applies
    :func:`is_writeup_document` to each indexed document, so it needs no graph,
    cannot be stale, and stays correct for a corpus that was never graphed.
    ``graph_writeups`` is read from the built graph for comparison: a *full*
    rebuild makes the two figures equal, so a difference means either the graph
    is stale or it was built scoped to a subset of sources
    (``rebuild(source_ids=[...])``), both of which are worth knowing rather than
    assuming.
    """
    coverage = WriteupCoverage()
    for doc in db.iter_documents():
        coverage.documents += 1
        source_id = str(doc.get("source_id") or "")
        coverage.docs_by_source[source_id] = coverage.docs_by_source.get(source_id, 0) + 1
        if is_writeup_document(
            source_id, _json_list(doc.get("categories")), _json_obj(doc.get("metadata"))
        ):
            coverage.by_source[source_id] = coverage.by_source.get(source_id, 0) + 1
            coverage.eligible += 1
    # A source with documents but no writeups never incremented anything, so it
    # is absent from ``by_source``. Recording the zero is what lets the gap loop
    # below tell "contributes none" apart from "not in this corpus".
    for source_id in coverage.docs_by_source:
        coverage.by_source.setdefault(source_id, 0)

    # Every gap carries the same reason, and that is a consequence of the rule
    # rather than a shortcut. `is_writeup_document` is a disjunction whose first
    # route is fixed per source (the `0xdf` identity test), so a source with zero
    # eligible documents is one where the three remaining routes failed for every
    # single document. There is no other way to arrive at zero, so there is
    # nothing per-source left to distinguish.
    for source_id in sorted(coverage.docs_by_source):
        if coverage.by_source[source_id]:
            continue
        total = coverage.docs_by_source[source_id]
        coverage.gaps.append(
            {
                "source_id": source_id,
                "documents": total,
                "reason": (
                    f"none of the {total} documents carries a writeup category "
                    "marker, a machine_name, or a kind"
                ),
            }
        )

    row = db.conn.execute(
        "SELECT COUNT(*) FROM entities WHERE entity_type = ?", (E_WRITEUP,)
    ).fetchone()
    coverage.graph_writeups = int(row[0]) if row else 0
    # "Built" means the entities table holds anything at all, not that it holds
    # writeups: a graph built from a corpus with no writeups is still built, and
    # reporting it as unbuilt would hide the very disagreement worth reporting.
    coverage.graph_built = bool(
        db.conn.execute("SELECT 1 FROM entities LIMIT 1").fetchone()
    )
    return coverage


def _json_list(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        val = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return [str(x) for x in val] if isinstance(val, list) else []


def _json_obj(raw) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        val = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return val if isinstance(val, dict) else {}


def is_writeup_document(source_id: str, categories: list[str] | None, meta: dict) -> bool:
    """Whether a document counts as a hands-on writeup / case study.

    This is the single definition of the rule. :class:`GraphBuilder` calls it to
    decide whether a document gets a ``writeup`` entity, and
    :func:`writeup_coverage` calls it to explain which sources produce none, so
    the diagnostic can never disagree with the thing it describes.

    Four routes, any one of them sufficient:

    * the document belongs to the ``0xdf`` source, every post of which is a
      writeup;
    * a category token marks it as one (``htb``, ``ctf``, ``writeup``, ...);
    * the metadata carries a ``machine_name`` — an HTB/TryHackMe box card;
    * the metadata ``kind`` is set to something other than unknown.

    Only the first is decided by the source's identity. The other three are data
    the adapter had to supply, so a source whose documents carry none of them
    contributes no writeups however writeup-shaped its content actually is. That
    gap is what :func:`writeup_coverage` exists to make visible.
    """
    return (
        source_id == "0xdf"
        or is_writeup_category(categories)
        or bool(meta.get("machine_name"))
        or (meta.get("kind") not in (None, "", "unknown"))
    )


# Field separator for the fingerprint. Any byte not expected in the joined
# fields will do; a unit separator cannot appear in a title, a hash, or the JSON
# that json.dumps emits, so the fields cannot be shifted across the boundary to
# forge a collision (title="a\x1fb" against title="a", source="b").
_FP_SEP = "\x1f"


def document_fingerprint(doc: dict) -> str:
    """Fingerprint everything the term extraction reads from a document.

    The cached terms for a document are reused only while this is unchanged, so
    it must cover every input the extraction sees and nothing else. That means
    the title and the chunk text (reached through ``content_hash``) plus the
    three fields that decide the writeup rule and term confidence —
    ``source_id``, ``categories``, ``metadata``. The content hash is a sound
    stand-in for the text because the ingestion pipeline rewrites a document's
    chunks only when that hash changes, so unchanged chunks follow from an
    unchanged hash.

    ``_TERMS_CACHE_VERSION`` is mixed in first, so bumping it invalidates every
    row at once.
    """
    return sha256_text(_FP_SEP.join([
        str(_TERMS_CACHE_VERSION),
        str(doc.get("source_id") or ""),
        str(doc.get("title") or ""),
        str(doc.get("content_hash") or ""),
        str(doc.get("categories") or "[]"),
        str(doc.get("metadata") or "{}"),
    ]))


def corpus_fingerprint(fingerprints: dict[int, str]) -> str:
    """One hash over a whole set of document fingerprints.

    Folded over doc_id order so the result depends on the corpus, not on dict
    iteration order. Two corpora differing by an added, removed, or changed
    document get different values; so does the same corpus under a different
    scope, because the set of doc_ids differs.
    """
    parts = [f"{doc_id}:{fingerprints[doc_id]}" for doc_id in sorted(fingerprints)]
    return sha256_text(_FP_SEP.join(parts)) if parts else sha256_text("")


def decode_cached_terms(raw: str) -> dict | None:
    """Parse a cached terms payload, or None if it is unusable.

    A row that does not parse, or parses to something that is not the expected
    shape, is treated as a cache miss rather than an error: the worst case is
    one document re-extracted. This keeps a corrupt row from failing a rebuild.
    """
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    heading, body = payload.get("heading"), payload.get("body")
    if not isinstance(heading, dict) or not isinstance(body, dict):
        return None
    return {"heading": heading, "body": body}


@dataclass
class _AccumulatedEdge:
    """One edge under construction, before it reaches the database.

    ``support`` means what it means on :class:`~blackbook.storage.models.Relationship`:
    1 for a documentary edge, the number of distinct witnessing documents for a
    collapsed co-occurrence edge.
    """

    confidence: float
    inferred: bool
    evidence_doc_id: int | None
    support: int = 1


class GraphBuilder:
    """Builds the entity/relationship graph from indexed documents."""

    def __init__(self, db: Database):
        self.db = db
        self._entity_ids: dict[tuple[str, str], int] = {}
        # (subject_id, predicate, object_id, evidence_doc_id) -> edge. The
        # evidence element is None for co-occurrence predicates, which collapse
        # every witness of a pair into a single entry.
        self._edges: dict[tuple[int, str, int, int | None], _AccumulatedEdge] = {}

    # -- public API --------------------------------------------------------

    def rebuild(
        self, source_ids: list[str] | None = None, *, incremental: bool = True
    ) -> GraphStats:
        """Rebuild the graph from the (optionally scoped) corpus.

        Idempotent: running twice over the same corpus yields the same graph.
        The clear + rebuild happen in one transaction, so a partial build is
        never observable.

        Term extraction is a regex pass over every document's full text and
        dominates the cost of a build, so it is cached per document in
        ``document_graph``, keyed by a fingerprint of everything it read. Three
        outcomes follow:

        * **Nothing changed.** Every document's fingerprint still matches its
          cache row and the stored graph was built from the same corpus, so the
          rebuild is skipped entirely (``stats.skipped``) and the existing graph
          is left alone.
        * **Something changed.** The graph is cleared and rebuilt from every
          document, but only the changed documents are re-extracted; the rest
          reuse their cached terms. The new graph is identical to what a full
          rebuild would produce, because the terms fed to the assembly step are
          the same values either way.
        * **``incremental=False``.** No cache is read or trusted: every document
          is re-extracted. The escape hatch for the case where the cache itself
          is suspect, and what the tests compare the incremental path against.

        A scope change (rebuilding one source after building all of them, or the
        reverse) counts as "something changed": the recorded scope is part of
        what makes a build current, so the narrower build replaces the wider one
        rather than being skipped as a subset of it.
        """
        self._entity_ids.clear()
        self._edges.clear()
        stats = GraphStats()

        docs = list(self.db.iter_documents(source_ids))
        fingerprints = {int(d["doc_id"]): document_fingerprint(d) for d in docs}
        corpus_fp = corpus_fingerprint(fingerprints)
        cache = self.db.document_graph_cache() if incremental else {}

        if incremental and self._graph_is_current(source_ids, docs, fingerprints, corpus_fp):
            stats.documents = len(docs)
            stats.reused = len(docs)
            stats.skipped = True
            stats.writeups = sum(
                1 for d in docs
                if is_writeup_document(
                    str(d["source_id"]), _json_list(d.get("categories")),
                    _json_obj(d.get("metadata")),
                )
            )
            self._tally(stats)
            log.info(
                "graph rebuild skipped: %d documents unchanged", stats.documents,
            )
            return stats

        with self.db.session():
            self.db.clear_graph()
            rows: list[tuple[int, str, str]] = []
            for doc in docs:
                stats.documents += 1
                doc_id = int(doc["doc_id"])
                fingerprint = fingerprints[doc_id]
                terms = self._cached_terms(cache.get(doc_id), fingerprint)
                if terms is not None:
                    stats.reused += 1
                else:
                    terms = self.extract_document_terms(doc)
                rows.append((doc_id, fingerprint, json.dumps(terms)))
                if self._process_document(doc, terms):
                    stats.writeups += 1
            self._flush_edges()
            self.db.replace_document_graph(rows, source_ids)
            self.db.set_graph_build({
                "corpus_fingerprint": corpus_fp,
                "documents": stats.documents,
                "sources": sorted(source_ids or []),
            })

        self._tally(stats)
        log.info(
            "graph rebuilt: %d entities, %d relationships from %d documents "
            "(%d reused from cache)",
            stats.entities, stats.relationships, stats.documents, stats.reused,
        )
        return stats

    def _graph_is_current(
        self,
        source_ids: list[str] | None,
        docs: list[dict],
        fingerprints: dict[int, str],
        corpus_fp: str,
    ) -> bool:
        """Whether the stored graph was built from exactly this corpus.

        Checked against the build record rather than the term cache alone: the
        cache also survives a ``clear_graph()``, which leaves a valid cache
        describing a graph that is no longer there.
        """
        marker = self.db.get_graph_build()
        if marker is None:
            return False
        if marker.get("corpus_fingerprint") != corpus_fp:
            return False
        if marker.get("documents") != len(docs):
            return False
        if sorted(marker.get("sources") or []) != sorted(source_ids or []):
            return False
        # The build record exists, so the graph should too. If the relationship
        # table was emptied by something other than a rebuild, the record is
        # stale and the graph has to be assembled again from the cached terms,
        # which is cheap.
        if docs and not self.db.has_graph():
            return False
        return True

    @staticmethod
    def _cached_terms(entry: tuple[str, str] | None, fingerprint: str) -> dict | None:
        """The cached terms for a document, or None if it must be re-extracted."""
        if entry is None or entry[0] != fingerprint:
            return None
        return decode_cached_terms(entry[1])

    # -- per-document extraction ------------------------------------------

    def _process_document(self, doc: dict, terms: dict) -> bool:
        """Assemble entities/edges for one document. True if a writeup.

        ``terms`` is the extraction output for this document, from
        :meth:`extract_document_terms` or from the cache. Assembly is fed the
        same values either way, which is what makes an incremental rebuild
        produce the same graph as a full one.
        """
        doc_id = int(doc["doc_id"])
        source_id = str(doc["source_id"])
        categories = _json_list(doc.get("categories"))
        meta = _json_obj(doc.get("metadata"))

        heading_terms, body_terms = terms["heading"], terms["body"]
        md_terms = {
            "service": [t.lower() for t in _json_list(meta.get("services"))],
            "technique": [t.lower() for t in _json_list(meta.get("techniques"))],
            "tool": [t.lower() for t in _json_list(meta.get("tools"))],
        }

        # Union of everything mentioned anywhere, per kind.
        all_terms: dict[str, set[str]] = {}
        for kind in ("service", "technique", "tool"):
            all_terms[kind] = (
                set(heading_terms.get(kind, []))
                | set(body_terms.get(kind, []))
                | set(md_terms[kind])
            )

        source_ent = self._ensure_source(doc)

        is_writeup = is_writeup_document(source_id, categories, meta)
        writeup_ent: int | None = None
        if is_writeup:
            writeup_ent = self._ensure_writeup(doc, meta, source_id)
            # Structural: this writeup is published by this source.
            self._add_edge(writeup_ent, P_DOCUMENTED_BY, source_ent,
                           doc_id, 1.0, inferred=False)
            os_name = meta.get("os")
            if os_name:
                os_ent = self._ensure_entity(str(os_name), E_OS)
                self._add_edge(writeup_ent, P_RUNS_ON, os_ent,
                               doc_id, 1.0, inferred=False)

        # Term entities + edges.
        for kind in ("technique", "tool", "service"):
            etype = _KIND_TO_ETYPE[kind]
            for term in sorted(all_terms[kind]):
                term_ent = self._ensure_entity(term, etype)
                conf = self._term_confidence(term, kind, heading_terms, md_terms)
                if writeup_ent is not None:
                    self._add_edge(term_ent, _KIND_TO_WRITEUP_PRED[kind],
                                   writeup_ent, doc_id, conf, inferred=True)
                if kind == "technique":
                    # A source *documents* a technique it names in a heading, or
                    # (weaker) merely mentions in body text.
                    dconf = _CONF_HEADING if term in set(
                        heading_terms.get("technique", [])
                    ) else _CONF_BODY
                    self._add_edge(term_ent, P_DOCUMENTED_BY, source_ent,
                                   doc_id, dconf, inferred=True)

        # Technique co-occurrence with tools/services in the same document.
        for tech in sorted(all_terms["technique"]):
            tech_ent = self._ensure_entity(tech, E_TECHNIQUE)
            for tool in sorted(all_terms["tool"]):
                self._add_edge(tech_ent, P_USES,
                               self._ensure_entity(tool, E_TOOL),
                               doc_id, _CONF_COOCCUR, inferred=True)
            for svc in sorted(all_terms["service"]):
                self._add_edge(tech_ent, P_TARGETS,
                               self._ensure_entity(svc, E_SERVICE),
                               doc_id, _CONF_COOCCUR, inferred=True)

        return is_writeup

    def extract_document_terms(self, doc: dict) -> dict:
        """Vocabulary terms for one document, as ``{"heading": ..., "body": ...}``.

        The expensive half of a build: a regex pass per vocabulary term over the
        document's headings and its full body text. Split out from the assembly
        step so its result can be cached and replayed, and so the tests can count
        how often it runs.

        Heading terms come from the title plus every section-path heading — a
        strong signal the document is *about* the term. Body terms come from the
        chunk text.
        """
        title = str(doc["title"])
        headings: list[str] = []
        body_parts: list[str] = []
        for c in self.db.document_chunks(int(doc["doc_id"])):
            headings.extend(_json_list(c.get("section_path")))
            if c.get("text"):
                body_parts.append(str(c["text"]))
        heading_text = title + "\n" + "\n".join(headings)
        body_text = "\n".join(body_parts)[:_BODY_SCAN_CAP]
        return {
            "heading": extract_terms(heading_text),
            "body": extract_terms(body_text),
        }

    @staticmethod
    def _term_confidence(term: str, kind: str, heading_terms: dict, md_terms: dict) -> float:
        if term in set(heading_terms.get(kind, [])):
            return _CONF_HEADING
        if term in set(md_terms.get(kind, [])):
            return _CONF_METADATA
        return _CONF_BODY

    # -- entity/edge helpers ----------------------------------------------

    def _ensure_entity(self, name: str, entity_type: str, description: str = "",
                       meta: dict | None = None) -> int:
        key = (name, entity_type)
        cached = self._entity_ids.get(key)
        if cached is not None:
            return cached
        eid = self.db.upsert_entity(
            Entity(name=name, entity_type=entity_type,
                   description=description, meta=meta or {})
        )
        self._entity_ids[key] = eid
        return eid

    def _ensure_source(self, doc: dict) -> int:
        source_id = str(doc["source_id"])
        row = self.db.get_source(source_id)
        name = row["name"] if row else source_id
        return self._ensure_entity(name, E_SOURCE, description=f"source:{source_id}",
                                   meta={"source_id": source_id})

    def _ensure_writeup(self, doc: dict, meta: dict, source_id: str) -> int:
        emeta = {
            "doc_id": int(doc["doc_id"]),
            "source_id": source_id,
            "external_id": doc.get("external_id"),
            "url": doc.get("url"),
            "os": meta.get("os"),
            "difficulty": meta.get("difficulty"),
            "kind": meta.get("kind"),
        }
        return self._ensure_entity(str(doc["title"]), E_WRITEUP,
                                   description=meta.get("summary") or "",
                                   meta=emeta)

    def _add_edge(self, subject_id: int, predicate: str, object_id: int,
                  evidence_doc_id: int | None, confidence: float,
                  inferred: bool) -> None:
        """Accumulate an edge, de-duplicating on (subj, pred, obj, evidence).

        ``add_relationship`` has no ON CONFLICT, so we dedupe in memory and keep
        the strongest confidence seen for an identical edge with the same
        evidence document. Edges with *different* evidence documents are kept
        separately — each is a distinct citation — except for the predicates in
        ``_COOCCURRENCE_PREDICATES``, where different documents witness one claim
        rather than making several, so they are folded into a single edge whose
        ``support`` counts the witnesses and whose evidence is the lowest doc id.
        """
        if subject_id == object_id:
            return  # never relate an entity to itself
        collapse = predicate in _COOCCURRENCE_PREDICATES
        key = (subject_id, predicate, object_id, None if collapse else evidence_doc_id)
        prev = self._edges.get(key)
        if prev is None:
            self._edges[key] = _AccumulatedEdge(confidence, inferred, evidence_doc_id)
            return
        evidence = prev.evidence_doc_id
        if collapse and evidence_doc_id is not None:
            # Lowest wins, so the citation is stable across rebuilds regardless
            # of the order documents happen to arrive in.
            evidence = (
                evidence_doc_id if evidence is None else min(evidence, evidence_doc_id)
            )
        self._edges[key] = _AccumulatedEdge(
            confidence=max(confidence, prev.confidence),
            # Once any occurrence is structural (inferred=False), keep it so.
            inferred=inferred and prev.inferred,
            evidence_doc_id=evidence,
            support=prev.support + 1 if collapse else prev.support,
        )

    def _flush_edges(self) -> None:
        # The evidence comes off the accumulated edge, never off the key: for a
        # collapsed co-occurrence edge the key's evidence slot is deliberately
        # None (that is what folds the witnesses together) while the edge still
        # carries the representative document it was built from.
        for (subj, pred, obj, _keyed_evidence), edge in self._edges.items():
            self.db.add_relationship(
                Relationship(subject_id=subj, predicate=pred, object_id=obj,
                             evidence_doc_id=edge.evidence_doc_id,
                             confidence=edge.confidence,
                             inferred=edge.inferred, support=edge.support)
            )

    def _tally(self, stats: GraphStats) -> None:
        """Read the final graph figures back out of the database.

        Read from the tables rather than from the in-memory accumulators, which
        are empty on the skip path but identical everywhere else: ``_flush_edges``
        writes exactly one row per accumulator key, so a group-by over
        ``relationships`` counts the same edges. One source of truth for both
        paths, and it also covers entities that already existed.
        """
        counts = self.db.counts()
        stats.entities = counts["entities"]
        stats.relationships = counts["relationships"]
        for e in self.db.list_entities():
            stats.by_entity_type[e["entity_type"]] = (
                stats.by_entity_type.get(e["entity_type"], 0) + 1
            )
        stats.by_predicate = self.db.relationship_counts_by_predicate()
