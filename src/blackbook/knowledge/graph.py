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
from blackbook.storage.database import Database
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

_KIND_TO_ETYPE = {"service": E_SERVICE, "technique": E_TECHNIQUE, "tool": E_TOOL}
_KIND_TO_WRITEUP_PRED = {
    "technique": P_DEMONSTRATED_IN,
    "tool": P_USED_IN,
    "service": P_PRESENT_IN,
}


@dataclass
class GraphStats:
    """Summary of a graph build."""

    entities: int = 0
    relationships: int = 0
    documents: int = 0
    writeups: int = 0
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


class GraphBuilder:
    """Builds the entity/relationship graph from indexed documents."""

    def __init__(self, db: Database):
        self.db = db
        self._entity_ids: dict[tuple[str, str], int] = {}
        # (subject_id, predicate, object_id, evidence_doc_id) -> (confidence, inferred)
        self._edges: dict[tuple[int, str, int, int | None], tuple[float, bool]] = {}

    # -- public API --------------------------------------------------------

    def rebuild(self, source_ids: list[str] | None = None) -> GraphStats:
        """Clear and rebuild the graph from the (optionally scoped) corpus.

        Idempotent: running twice over the same corpus yields the same graph.
        The clear + full rebuild happen in one transaction.
        """
        self._entity_ids.clear()
        self._edges.clear()
        stats = GraphStats()

        with self.db.session():
            self.db.clear_graph()
            for doc in self.db.iter_documents(source_ids):
                stats.documents += 1
                if self._process_document(doc):
                    stats.writeups += 1
            self._flush_edges()

        self._tally(stats)
        log.info(
            "graph rebuilt: %d entities, %d relationships from %d documents",
            stats.entities, stats.relationships, stats.documents,
        )
        return stats

    # -- per-document extraction ------------------------------------------

    def _process_document(self, doc: dict) -> bool:
        """Extract entities/edges for one document. Returns True if a writeup."""
        doc_id = int(doc["doc_id"])
        source_id = str(doc["source_id"])
        title = str(doc["title"])
        categories = _json_list(doc.get("categories"))
        meta = _json_obj(doc.get("metadata"))

        heading_terms, body_terms = self._doc_terms(doc, title)
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

        is_writeup = (
            source_id == "0xdf"
            or is_writeup_category(categories)
            or bool(meta.get("machine_name"))
            or (meta.get("kind") not in (None, "", "unknown"))
        )
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

    def _doc_terms(self, doc: dict, title: str) -> tuple[dict, dict]:
        """Return (heading_terms, body_terms) for a document.

        Heading terms come from the title plus every section-path heading — a
        strong signal the document is *about* the term. Body terms come from the
        chunk text.
        """
        headings: list[str] = []
        body_parts: list[str] = []
        for c in self.db.document_chunks(int(doc["doc_id"])):
            headings.extend(_json_list(c.get("section_path")))
            if c.get("text"):
                body_parts.append(str(c["text"]))
        heading_text = title + "\n" + "\n".join(headings)
        body_text = "\n".join(body_parts)[:_BODY_SCAN_CAP]
        return extract_terms(heading_text), extract_terms(body_text)

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
        separately — each is a distinct citation.
        """
        if subject_id == object_id:
            return  # never relate an entity to itself
        key = (subject_id, predicate, object_id, evidence_doc_id)
        prev = self._edges.get(key)
        if prev is None or confidence > prev[0]:
            # Once any occurrence is structural (inferred=False), keep it so.
            inferred_flag = inferred and (prev is None or prev[1])
            self._edges[key] = (max(confidence, prev[0]) if prev else confidence,
                                inferred_flag)

    def _flush_edges(self) -> None:
        for (subj, pred, obj, evidence), (conf, inferred) in self._edges.items():
            self.db.add_relationship(
                Relationship(subject_id=subj, predicate=pred, object_id=obj,
                             evidence_doc_id=evidence, confidence=conf,
                             inferred=inferred)
            )

    def _tally(self, stats: GraphStats) -> None:
        counts = self.db.counts()
        stats.entities = counts["entities"]
        stats.relationships = counts["relationships"]
        for e in self.db.list_entities():
            stats.by_entity_type[e["entity_type"]] = (
                stats.by_entity_type.get(e["entity_type"], 0) + 1
            )
        for (_s, pred, _o, _e) in self._edges:
            stats.by_predicate[pred] = stats.by_predicate.get(pred, 0) + 1
