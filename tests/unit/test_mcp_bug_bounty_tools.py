from blackbook.config import Settings
from blackbook.mcp.schemas import (
    ContextInput,
    FindingReviewInput,
    HuntPlanInput,
    KnowledgeCompareInput,
    KnowledgeSourceInput,
    ReportDraftInput,
)
from blackbook.mcp.tools import KnowledgeTools


def _tools(seeded_db):
    return KnowledgeTools(seeded_db, Settings())


def test_hunt_plan_returns_cited_plan(seeded_db):
    output = _tools(seeded_db).knowledge_hunt_plan(
        HuntPlanInput(
            observation="Kerberoasting is possible against a Windows service account",
            sources=["hacktricks", "0xdf"],
            limit=2,
        )
    )
    assert output.plans
    assert output.signals.techniques
    assert all(item.references for item in output.plans)
    assert all(item.references[0].ref.chunk_id > 0 for item in output.plans)
    assert "never" in output.note.lower() or "no target" in output.note.lower()


def test_finding_review_does_not_call_documentation_proof(seeded_db):
    output = _tools(seeded_db).knowledge_finding_review(
        FindingReviewInput(
            finding="Kerberoasting may expose crackable service-account tickets",
            sources=["hacktricks"],
        )
    )
    assert output.evidence_status == "documentation_only"
    assert output.missing_evidence
    assert "does not prove" in output.note
    assert all(item.ref.chunk_id > 0 for item in output.references)


def test_report_draft_only_marks_confirmed_case_observations(seeded_db):
    tools = _tools(seeded_db)
    tools.knowledge_context(ContextInput(action="create", case="case-a", target="example.test"))
    tools.knowledge_context(
        ContextInput(
            action="add",
            case="case-a",
            kind="finding",
            text="Kerberoasting was observed in an authorized test.",
        )
    )
    state = tools.knowledge_context(
        ContextInput(action="get", case="case-a")
    )
    obs_id = state.case.observations[0].obs_id
    tools.knowledge_context(
        ContextInput(action="update_observation", case="case-a", obs_id=obs_id, status="confirmed")
    )
    output = tools.knowledge_report_draft(ReportDraftInput(case="case-a", sources=["hacktricks"]))
    assert output.observed_evidence
    assert any("confirmed" in evidence for evidence in output.observed_evidence)
    assert output.warnings
    assert output.references


def test_sources_reports_actual_index_counts(seeded_db):
    output = _tools(seeded_db).knowledge_sources(KnowledgeSourceInput(source="hacktricks"))
    assert output.count == 1
    assert output.sources[0].indexed_documents == 1
    assert output.sources[0].indexed_chunks == 2


def test_compare_keeps_sources_separate_and_rejects_unknown_filter(seeded_db):
    tools = _tools(seeded_db)
    output = tools.knowledge_compare(
        KnowledgeCompareInput(topic="kerberoasting", sources=["hacktricks", "0xdf"])
    )
    assert output.sources_compared == ["hacktricks", "0xdf"]
    assert len(output.views) == 2
    assert all(item.ref.source == view.source for view in output.views for item in view.results)

    empty = tools.knowledge_compare(
        KnowledgeCompareInput(topic="kerberoasting", sources=["not-a-source", "also-not-a-source"])
    )
    assert empty.views == []
    assert "nothing was searched" in empty.note
