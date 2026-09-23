"""MCP prompts: reusable framings for the questions this corpus can answer.

A prompt is a starting message an MCP client can offer the user. These exist
because the failure mode with a knowledge server is not "no answer", it is a
fluent answer the corpus never supported. Every template here therefore says
the same thing in a different shape: call the tools, cite what comes back, and
mark the gap where nothing did.

They are deliberately short. A long template would add instructions the tools
already enforce (structured provenance, bounded hypotheses, no execution), and
text an agent might follow instead of the tool contract.
"""

from __future__ import annotations


def triage_observation(observation: str, target: str = "") -> str:
    """Turn a raw observation into a source-grounded triage starting point."""
    where = f" Target: {target}." if target.strip() else ""
    return (
        f"I have this observation from an engagement:{where}\n\n"
        f"{observation.strip()}\n\n"
        "Work through it with BlackBook, in this order:\n"
        "1. knowledge_research on the observation text, to get the detected "
        "signals, the techniques it maps to, and their cited references.\n"
        "2. knowledge_case_search for hands-on writeups of the same situation, "
        "to see how it has actually played out.\n"
        "3. knowledge_hunt_plan for bounded validation hypotheses.\n\n"
        "Then tell me what the corpus supports and what it does not. Cite the "
        "technique, document and section behind each claim. Where a step is a "
        "hypothesis rather than something documented, say so plainly instead of "
        "writing it as fact. Do not run anything against a system: the plan is "
        "for me to execute against a target I am authorised to test."
    )


def explain_technique(technique: str, platform: str = "") -> str:
    """Explain a technique strictly from what the indexed sources document."""
    on = f" on {platform}" if platform.strip() else ""
    return (
        f"Explain the technique {technique!r}{on} using only what BlackBook "
        "documents.\n\n"
        "Start with knowledge_technique to get the dossier: which sources "
        "document it, the tools and services the graph associates with it, and "
        "the cited excerpts. Use knowledge_source to pull the exact passage "
        "behind anything you want to state.\n\n"
        "Cover what it is, what it needs, what it gets you, and how it is "
        "detected or defended against, but only where a source says so. If the "
        "index is thin on a part of that, say which part is thin rather than "
        "filling it in from general knowledge. Label anything you add from "
        "outside the corpus as outside the corpus. Do not provide runnable "
        "commands."
    )


def review_finding(finding: str) -> str:
    """Review a suspected finding for evidence gaps before it is reported."""
    return (
        "Review this suspected finding before I report it:\n\n"
        f"{finding.strip()}\n\n"
        "Run knowledge_finding_review to check it against the indexed guidance "
        "and tell me what proof is missing. Then use knowledge_search and "
        "knowledge_case_search to see whether the corpus documents this class "
        "of issue and what it usually takes to make one stick.\n\n"
        "Be specific about the gap between what I have observed and what I "
        "would need to demonstrate. Tell me if it is not yet a finding. A "
        "would-be report that triage rejects as informational is worse than "
        "one I held back, so err toward naming the missing evidence."
    )


def draft_report(case: str) -> str:
    """Draft a cautious report from an investigation recorded locally."""
    return (
        f"Draft a report from my local BlackBook case {case!r}.\n\n"
        "Run knowledge_report_draft on that case. It will include the "
        "observations I have recorded, reproducibility prompts, cited guidance, "
        "and explicit warnings wherever evidence is missing.\n\n"
        "Keep those warnings. Write the report so that every claim traces to "
        "something in the case or to a citation, and so that anything I have "
        "not confirmed reads as unconfirmed rather than as established. Do not "
        "upgrade an observation's status on my behalf. If the case is too thin "
        "to support a report, tell me what to go and record instead of writing "
        "around it."
    )
