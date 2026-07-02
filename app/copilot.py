"""Underwriter Copilot (architecture doc §6 agent #17).

Answers evidence-backed questions about a processed submission. RAG-style but
grounded in the agent's own structured outputs (extracted fields + evidence,
missing list, risk, audit) — never fabricates. If nothing grounds the answer it
says so. An LLM can be layered for phrasing, but answers are credential-free.

Governance: the copilot cannot override validated data; it only reports it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def answer(
    question: str,
    state: Dict[str, Any],
    llm: Optional[Any] = None,
    retrieved: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    q = (question or "").lower().strip()
    fields: List[Dict[str, Any]] = state.get("extracted_fields", [])

    # 1. Missing-field questions
    if "missing" in q or "what's left" in q or "incomplete" in q:
        missing = state.get("missing_fields", [])
        return _resp(
            ("Missing/required-but-absent: " + ", ".join(missing)) if missing else "Nothing missing.",
            evidence=[{"missing_fields": missing}],
        )

    # 2. Risk questions
    if "risk" in q:
        risk = state.get("risk_assessment") or {}
        return _resp(
            f"Risk level: {risk.get('level', 'unknown')}. {risk.get('rationale', '')}",
            evidence=[risk],
        )

    # 3. Attribute value lookup (grounded with evidence)
    for f in fields:
        name = f.get("name", "")
        if name and (name in q or name.replace("_", " ") in q):
            return _resp(
                f"{name} = {f.get('value')} "
                f"(status {f.get('status')}, confidence {f.get('confidence')}, "
                f"source {f.get('source_document')} p{f.get('source_page')})",
                evidence=[
                    {
                        "attribute": name,
                        "value": f.get("value"),
                        "evidence_snippet": f.get("evidence_snippet"),
                        "source_document": f.get("source_document"),
                    }
                ],
            )

    # 4. Status / summary
    if "status" in q or "summary" in q or "overview" in q:
        audit = state.get("audit") or {}
        return _resp(
            f"Counts: {audit.get('counts', {})}. Rater: {audit.get('rater', {})}.",
            evidence=[audit.get("counts", {})],
        )

    # 5. Fall back to retrieved chunks (vector search) — still grounded in evidence.
    if retrieved:
        top = retrieved[0]
        payload = top.get("payload", {})
        return _resp(
            f"Most relevant passage ({payload.get('file_name')} p{payload.get('page_number')}): "
            f"{payload.get('evidence_snippet') or payload.get('text', '')}",
            evidence=[
                {
                    "chunk_id": payload.get("chunk_id"),
                    "file_name": payload.get("file_name"),
                    "page_number": payload.get("page_number"),
                    "score": top.get("score"),
                    "snippet": payload.get("evidence_snippet"),
                }
                for top in retrieved[:3]
                for payload in [top.get("payload", {})]
            ],
        )

    # 6. No grounded answer (governance: don't guess)
    return _resp(
        "No grounded answer found in this submission's evidence. Try asking about a "
        "specific attribute, what's missing, or the risk summary.",
        evidence=[],
    )


def _resp(text: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"answer": text, "evidence": evidence, "grounded": bool(evidence)}
