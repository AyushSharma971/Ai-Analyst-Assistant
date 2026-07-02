"""Explainability & Audit — the reproducible decision record
(architecture doc §5 / the 'explainable, reproducible' pillar).

Compiles a complete, structured trail for a submission: which documents were
parsed (with content hashes), what was extracted and from where, how validation
ruled, what got written to which rater cell, and the counts. The run id is
DERIVED FROM THE INPUTS (not wall-clock), so the same submission yields the same
audit id — satisfying the reproducibility requirement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common import sha1_hex


def _run_id(state: Dict[str, Any]) -> str:
    return "run_" + sha1_hex(
        state.get("submission_id", ""),
        state.get("canonical_record", {}),
        [f.get("name") for f in state.get("extracted_fields", [])],
        length=12,
    )


def build_audit(state: Dict[str, Any], settings: Optional[Any] = None) -> Dict[str, Any]:
    fields = state.get("extracted_fields", [])
    plan = state.get("mapping_plan") or {}
    cells = plan.get("cells", {})

    decisions: List[Dict[str, Any]] = []
    for f in fields:
        decisions.append(
            {
                "attribute": f.get("name"),
                "value": f.get("value"),
                "status": f.get("status"),
                "confidence": f.get("confidence"),
                "extractor": f.get("extractor"),
                "source_document": f.get("source_document"),
                "source_page": f.get("source_page"),
                "evidence_snippet": f.get("evidence_snippet"),
                "retrieval_method": f.get("retrieval_method", "full_text"),
                "citations": f.get("citations", []),
                "reconciliation": f.get("reconciliation"),
                "validation_reasons": f.get("validation_reasons", []),
                "written_to_cell": cells.get(f.get("name"), {}).get("cell"),
                "written": f.get("name") in cells,
            }
        )

    statuses = [d["status"] for d in decisions]
    documents = [
        {
            "filename": d.get("filename"),
            "kind": d.get("kind"),
            "parse_status": d.get("parse_status"),
            "parser": d.get("parser"),
            "sha1": d.get("sha1"),
            "pages": len(d.get("pages", [])) or d.get("page_count", 0),
        }
        for d in state.get("parsed_documents", [])
    ]

    # Deterministic output hash for "same input == same output" verification.
    output_hash = sha1_hex(
        state.get("canonical_record", {}),
        [(d["attribute"], d["value"], d["status"]) for d in decisions],
        state.get("rater_file_url") or "",
    )
    rater_analysis = state.get("rater_analysis") or {}
    versions = {
        "schema_version": state.get("schema_version"),
        "prompt_version": getattr(settings, "prompt_version", None) if settings else None,
        "canonical_schema_version": getattr(settings, "canonical_schema_version", None) if settings else None,
        "rater_template_fingerprint": rater_analysis.get("fingerprint"),
    }

    return {
        "run_id": _run_id(state),
        "submission_id": state.get("submission_id"),
        "output_hash": output_hash,
        "versions": versions,
        "counts": {
            "required": len(state.get("rater_required_attributes", [])),
            "extracted": len(fields),
            "approved": statuses.count("approved"),
            "review": statuses.count("review"),
            "rejected": statuses.count("rejected"),
            "missing": len(state.get("missing_fields", [])),
            "cells_written": len(cells),
        },
        "missing_fields": state.get("missing_fields", []),
        "submission_context": state.get("submission_context"),
        "risk_assessment": state.get("risk_assessment"),
        "rater": {
            "carrier": plan.get("carrier"),
            "sheet": plan.get("sheet"),
            "file_url": state.get("rater_file_url"),
        },
        "documents": documents,
        "decisions": decisions,
        "status_trail": list(state.get("status_trail", [])),
    }
