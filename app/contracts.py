"""I/O contracts for the nebag agent.

Two layers, deliberately separated:

1. INTERNAL contract (nebagInput / nebagResult) — what the nebag brain speaks.
   Stable, host-agnostic. Everything inside the service uses these.

2. HOST envelope (One AI data-agent contract) — what the One AI chatbot expects.
   Defined in `oneai_*` helpers below. This is the ONLY place coupled to the
   host. To support a different host later, add another mapping; the brain never
   changes.

One AI data-agent response shape (from the repo README §11 / §10.4):
    { "result": { "response": <str | {table_header, table_data} | {chartTitle,...}>,
                  "sql_result"?: ..., "token_data"?: ... } }
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Internal domain models
# --------------------------------------------------------------------------- #
class FieldStatus(str, Enum):
    APPROVED = "approved"        # evidence-backed, safe to autofill (green)
    REVIEW = "review"            # low confidence / contradiction (yellow)
    REJECTED = "rejected"        # failed validation (red)
    MISSING = "missing"          # required by rater, not found
    OCR_FAILED = "ocr_failed"    # found but OCR unreliable


class ExtractedField(BaseModel):
    """A single rater attribute with its evidence trail (governance core)."""

    name: str                                  # canonical name, e.g. annual_revenue
    value: Optional[Any] = None
    confidence: float = 0.0
    status: FieldStatus = FieldStatus.MISSING
    source_document: Optional[str] = None      # filename / doc id
    source_page: Optional[int] = None
    evidence_snippet: Optional[str] = None
    notes: Optional[str] = None


class IntakeMode(str, Enum):
    UPLOAD = "upload"      # files came via the custom UI / multipart route
    MAILBOX = "mailbox"    # pulled server-side from mailbox / blob
    REFERENCE = "reference"  # query references an existing submission_id


class nebagInput(BaseModel):
    """The stable internal entry contract for the nebag brain."""

    query: str = ""                                   # chat text / instruction
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    submission_id: Optional[str] = None               # set for resume / reference
    mode: IntakeMode = IntakeMode.REFERENCE
    # For UPLOAD mode: list of {filename, content_b64 | path, content_type}
    files: List[Dict[str, Any]] = Field(default_factory=list)
    # For MAILBOX mode: where to pull from (folder, message id, broker, etc.)
    source_ref: Optional[Dict[str, Any]] = None
    app_user_id: Optional[str] = None                 # carry attribution through


class nebagResult(BaseModel):
    """The stable internal result contract."""

    submission_id: str
    summary: str = ""
    fields: List[ExtractedField] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    review_required: bool = False
    rater_file_url: Optional[str] = None              # filled workbook location
    audit_run_id: Optional[str] = None
    submission_context: Optional[Dict[str, Any]] = None
    confirmed_context: Optional[Dict[str, Any]] = None
    classification: Optional[Dict[str, Any]] = None
    graph_facts: List[Dict[str, Any]] = Field(default_factory=list)
    gap_report: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    review_workbench: Optional[Dict[str, Any]] = None
    learning_signals: Optional[Dict[str, Any]] = None
    retrieval: Optional[Dict[str, Any]] = None        # vector-store ingestion report
    metrics: Optional[Dict[str, Any]] = None          # timings + counts (observability)
    audit: Optional[Dict[str, Any]] = None            # full reproducible decision record
    status_trail: List[str] = Field(default_factory=list)  # node-by-node progress


# --------------------------------------------------------------------------- #
# One AI host envelope mapping (the only host-coupled code)
# --------------------------------------------------------------------------- #
def to_oneai_envelope(result: nebagResult, as_table: bool = True) -> Dict[str, Any]:
    """Map an internal nebagResult to the One AI data-agent response envelope.

    Default (generic-runner-first): return extracted fields as a TABLE so the
    existing AG-Table renderer shows them, with the human summary attached in
    token_data for context.
    """
    if as_table:
        table_header = ["Field", "Value", "Confidence", "Source", "Status"]
        table_data = [
            [
                f.name,
                "" if f.value is None else str(f.value),
                round(f.confidence, 2),
                f.source_document or "",
                f.status.value,
            ]
            for f in result.fields
        ]
        response: Any = {"table_header": table_header, "table_data": table_data}
    else:
        response = result.summary

    token_data: Dict[str, Any] = {
        "submission_id": result.submission_id,
        "summary": result.summary,
        "missing_fields": result.missing_fields,
        "review_required": result.review_required,
        "rater_file_url": result.rater_file_url,
        "audit_run_id": result.audit_run_id,
        "risk_assessment": result.risk_assessment,
        "submission_context": result.submission_context,
    }

    # Filled-rater delivery (design checklist): when the workbook exists, expose it
    # as a downloadable link in token_data. The host can render this as a download
    # button. For inline-bytes delivery instead, the host's <<FILE_DATA>> convention
    # can be populated from this same link.
    if result.rater_file_url:
        token_data["file"] = {
            "filename": f"{result.submission_id}_filled.xlsx",
            "url": result.rater_file_url,
            "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }

    return {"result": {"response": response, "token_data": token_data}}


class OneAIRequest(BaseModel):
    """The One AI data-agent request body: { chat_history, query }."""

    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    query: str = ""
    # Optional extensions nebag understands (ignored by generic runner):
    submission_id: Optional[str] = None
    app_user_id: Optional[str] = None
