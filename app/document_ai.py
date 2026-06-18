"""Document AI — real document parsing (architecture doc §5 step 1).

Turns the raw `files` list (whatever intake produced) into structured
`parsed_documents` that Semantic Extraction can mine for rater attributes.

Design contract (same philosophy as the rest of the service):
  - Heavy parser libraries are imported LAZILY and degrade GRACEFULLY. A missing
    wheel (e.g. on a bleeding-edge Python) or a missing OCR binary downgrades a
    document to a clear `parse_status` + warning instead of crashing the pipeline.
  - Every parser returns the SAME ParsedDocument dict shape, so the brain is
    format-agnostic and Semantic Extraction has one structure to read.
  - Page/section granularity is preserved so downstream extraction can cite
    `source_document` + `source_page` + an `evidence_snippet` (governance core).
  - A content `sha1` is recorded per document for the reproducible audit trail.

Formats handled: digital PDF, scanned PDF / image (OCR), Excel, Word, email
(.eml via stdlib; .msg via optional extract-msg), with email attachments parsed
recursively into the same shape.
"""

from __future__ import annotations

import hashlib
import io
import os
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .common import resolve_bytes
from .config import Settings

# --------------------------------------------------------------------------- #
# ParsedDocument shape (plain dicts to match WorkflowState style)
# --------------------------------------------------------------------------- #
# {
#   "filename": str,
#   "content_type": str | None,
#   "kind": "digital_pdf"|"scanned_pdf"|"image"|"excel"|"word"|"email"|"unknown",
#   "parser": str,                 # which library did the work
#   "parse_status": "ok"|"partial"|"failed"|"skipped",
#   "text": str,                   # full concatenated plain text
#   "pages": [{"page": int, "text": str, "ocr": bool, "tables": [[...]]}],
#   "tables": [{"page": int, "rows": [[...]]}],   # convenience: all tables
#   "sheets": {sheet_name: [[...]]},              # excel only
#   "metadata": {...},
#   "warnings": [str],
#   "attachments": [ParsedDocument, ...],         # email only
#   "size_bytes": int,
#   "sha1": str,
# }

_PDF_EXTS = {".pdf"}
_EXCEL_EXTS = {".xlsx", ".xlsm", ".xls"}
_WORD_EXTS = {".docx", ".doc"}
_EMAIL_EXTS = {".eml", ".msg"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


def _new_doc(filename: str, content_type: Optional[str]) -> Dict[str, Any]:
    return {
        "filename": filename,
        "content_type": content_type,
        "kind": "unknown",
        "parser": "none",
        "parse_status": "skipped",
        "text": "",
        "pages": [],
        "tables": [],
        "sheets": {},
        "metadata": {},
        "warnings": [],
        "attachments": [],
        "size_bytes": 0,
        "sha1": "",
    }


# --------------------------------------------------------------------------- #
# OCR engines (pluggable; degrade to NullOCR when nothing is available)
# --------------------------------------------------------------------------- #
class OCREngine(Protocol):
    name: str
    available: bool

    def image_to_text(self, image_png: bytes) -> "Tuple[str, Optional[float]]":
        """Return (text, confidence in 0..1 or None when unknown)."""
        ...


class NullOCR:
    """No OCR backend present. Detection still works; pages are flagged."""

    name = "none"
    available = False

    def image_to_text(self, image_png: bytes):  # noqa: ARG002
        return "", 0.0


class TesseractOCR:
    """Local OCR via pytesseract. Requires both the `pytesseract` package and
    the Tesseract binary on PATH (or nebag_TESSERACT_CMD)."""

    name = "tesseract"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._pt = None
        self.available = self._probe()

    def _probe(self) -> bool:
        try:
            import pytesseract  # lazy
        except Exception:
            return False
        cmd = getattr(self._settings, "tesseract_cmd", None)
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            return False
        self._pt = pytesseract
        return True

    def image_to_text(self, image_png: bytes):
        from PIL import Image  # lazy
        with Image.open(io.BytesIO(image_png)) as img:
            text = self._pt.image_to_string(img) or ""  # keeps line layout
            conf = self._mean_conf(img)                  # word-level confidence
        return text, conf

    def _mean_conf(self, img) -> Optional[float]:
        try:
            from pytesseract import Output

            data = self._pt.image_to_data(img, output_type=Output.DICT)
            confs = []
            for word, c in zip(data.get("text", []), data.get("conf", [])):
                if not str(word).strip():
                    continue
                try:
                    cv = float(c)
                except Exception:
                    continue
                if cv >= 0:
                    confs.append(cv)
            return (sum(confs) / len(confs) / 100.0) if confs else None
        except Exception:
            return None


class AzureDocIntelOCR:
    """OCR via Azure AI Document Intelligence — the One AI org's cloud standard.
    Requires `azure-ai-documentintelligence` + nebag_AZURE_DOC_INTEL_* config."""

    name = "azure_doc_intel"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = None
        self.available = bool(
            getattr(settings, "azure_doc_intel_endpoint", None)
            and getattr(settings, "azure_doc_intel_key", None)
        ) and self._probe()

    def _probe(self) -> bool:
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient  # noqa: F401
        except Exception:
            return False
        return True

    def _ensure_client(self):
        if self._client is not None:
            return
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        s = self._settings
        self._client = DocumentIntelligenceClient(
            endpoint=s.azure_doc_intel_endpoint,
            credential=AzureKeyCredential(s.azure_doc_intel_key),
        )

    def image_to_text(self, image_png: bytes):
        self._ensure_client()
        poller = self._client.begin_analyze_document(
            "prebuilt-read", body=image_png, content_type="image/png"
        )
        result = poller.result()
        return (result.content or ""), None  # Azure DI confidence not surfaced here


def build_ocr_engine(settings: Settings) -> OCREngine:
    """Factory: choose OCR backend by config, with sensible auto-detection.

    nebag_OCR_PROVIDER: auto (default) | azure_doc_intel | tesseract | none
    """
    provider = (getattr(settings, "ocr_provider", "auto") or "auto").lower()
    if provider == "none":
        return NullOCR()
    if provider == "azure_doc_intel":
        return AzureDocIntelOCR(settings)
    if provider == "tesseract":
        return TesseractOCR(settings)
    # auto: prefer cloud (org standard) if configured; cascade only if fallback on.
    azure = AzureDocIntelOCR(settings)
    if azure.available:
        return azure
    if getattr(settings, "ocr_enable_fallback", True):
        tess = TesseractOCR(settings)
        if tess.available:
            return tess
    return NullOCR()


# --------------------------------------------------------------------------- #
# Byte access + format detection
# --------------------------------------------------------------------------- #
def _ext(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


def detect_kind(filename: str, content_type: Optional[str], data: bytes) -> str:
    """Classify by extension, then content-type, then magic bytes. PDFs are
    refined into digital vs. scanned later, once text extraction is attempted."""
    ext = _ext(filename)
    if ext in _PDF_EXTS:
        return "pdf"  # refined to digital_pdf / scanned_pdf during parse
    if ext in _EXCEL_EXTS:
        return "excel"
    if ext in _WORD_EXTS:
        return "word"
    if ext in _EMAIL_EXTS:
        return "email"
    if ext in _IMAGE_EXTS:
        return "image"

    ct = (content_type or "").lower()
    if "pdf" in ct:
        return "pdf"
    if "spreadsheet" in ct or "excel" in ct:
        return "excel"
    if "word" in ct or "officedocument.wordprocessing" in ct:
        return "word"
    if "message/rfc822" in ct or ct == "application/vnd.ms-outlook":
        return "email"
    if ct.startswith("image/"):
        return "image"

    # Magic bytes as a last resort.
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        # OOXML container; can't tell xlsx vs docx without peeking — default word
        return "word"
    return "unknown"


# --------------------------------------------------------------------------- #
# Per-format parsers
# --------------------------------------------------------------------------- #
def _parse_pdf(doc: Dict[str, Any], data: bytes, ocr: OCREngine, settings: Settings) -> None:
    try:
        import fitz  # PyMuPDF, lazy
    except Exception as exc:  # pragma: no cover - missing wheel path
        doc["parse_status"] = "failed"
        doc["warnings"].append(f"PyMuPDF unavailable: {exc}")
        return

    threshold = int(settings.scanned_text_threshold)
    dpi = int(settings.ocr_dpi)

    try:
        pdf = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        doc["parse_status"] = "failed"
        doc["warnings"].append(f"could not open PDF: {exc}")
        return

    # pdfplumber for table extraction (optional; text comes from PyMuPDF).
    plumber_pages = None
    try:
        import pdfplumber  # lazy

        plumber_pdf = pdfplumber.open(io.BytesIO(data))
        plumber_pages = plumber_pdf.pages
    except Exception as exc:
        doc["warnings"].append(f"table extraction unavailable: {exc}")

    scanned_pages = 0
    ocr_used = False
    parts: List[str] = []
    with pdf:
        doc["metadata"] = {"page_count": pdf.page_count, **(pdf.metadata or {})}
        for i, page in enumerate(pdf):
            text = (page.get_text("text") or "").strip()
            is_scanned = len(text) < threshold
            page_ocr = False
            ocr_conf: Optional[float] = None
            if is_scanned:
                scanned_pages += 1
                if ocr.available:
                    try:
                        pix = page.get_pixmap(dpi=dpi)
                        ocr_text, ocr_conf = ocr.image_to_text(pix.tobytes("png"))
                        text = (ocr_text or "").strip()
                        page_ocr = bool(text)
                        ocr_used = ocr_used or page_ocr
                    except Exception as exc:
                        doc["warnings"].append(f"OCR failed on page {i + 1}: {exc}")
                else:
                    doc["warnings"].append(
                        f"page {i + 1} appears scanned but no OCR engine available "
                        f"(provider='{ocr.name}')"
                    )

            tables: List[List[List[Any]]] = []
            if plumber_pages and i < len(plumber_pages):
                try:
                    tables = plumber_pages[i].extract_tables() or []
                except Exception:
                    tables = []
            for rows in tables:
                doc["tables"].append({"page": i + 1, "rows": rows})

            doc["pages"].append({"page": i + 1, "text": text, "ocr": page_ocr,
                                 "ocr_confidence": ocr_conf, "tables": tables})
            if text:
                parts.append(text)

    doc["text"] = "\n\n".join(parts)
    page_count = doc["metadata"].get("page_count", len(doc["pages"]))
    if scanned_pages == 0:
        doc["kind"] = "digital_pdf"
        doc["parser"] = "pymupdf"
        doc["parse_status"] = "ok"
    elif scanned_pages >= page_count:
        doc["kind"] = "scanned_pdf"
        doc["parser"] = f"pymupdf+ocr:{ocr.name}" if ocr_used else "pymupdf"
        doc["parse_status"] = "ok" if ocr_used else "partial"
    else:
        doc["kind"] = "digital_pdf"  # mixed; mostly digital
        doc["parser"] = f"pymupdf+ocr:{ocr.name}" if ocr_used else "pymupdf"
        doc["parse_status"] = "ok" if (ocr_used or scanned_pages == 0) else "partial"


def _parse_image(doc: Dict[str, Any], data: bytes, ocr: OCREngine) -> None:
    doc["kind"] = "image"
    if not ocr.available:
        doc["parse_status"] = "partial"
        doc["warnings"].append(f"image needs OCR but no engine available (provider='{ocr.name}')")
        doc["pages"].append({"page": 1, "text": "", "ocr": False, "ocr_confidence": None, "tables": []})
        return
    try:
        ocr_text, ocr_conf = ocr.image_to_text(data)
        text = (ocr_text or "").strip()
    except Exception as exc:
        doc["parse_status"] = "failed"
        doc["warnings"].append(f"OCR failed: {exc}")
        return
    doc["text"] = text
    doc["pages"].append({"page": 1, "text": text, "ocr": True, "ocr_confidence": ocr_conf, "tables": []})
    doc["parser"] = f"ocr:{ocr.name}"
    doc["parse_status"] = "ok" if text else "partial"


def _parse_excel(doc: Dict[str, Any], data: bytes) -> None:
    try:
        import openpyxl  # lazy
    except Exception as exc:
        doc["parse_status"] = "failed"
        doc["warnings"].append(f"openpyxl unavailable: {exc}")
        return
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:
        doc["parse_status"] = "failed"
        doc["warnings"].append(f"could not open workbook: {exc}")
        return

    doc["kind"] = "excel"
    doc["parser"] = "openpyxl"
    parts: List[str] = []
    page = 0
    for ws in wb.worksheets:
        page += 1
        rows: List[List[Any]] = []
        for row in ws.iter_rows(values_only=True):
            if any(c is not None for c in row):
                rows.append([("" if c is None else c) for c in row])
        doc["sheets"][ws.title] = rows
        doc["tables"].append({"page": page, "sheet": ws.title, "rows": rows})
        # Flatten each sheet to text so the LLM/extractor can read it linearly.
        sheet_text = "\n".join("\t".join(str(c) for c in r) for r in rows)
        doc["pages"].append({"page": page, "sheet": ws.title, "text": sheet_text, "ocr": False, "tables": [rows]})
        if sheet_text:
            parts.append(f"# Sheet: {ws.title}\n{sheet_text}")
    wb.close()
    doc["text"] = "\n\n".join(parts)
    doc["metadata"] = {"sheet_count": len(doc["sheets"])}
    doc["parse_status"] = "ok"


def _parse_word(doc: Dict[str, Any], data: bytes) -> None:
    try:
        import docx  # python-docx, lazy
    except Exception as exc:
        doc["parse_status"] = "failed"
        doc["warnings"].append(f"python-docx unavailable: {exc}")
        return
    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as exc:
        doc["parse_status"] = "failed"
        doc["warnings"].append(f"could not open document (legacy .doc not supported): {exc}")
        return

    doc["kind"] = "word"
    doc["parser"] = "python-docx"
    paras = [p.text for p in d.paragraphs if p.text and p.text.strip()]
    tables: List[List[List[Any]]] = []
    for t in d.tables:
        rows = [[c.text for c in row.cells] for row in t.rows]
        tables.append(rows)
        doc["tables"].append({"page": 1, "rows": rows})
    text = "\n".join(paras)
    doc["text"] = text
    doc["pages"].append({"page": 1, "text": text, "ocr": False, "tables": tables})
    doc["metadata"] = {"paragraphs": len(paras), "tables": len(tables)}
    doc["parse_status"] = "ok"


def _parse_email(doc: Dict[str, Any], data: bytes, ctx: "ParseContext") -> None:
    ext = _ext(doc["filename"])
    if ext == ".msg":
        _parse_outlook_msg(doc, data, ctx)
        return
    import email
    from email import policy

    try:
        msg = email.message_from_bytes(data, policy=policy.default)
    except Exception as exc:
        doc["parse_status"] = "failed"
        doc["warnings"].append(f"could not parse email: {exc}")
        return

    doc["kind"] = "email"
    doc["parser"] = "stdlib-email"
    headers = {
        "from": str(msg.get("From", "")),
        "to": str(msg.get("To", "")),
        "subject": str(msg.get("Subject", "")),
        "date": str(msg.get("Date", "")),
    }
    doc["metadata"] = headers

    body = ""
    try:
        body_part = msg.get_body(preferencelist=("plain", "html"))
        if body_part is not None:
            body = body_part.get_content() or ""
    except Exception:
        body = ""

    header_text = "\n".join(f"{k.title()}: {v}" for k, v in headers.items() if v)
    full = (header_text + "\n\n" + body).strip()
    doc["text"] = full
    doc["pages"].append({"page": 1, "text": full, "ocr": False, "tables": []})

    # Recurse into attachments — they're real submission documents too.
    for part in msg.iter_attachments():
        fname = part.get_filename() or "attachment"
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        child = parse_document(
            {"filename": fname, "content_type": part.get_content_type(), "content_bytes": payload},
            ctx,
        )
        doc["attachments"].append(child)
    doc["parse_status"] = "ok"


def _parse_outlook_msg(doc: Dict[str, Any], data: bytes, ctx: "ParseContext") -> None:
    try:
        import extract_msg  # lazy, optional
    except Exception as exc:
        doc["parse_status"] = "failed"
        doc["warnings"].append(f"extract-msg unavailable for .msg: {exc}")
        return
    try:
        m = extract_msg.Message(io.BytesIO(data))
    except Exception as exc:
        doc["parse_status"] = "failed"
        doc["warnings"].append(f"could not parse .msg: {exc}")
        return
    doc["kind"] = "email"
    doc["parser"] = "extract-msg"
    headers = {"from": m.sender or "", "to": m.to or "", "subject": m.subject or "", "date": str(m.date or "")}
    doc["metadata"] = headers
    body = m.body or ""
    header_text = "\n".join(f"{k.title()}: {v}" for k, v in headers.items() if v)
    full = (header_text + "\n\n" + body).strip()
    doc["text"] = full
    doc["pages"].append({"page": 1, "text": full, "ocr": False, "tables": []})
    for att in getattr(m, "attachments", []) or []:
        fname = getattr(att, "longFilename", None) or getattr(att, "shortFilename", None) or "attachment"
        payload = getattr(att, "data", None)
        if not payload:
            continue
        child = parse_document({"filename": fname, "content_bytes": payload}, ctx)
        doc["attachments"].append(child)
    doc["parse_status"] = "ok"


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
class ParseContext:
    """Lightweight bundle of parsing dependencies (OCR + settings)."""

    def __init__(self, settings: Settings, ocr: Optional[OCREngine] = None):
        self.settings = settings
        self.ocr = ocr or build_ocr_engine(settings)


def parse_document(file: Dict[str, Any], ctx: ParseContext) -> Dict[str, Any]:
    """Parse a single file dict into the ParsedDocument shape. Never raises for
    expected failure modes — records `parse_status` + `warnings` instead."""
    filename = file.get("filename") or "unnamed"
    doc = _new_doc(filename, file.get("content_type"))

    data = resolve_bytes(file)
    if data is None:
        doc["parse_status"] = "skipped"
        doc["warnings"].append("no content (no content_b64 / path / content_bytes)")
        return doc

    doc["size_bytes"] = len(data)
    doc["sha1"] = hashlib.sha1(data).hexdigest()
    kind = detect_kind(filename, file.get("content_type"), data)

    try:
        if kind == "pdf":
            _parse_pdf(doc, data, ctx.ocr, ctx.settings)
        elif kind == "image":
            _parse_image(doc, data, ctx.ocr)
        elif kind == "excel":
            _parse_excel(doc, data)
        elif kind == "word":
            _parse_word(doc, data)
        elif kind == "email":
            _parse_email(doc, data, ctx)
        else:
            doc["parse_status"] = "skipped"
            doc["warnings"].append(f"unsupported/unknown format (kind='{kind}')")
    except Exception as exc:  # last-resort guard; one bad file never kills the run
        doc["parse_status"] = "failed"
        doc["warnings"].append(f"unexpected parse error: {type(exc).__name__}: {exc}")

    return doc


def parse_documents(
    files: List[Dict[str, Any]], settings: Settings, ocr: Optional[OCREngine] = None
) -> List[Dict[str, Any]]:
    """Parse the whole submission. Returns one ParsedDocument per input file.

    Per-file parsing can run in parallel (nebag_DOC_AI_MAX_WORKERS > 1); results are
    reassembled in INPUT ORDER, so output is identical regardless of worker count
    (deterministic). Default 1 = sequential."""
    ctx = ParseContext(settings, ocr)
    workers = int(getattr(settings, "doc_ai_max_workers", 1) or 1)
    if workers <= 1 or len(files) <= 1:
        return [parse_document(f, ctx) for f in files]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda f: parse_document(f, ctx), files))  # preserves order
