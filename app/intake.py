"""Intake — both paths the user chose: manual UPLOAD and server-side MAILBOX.

This normalizes whatever arrives into the `files` list the Submission Intake
Agent expects. Real implementations:
  - UPLOAD : decode multipart files from the /api/nevag BFF route.
  - MAILBOX: pull .eml/.msg + attachments from Outlook/Graph or Blob/S3.
Both produce the same shape, so the brain is source-agnostic.
"""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from typing import Any, Dict, List, Optional, Tuple

from .common import resolve_bytes
from .config import Settings
from .contracts import IntakeMode, NevagInput

# Backwards-compatible alias: byte resolution now lives in app.common (DRY).
_file_bytes = resolve_bytes


def expand_and_fingerprint(files: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Submission Intake governance (agent #1): expand ZIPs into their entries,
    fingerprint every file by SHA-1, and mark duplicates / version lineage. The
    same bytes always produce the same hash -> deterministic, dedup-friendly."""
    out: List[Dict[str, Any]] = []
    lineage: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}

    def _add(f: Dict[str, Any]) -> None:
        data = _file_bytes(f)
        sha = hashlib.sha1(data).hexdigest() if data is not None else None
        rec = dict(f)
        rec["sha1"] = sha
        rec["is_duplicate"] = bool(sha and sha in seen)
        if rec["is_duplicate"]:
            rec["duplicate_of"] = seen[sha]
        elif sha:
            seen[sha] = rec.get("filename")
        out.append(rec)
        lineage.append(
            {
                "filename": rec.get("filename"),
                "sha1": sha,
                "duplicate": rec["is_duplicate"],
                "from_zip": rec.get("from_zip"),
            }
        )

    for f in files:
        data = _file_bytes(f)
        name = (f.get("filename") or "").lower()
        if name.endswith(".zip") and data is not None:
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for zi in z.infolist():
                        if zi.is_dir():
                            continue
                        _add(
                            {
                                "filename": os.path.basename(zi.filename),
                                "content_bytes": z.read(zi),
                                "from_zip": f.get("filename"),
                            }
                        )
                continue
            except Exception:
                pass  # not a valid zip -> treat as a normal file
        _add(f)
    return out, lineage


def gather_files(inp: NevagInput, settings: Settings) -> List[Dict[str, Any]]:
    if inp.mode == IntakeMode.UPLOAD:
        if not settings.upload_enabled:
            raise RuntimeError("Upload intake disabled (NEVAG_UPLOAD_ENABLED=false).")
        return list(inp.files)

    if inp.mode == IntakeMode.MAILBOX:
        if not settings.mailbox_enabled:
            raise RuntimeError("Mailbox intake disabled (NEVAG_MAILBOX_ENABLED=false).")
        return _pull_from_mailbox(inp.source_ref or {}, settings)

    # REFERENCE: files already attached to a known submission_id (loaded elsewhere)
    return list(inp.files)


def _pull_from_mailbox(source_ref: Dict[str, Any], settings: Settings) -> List[Dict[str, Any]]:
    """Server-side ingestion. Uses SharePoint document-library pull when configured
    (ACS app-only). Outlook/Graph/Blob remain future adapters."""
    from .sharepoint import SharePointClient

    client = SharePointClient(settings)
    if client.is_configured():
        return client.pull_files()
    return []
