"""Run a REAL submission through the full pipeline on Azure GPT-4.1 + ada-002 +
Tesseract. Reports structured extraction + provenance + metrics (no raw-document
dump). Usage: python scripts/run_real_submission.py "<path-to-.eml-or-file>"
"""

from __future__ import annotations

import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings, _env_overlay
from app.contracts import IntakeMode, nebagInput
from app.service import nebagAgent


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else (glob.glob("*.eml") or [None])[0]
    if not arg or not os.path.exists(arg):
        print("No submission file found.")
        return

    host = _env_overlay()
    azure = bool(host.get("azure_openai_api_key") and host.get("azure_openai_endpoint"))
    base = dict(host)
    base.update(
        llm_provider="azure_openai" if azure else "mock",
        embedding_provider="azure_openai" if azure else "mock",
        embedding_dim=1536,
        ocr_provider="tesseract",
        tesseract_cmd=host.get("tesseract_cmd") or r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        rag_extraction_enabled=azure,
        retrieval_min_score=0.0,
        rater_output_dir=os.path.join(os.getcwd(), "output"),
    )
    s = Settings(**base)
    print(f"ENV: llm={s.llm_provider} emb={s.embedding_provider}(dim {s.embedding_dim}) ocr={s.ocr_provider} rag={s.rag_extraction_enabled}")
    print(f"FILE: {os.path.basename(arg)} ({os.path.getsize(arg)} bytes)\n")

    t0 = time.time()
    agent = nebagAgent(settings=s)
    res = agent.run(nebagInput(query="Process this D&O submission.", mode=IntakeMode.UPLOAD,
                               files=[{"filename": os.path.basename(arg), "path": arg}]))
    elapsed = time.time() - t0

    print("=== DOCUMENT AI ===")
    for line in res.status_trail:
        if "document_ai" in line or "intake" in line:
            print(" ", line)
    a = res.audit or {}
    for d in a.get("documents", []):
        print(f"  doc: {d.get('filename')} kind={d.get('kind')} status={d.get('parse_status')} "
              f"parser={d.get('parser')} pages={d.get('pages')}")

    print("\n=== EXTRACTION (real GPT-4.1) ===")
    print(f"required attributes (rater-driven/catalog): {len(res.fields) + len(res.missing_fields or [])}")
    for f in res.fields:
        print(f"  {f.status.value:9s} {f.name}: {f.value!r} (conf {f.confidence}, "
              f"src {f.source_document} p{f.source_page}, via {f.notes or ''})")
    print(f"  MISSING: {res.missing_fields}")
    print(f"  gap_report: {res.gap_report}")

    print("\n=== CONTEXT / CLASSIFICATION / RISK ===")
    print("  submission_context:", res.submission_context)
    print("  classification:", res.classification)
    print("  risk:", (res.risk_assessment or {}).get("level"), (res.risk_assessment or {}).get("flags"))

    print("\n=== OUTCOME ===")
    print(f"  summary: {res.summary}")
    print(f"  review_required: {res.review_required}  rater_file_url: {res.rater_file_url}")
    print(f"  retrieval: {res.retrieval}")
    print(f"  wall_time: {elapsed:.1f}s  metrics.counts: {(res.metrics or {}).get('counts')}")


if __name__ == "__main__":
    main()
