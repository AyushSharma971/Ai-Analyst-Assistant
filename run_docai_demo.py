"""Document AI demo — proves the real parsers on synthetic fixtures.

No server, no credentials. Generates one sample per format, parses each through
app/document_ai.py, prints what was extracted, then runs the full nebagAgent
pipeline on the same files so you can see Document AI in the status trail.

    python run_docai_demo.py
"""

from __future__ import annotations

from app.config import Settings
from app.contracts import IntakeMode, nebagInput
from app.document_ai import build_ocr_engine, parse_documents
from app.service import nebagAgent
from scripts.make_fixtures import build_all


def _snippet(text: str, n: int = 120) -> str:
    one_line = " ".join(text.split())
    return (one_line[:n] + "...") if len(one_line) > n else one_line


def main() -> None:
    paths = build_all()
    rater = paths.pop("rater_template", None)  # a template, not a submission doc
    files = [{"filename": __basename(p), "path": p} for p in paths.values()]

    # Point autofill at the fixture rater so the full Start->End path writes it.
    settings = Settings(llm_provider="mock", rater_template_path=rater)
    ocr = build_ocr_engine(settings)

    print("=== OCR engine ===")
    print(f"  provider resolved to: {ocr.name} (available={ocr.available})")
    if not ocr.available:
        print("  -> scanned/image docs will be detected & routed, then flagged "
              "(install Tesseract or set nebag_AZURE_DOC_INTEL_* to extract text).")

    print("\n=== Per-document parse results ===")
    docs = parse_documents(files, settings, ocr)
    for d in docs:
        print(f"\n* {d['filename']}")
        print(f"    kind={d['kind']}  status={d['parse_status']}  parser={d['parser']}")
        print(f"    pages={len(d['pages'])}  tables={len(d['tables'])}  "
              f"size={d['size_bytes']}B  sha1={d['sha1'][:12]}")
        if d["text"]:
            print(f"    text: {_snippet(d['text'])}")
        for w in d["warnings"]:
            print(f"    ! {w}")
        for att in d["attachments"]:
            print(f"    >> attachment: {att['filename']}  kind={att['kind']}  "
                  f"status={att['parse_status']}  text={_snippet(att['text'], 60)!r}")

    print("\n=== Full pipeline (Start->End) with real Document AI ===")
    agent = nebagAgent(settings=settings)
    result = agent.run(
        nebagInput(
            query="Process this new D&O submission from Acme Health Systems.",
            mode=IntakeMode.UPLOAD,
            files=files,
            app_user_id="docai-demo",
        )
    )
    print("submission_id:", result.submission_id)
    print("summary:", result.summary)
    print("status trail:")
    for line in result.status_trail:
        print("  -", line)


def __basename(p: str) -> str:
    import os

    return os.path.basename(p)


if __name__ == "__main__":
    main()
