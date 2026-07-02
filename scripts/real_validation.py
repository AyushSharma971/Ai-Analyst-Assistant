"""Real-execution validation harness (local open-source stack).

Runs each component FOR REAL on the available inputs and reports accuracy,
failures, bottlenecks, and gaps:
  1. Real OCR (Tesseract) on the scanned fixture, vs known rendered text.
  2. Real Qdrant (local on-disk engine via qdrant-client) ingest + retrieve + persistence.
  3. Real embeddings: Ollama bge-m3 if reachable, else deterministic mock (labeled).
  4. Real LLM extraction: Ollama (Qwen/Llama) if reachable, else heuristic (labeled).
  5. Real autofill into an ENRICHED rater (dropdown data-validation + formula chain +
     merged cells); verifies formula/dropdown/merge preservation + written values.

NOTE: no real customer submissions or real carrier raters were provided, so
accuracy here is measured against the synthetic fixtures' known ground truth and
an enriched synthetic rater. Real-world accuracy still requires real files.
"""

from __future__ import annotations

import difflib
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings
from app.contracts import IntakeMode, NevagInput
from app.document_ai import build_ocr_engine, parse_documents
from app.embeddings import OllamaEmbedding, MockEmbedding
from app.retrieval import Retriever
from app.service import NevagAgent
from app.vectorstore import QdrantVectorStore
from scripts.make_fixtures import build_all

TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OLLAMA = "http://localhost:11434"
GROUND_TRUTH = {
    "insured_name": "Acme Health Systems, Inc.",
    "annual_revenue": 125000000.0,
    "employee_count": 1200,
    "industry": "Healthcare (Hospitals & Clinics)",
    "mfa_enabled": True,
    "prior_claims_count": 0,
}
OCR_EXPECTED = (
    "ACORD 125 Commercial Insurance Application (SCANNED) Named Insured: Acme Health "
    "Systems, Inc. Annual Revenue: $125,000,000 Employees 1200 MFA Enabled: YES"
)


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _ollama_models():
    import json
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=2) as r:
            data = json.loads(r.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def make_rich_rater(path):
    """Enriched carrier rater: merged title, dropdown (data validation), and a
    formula chain referencing the input cells — to test preservation on fill."""
    import openpyxl
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Rater"
    ws["A1"] = "Carrier D&O Rater (RICH TEMPLATE)"
    ws.merge_cells("A1:B1")  # merged region
    for cell, label in [
        ("A2", "Insured Name"), ("A3", "Annual Revenue"), ("A4", "Employee Count"),
        ("A5", "Industry"), ("A6", "MFA Enabled"), ("A7", "Prior Claims"),
    ]:
        ws[cell] = label
    # Dropdown (data validation) on the MFA input cell — carrier vocabulary Yes/No.
    dv = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(ws["B6"])
    # Formula chain: base premium from revenue, loaded by MFA + prior claims.
    ws["A9"] = "Base Premium"
    ws["B9"] = "=B3*0.001"
    ws["A10"] = "Adjusted Premium"
    ws["B10"] = '=B9*(1+IF(B6="No",0.25,0))+B7*1000'
    wb.save(path)


def report(section, lines):
    print(f"\n{'='*70}\n{section}\n{'='*70}")
    for ln in lines:
        print(ln)


def main():
    findings = []
    paths = build_all()
    rich = os.path.join(os.path.dirname(paths["rater_template"]), "rater_rich.xlsx")
    make_rich_rater(rich)
    out_dir = os.path.join(os.getcwd(), "output")

    from app.config import _env_overlay
    host = _env_overlay()  # azure creds + deployments from .env (no secrets printed)
    azure_ok = bool(host.get("azure_openai_api_key") and host.get("azure_openai_endpoint"))
    ollama_up = _ollama_up()
    models = _ollama_models()

    if azure_ok:
        llm_provider = emb_provider = "azure_openai"
        emb_dim = 1536  # text-embedding-ada-002
    elif ollama_up and any("bge" in m for m in models):
        llm_provider = "ollama" if any(("qwen" in m or "llama" in m) for m in models) else "mock"
        emb_provider, emb_dim = "ollama", 1024
    else:
        llm_provider = emb_provider = "mock"
        emb_dim = 1024

    report("ENVIRONMENT", [
        f"tesseract: {os.path.exists(TESS)}",
        f"azure openai reachable: {azure_ok} (endpoint host: {(host.get('azure_openai_endpoint') or '').split('//')[-1].split('/')[0]})",
        f"ollama up: {ollama_up} models: {models}",
        f"-> LLM provider: {llm_provider} | embeddings: {emb_provider} (dim {emb_dim})",
        "real submissions provided: NO (using synthetic fixtures)",
    ])

    base = dict(host)  # carry azure creds + deployments
    base.update(
        ocr_provider="tesseract", tesseract_cmd=host.get("tesseract_cmd") or TESS,
        embedding_provider=emb_provider, embedding_dim=emb_dim,
        llm_provider=llm_provider,
        rater_template_path=rich, rater_output_dir=out_dir,
        rag_extraction_enabled=(emb_provider != "mock"),
    )

    # --- 1. REAL OCR --------------------------------------------------------- #
    s = Settings(**base)
    ocr = build_ocr_engine(s)
    docs = parse_documents([{"filename": "scanned_acord.pdf", "path": paths["scanned_pdf"]}], s, ocr)
    d = docs[0]
    ratio = difflib.SequenceMatcher(None, " ".join(d["text"].split()).lower(),
                                    OCR_EXPECTED.lower()).ratio()
    report("1. REAL OCR (Tesseract)", [
        f"engine: {ocr.name} available={ocr.available}",
        f"kind={d['kind']} parse_status={d['parse_status']} parser={d['parser']}",
        f"OCR text: {' '.join(d['text'].split())[:200]}",
        f"char similarity vs expected: {ratio:.2%}",
    ])
    if ratio < 0.90:
        findings.append(f"OCR similarity {ratio:.0%} (<90%): real OCR introduces noise.")

    # --- 2. REAL QDRANT (local on-disk engine) ------------------------------- #
    import tempfile
    os.makedirs(out_dir, exist_ok=True)
    qpath = tempfile.mkdtemp(prefix="nevag_qdrant_")  # fresh collection (no stale vectors)
    try:
        from qdrant_client import QdrantClient
        from app.embeddings import build_embedding_provider
        client = QdrantClient(path=qpath)
        embedder = build_embedding_provider(Settings(**base))  # real provider (azure/ollama/mock)
        vs = QdrantVectorStore(Settings(**{**base, "qdrant_collection": "nevag_real"}), embedder.dim, client=client)
        retr = Retriever(Settings(**base), embedder, vs)
        fixt_docs = parse_documents(
            [{"filename": os.path.basename(p), "path": p}
             for k, p in paths.items() if k != "rater_template"], s, ocr)
        rep = retr.ingest(fixt_docs, "subREAL")
        # Raw DENSE cosine (real embedding-quality signal, not the RRF fusion score):
        qv = embedder.embed(["annual revenue total gross revenue"])[0]
        dense = vs.search(qv, top_k=3, filters={"submission_id": "subREAL"})
        hits = retr.search("annual revenue", top_k=3, filters={"submission_id": "subREAL"})
        client.close()  # release the on-disk lock before reopening
        # persistence: reopen client and confirm collection survives
        client2 = QdrantClient(path=qpath)
        persisted = "nevag_real" in [c.name for c in client2.get_collections().collections]
        client2.close()
        report("2. REAL QDRANT (local on-disk)", [
            f"ingested chunks: {rep['chunks']} (embedder={rep['embedder']})",
            f"DENSE cosine ranking (real embedding quality): "
            f"{[(h['payload'].get('file_name'), round(h.get('score', 0), 3)) for h in dense[:3]]}",
            f"hybrid (RRF+MMR+rerank) top: "
            f"{[(h['payload'].get('file_name'), round(h.get('score', 0), 3)) for h in hits[:3]]}",
            f"collection persisted across reopen: {persisted}",
        ])
        if not hits:
            findings.append("Qdrant retrieval returned no hits for 'annual revenue'.")
    except Exception as exc:
        report("2. REAL QDRANT", [f"FAILED: {type(exc).__name__}: {exc}"])
        findings.append(f"Qdrant local execution failed: {exc}")

    # --- 3 & 4. REAL EXTRACTION ACCURACY ------------------------------------- #
    t0 = time.time()
    agent = NevagAgent(settings=Settings(**base))
    files = [{"filename": os.path.basename(p), "path": p}
             for k, p in paths.items() if k != "rater_template"]
    res = agent.run(NevagInput(query="Process submission", mode=IntakeMode.UPLOAD, files=files))
    elapsed = time.time() - t0
    by = {f.name: f.value for f in res.fields}
    correct, lines = 0, []
    for attr, exp in GROUND_TRUTH.items():
        got = by.get(attr, "<MISSING>")
        ok = (got == exp) or (isinstance(exp, str) and isinstance(got, str) and exp.split()[0].lower() in got.lower())
        correct += int(ok)
        lines.append(f"  {'OK ' if ok else 'XX '} {attr}: got={got!r} expected={exp!r}")
    acc = correct / len(GROUND_TRUTH)
    report(f"3/4. REAL EXTRACTION (llm={llm_provider}, emb={emb_provider}, ALL docs incl. OCR scan)", [
        f"pipeline wall time: {elapsed:.2f}s",
        f"extracted VALUE accuracy: {correct}/{len(GROUND_TRUTH)} = {acc:.0%}",
        *lines,
        f"approved={sum(1 for f in res.fields if f.status.value=='approved')} "
        f"review={sum(1 for f in res.fields if f.status.value=='review')} review_required={res.review_required}",
        ("NOTE: source-priority resolved the clean-PDF-vs-OCR'd-scan conflict -> "
         "straight-through (no false review)." if not res.review_required else
         "NOTE: OCR noise still forced some fields to review."),
    ])
    if llm_provider == "mock":
        findings.append("LLM extraction NOT validated with a real model (Ollama unavailable) — heuristic path measured instead.")
    if acc < 1.0:
        findings.append(f"Extraction accuracy {acc:.0%} (<100%) on fixtures.")

    # --- 5. REAL AUTOFILL + PRESERVATION ------------------------------------- #
    # Clean single-doc run (no OCR conflict) so the pipeline reaches autofill.
    import openpyxl
    clean = NevagAgent(settings=Settings(**base))
    cres = clean.run(NevagInput(query="fill", mode=IntakeMode.UPLOAD,
                                files=[{"filename": "application.pdf", "path": paths["digital_pdf"]}]))
    out_path = os.path.join(out_dir, cres.submission_id + "_filled.xlsx")
    pres = []
    try:
        wb = openpyxl.load_workbook(out_path)  # formulas as formulas
        ws = wb["Rater"]
        b9 = ws["B9"].value
        b10 = ws["B10"].value
        merged = "A1:B1" in [str(r) for r in ws.merged_cells.ranges]
        dvs = []
        try:
            dvs = [str(dv.sqref) for dv in ws.data_validations.dataValidation]
        except Exception:
            pass
        b3, b6 = ws["B3"].value, ws["B6"].value
        pres = [
            f"input B3 (annual_revenue) written: {b3!r}",
            f"input B6 (mfa_enabled) written: {b6!r}  (dropdown vocab is 'Yes'/'No')",
            f"formula B9 preserved: {b9!r}",
            f"formula B10 preserved: {b10!r}",
            f"merged A1:B1 preserved: {merged}",
            f"data validations preserved: {dvs}",
        ]
        report("5. REAL AUTOFILL + PRESERVATION", pres)
        if b9 != "=B3*0.001" or not (isinstance(b10, str) and b10.startswith("=")):
            findings.append("Formula NOT preserved on autofill.")
        if not merged:
            findings.append("Merged cell NOT preserved on autofill.")
        if not dvs:
            findings.append("Dropdown/data-validation NOT preserved on autofill.")
        if b6 is True or b6 == "True":
            findings.append("Dropdown VALUE MISMATCH: wrote boolean True into a 'Yes/No' dropdown cell "
                            "(no carrier-vocabulary mapping).")
    except Exception as exc:
        report("5. REAL AUTOFILL", [f"FAILED: {type(exc).__name__}: {exc}"])
        findings.append(f"Autofill/preservation check failed: {exc}")

    # --- FINDINGS ------------------------------------------------------------ #
    report("DISCOVERED FAILURES / BOTTLENECKS / GAPS", [f"- {x}" for x in findings] or ["(none)"])


if __name__ == "__main__":
    main()
