
A reusable, document-heavy underwriting ** agent**: messy
submission in (email + PDF/scanned PDF/Excel/Word/ZIP) → every rater-required
attribute extracted **with evidence**, validated, and safely written into the
carrier's Excel rater. Built **standalone** but designed to drop into the **your AI platform**
chatbot as a data agent via a thin adapter + marketplace registration — no rewrite.

Design principles enforced throughout: **fully local / open-source first**,
**config-driven (nothing business-specific hardcoded)**, **deterministic with
graceful fallbacks**, **evidence-gated governance** (no evidence ⇒ no autofill),
and **clean provider interfaces** so cloud services are optional adapters.

---

## Technology stack (local-first; cloud is optional)

| Layer | Default (local, open source) | Optional adapter | Offline fallback |
|---|---|---|---|
| LLM | **Ollama** (Llama 3.1 / Qwen / Mistral) | Azure OpenAI (GPT‑4.1) | deterministic `MockLLM` / heuristic |
| Embeddings | **Ollama BGE‑M3** / sentence-transformers (MiniLM) | Azure OpenAI (ada‑002) | deterministic `MockEmbedding` |
| Vector store | **Qdrant** (Docker or local on‑disk) | Azure AI Search | in-memory store |
| Hybrid search | dense ⊕ **BM25** → RRF → MMR → rerank | cross-encoder reranker | heuristic reranker |
| OCR | **Tesseract** | Azure Document Intelligence | detect + flag (NullOCR) |
| Parsing | PyMuPDF, pdfplumber, python-docx, openpyxl | — | per-file graceful skip |
| State / audit | **SQLite** / in-memory | PostgreSQL | in-memory |
| Intake | manual upload | **SharePoint** (ACS app-only) | — |
| Research (#10) | — | **Perplexity** (sonar) | none (never invents) |
| Workflow | **LangGraph** + checkpointing | — | sequential executor |
| Config | JSON files + `NEVAG_*` env | host-name alias loader | built-in defaults |

> The proxy on the validated environment blocks third-party model registries
> (Ollama/Perplexity) but allows the org's Azure OpenAI + SharePoint — so the
> system was validated end-to-end on **Azure GPT‑4.1 + ada‑002 + Tesseract + local
> Qdrant**. Every provider is swappable purely via config.

---

## Architecture (boundary-first)

```
host (your ai platform) ──HTTP──▶ app/main.py        your ai platform adapter (only host-coupled layer)
                              │
                              ▼
                        app/service.py      NevagAgent.run()/.resume()/.ask()  ◀ the ONE boundary
                              │
                              ▼
                        app/workflow.py     LangGraph graph (+ sequential fallback), HITL checkpoint
                              │
                              ▼
                        app/agents.py       the reusable agents (all real)
        injected deps: config (NEVAG_*), llm, embeddings, vector store/retriever,
                       ocr, state store, observability — no globals, all injected.
```

Foundation: an **Agent Registry** ([registry.py](app/registry.py)) cataloging every
agent's capabilities/IO/version, and formal **Node Types** + declarative template
([nodes.py](app/nodes.py)). Integration-safety rules: no globals, all deps injected,
`NEVAG_`-prefixed config, typed I/O contracts, pluggable providers, HITL as
resumable state (not a blocking call), deterministic reproducible runs.

---

## The pipeline (reusable agents)

`submission_intake → pre_submission_context → rater_template → document_ai →
semantic_extraction → post_extraction_context → industry_product_detection →
canonical_schema → ontology_knowledge_graph → missing_attribute_detection →
[external_research] → truth_validation → [human_review] → risk_reasoning →
carrier_mapping → safe_excel_autofill → explainability_audit → underwriter_copilot →
human_review_workbench → learning`

Highlights of what's real:

- **Document AI** ([document_ai.py](app/document_ai.py)) — digital PDF (PyMuPDF + pdfplumber tables), **scanned PDF/image → Tesseract OCR with per-page confidence**, Excel, Word, email (`.eml` + `.msg`) with **attachments parsed recursively**; one bad file never kills the run.
- **Semantic Extraction** ([extraction.py](app/extraction.py)) — rater-driven, evidence-backed; **LLM-first** (strict-JSON, never-guess) with a **deterministic heuristic fallback** (label + **table** extraction); **locale-aware number parsing** (en `1,234.56` / de `1.234,56` / scale units k·m·bn·Mio·Mrd); **RAG** retrieval of focused per-attribute context; **duplicate-attribute reconciliation** (collapses multiple values per attribute by source priority/confidence/provenance; conflicts → review).
- **Truth Validation** ([validation.py](app/validation.py)) — six gates: evidence presence, **grounding** (scale- and translation-tolerant, anti-hallucination), confidence, type, config-driven constraints, **cross-document contradiction with source-priority resolution**.
- **Carrier Mapping + Safe Autofill** ([carrier_mapping.py](app/carrier_mapping.py), [excel_autofill.py](app/excel_autofill.py)) — rater-derived (or config) cell map; writes **only approved** values; **preserves formulas/macros/merged cells/dropdowns**; **maps canonical values to carrier dropdown vocabulary** (e.g. `True → "Yes"`).
- **Canonical Schema / Ontology / Industry / Risk / Missing / Research / Copilot / Workbench / Learning** — all implemented and config-driven. (Ontology is a config **rules engine**, not a graph DB; Risk uses config rules + scoring bands; Research ships a **Perplexity** connector, approved-source only.)
- **Explainability & Audit** ([audit.py](app/audit.py)) — full decision record with provenance, citations, reconciliation, versions, and a **deterministic `output_hash`**.

---

## Retrieval / RAG

Chunk → embed → upsert (Qdrant) + per-submission **BM25** index → **dense ⊕ BM25 →
RRF fusion → MMR diversification → rerank → evidence-quality gate**. Config-gated
(`NEVAG_RAG_EXTRACTION_ENABLED`); off by default so the deterministic full-text
path remains the offline floor. Embeddings cached by content hash (memory or
SQLite). See [retrieval.py](app/retrieval.py), [vectorstore.py](app/vectorstore.py),
[rerank.py](app/rerank.py), [embeddings.py](app/embeddings.py).

---

## Config-driven design

Single source of truth: one pydantic `Settings` ([config.py](app/config.py)) with a
generic `NEVAG_*` env overlay **and a host-alias loader** (so an existing host
`.env` using `AZURE_OPENAI_*`, `TXTEMBD_*`, `SH_*`, `TESSERACT_CMD`, … works without
renaming). Business behavior lives in **`config/*.json`**, not code:

| File | Drives |
|---|---|
| `attribute_catalog.json` | which attributes + types/aliases/constraints |
| `carrier_mapping.json` | canonical → carrier rater cells |
| `value_mappings.json` | canonical → dropdown vocabulary |
| `industry_taxonomy.json` | industry/product classification |
| `ontology_rules.json` | evidence-backed relationship facts |
| `risk_rules.json` | risk flags + scoring bands |
| `source_priority.json` | contradiction resolution order (+ OCR/degraded penalty) |
| `prompts.json` | extraction prompt (+ version) |
| `value_tokens.json` / `number_units.json` | boolean tokens / magnitude units |

---

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 1) Fully offline (deterministic mock LLM/embeddings, no services):
python run_demo.py            # pause/resume (HITL) demo
python run_docai_demo.py      # full pipeline on generated fixtures

# 2) Local open-source stack:
#   ollama serve && ollama pull llama3.1 && ollama pull bge-m3
#   docker run -p 6333:6333 qdrant/qdrant     (or local on-disk mode)
#   winget install UB-Mannheim.TesseractOCR
cp env.example .env           # then set the lines below
```

`.env` for the **local** stack:
```
NEVAG_LLM_PROVIDER=ollama
NEVAG_EMBEDDING_PROVIDER=ollama
NEVAG_VECTOR_STORE=qdrant
NEVAG_QDRANT_URL=http://localhost:6333
NEVAG_OCR_PROVIDER=tesseract
NEVAG_TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
NEVAG_RAG_EXTRACTION_ENABLED=true
NEVAG_DATABASE_URL=sqlite:///./nevag_state.db
```

`.env` for the **Azure adapter** (optional; host names auto-aliased):
```
NEVAG_LLM_PROVIDER=azure_openai
NEVAG_EMBEDDING_PROVIDER=azure_openai
NEVAG_EMBEDDING_DIM=1536
# AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / TXTEMBD_DEPLOYMENT_NAME picked up via alias loader
```

> **Security:** never commit real secrets. Keep `.env` gitignored and rotate any
> key that has been shared.

## Run as a service

```bash
uvicorn app.main:app --reload --port 8088
```
- `GET  /healthz` · `GET /readyz` — liveness + LLM/OCR/embeddings/vector-store status
- `POST /nevag_submission_triage` — your ai platform base contract (`{chat_history, query}`)
- `POST /upload/nevag_submission_triage` — manual file upload intake
- `POST /resume/nevag_submission_triage` — HITL resume after review
- `POST /copilot/nevag_submission_triage` — grounded Q&A over a submission
- `GET  /registration` · `GET /registry/catalog` — your ai platform payload + agent catalog

---

## Tests & scripts

```bash
python tests/run_tests.py     # 115 local, deterministic, offline checks
```
Covers determinism, every config override, extraction (incl. locale numbers +
tables), validation gates, source-priority + OCR tie-break, dropdown mapping,
duplicate reconciliation, hybrid retrieval (BM25/RRF/MMR/rerank, real local Qdrant),
RAG, embedding cache, SQLite store, connectors, and more.

Helper scripts ([scripts/](scripts/)): `make_fixtures.py`, `register_agent.py`,
`real_validation.py` (real OCR/Qdrant/autofill harness), `azure_probe.py`,
`connectivity_probe.py`, `sharepoint_probe.py`, `run_real_submission.py`.

---

## Status

**Implemented & real:** all reusable agents; foundation (registry, node types,
LangGraph + HITL checkpointing); Document AI (+ real Tesseract OCR); extraction
(LLM + heuristic + table + RAG, locale numbers, reconciliation); validation
(6 gates + source priority); canonical/mapping/safe-autofill (formula/macro/dropdown
preservation); audit (deterministic hash); copilot; observability (structured
logging + run metrics); slim persisted state; SQLite/Postgres stores; hybrid
retrieval over Qdrant; Perplexity research + SharePoint intake connectors.

**Validated on real data:** a real German D&O submission (FRoSTA AG + half-year
financial report PDF) ran end-to-end on Azure GPT‑4.1 — correctly extracting
insured name, industry (multilingual), and revenue from German text, with
governance routing missing/ambiguous fields to human review.

**Placeholder / partial (by design):** Ontology is a config rules-engine (not a
graph DB); Canonical is a light normalizer (no currency/date normalization suite);
Carrier Mapping has no versioned registry yet.

**Blocked in the validated environment (not a code issue):** Ollama model registry
and Perplexity are proxy-blocked (third-party SaaS); Azure OpenAI + SharePoint are
allow-listed and work.

**Needs your input for full validation:** a real **carrier D&O rater** workbook (to
replace the generic demo attribute catalog with rater-driven attributes and
validate autofill/macro preservation on a real template).

---

## your ai platform integration checklist

- [ ] Register agent: `AgentName="Nevag Submission Triage"`, `AgentSource="agent studio"` (`GET /registration` emits the payload).
- [ ] Host: set `NEVAG_AGENT_BACKEND_URL`.
- [ ] Keep the `{result.response / table_data}` envelope (already emitted) + filled-rater file delivery.
- [ ] Choose providers via config (local Ollama/Qdrant or Azure adapter).
