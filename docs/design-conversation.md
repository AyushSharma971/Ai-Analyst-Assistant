
# Nevag Submission Triage Agent — Design Record

> A reconstructed record of the design conversation and decisions behind this
> service. Lives alongside the code so the *why* travels with the *what*.
> Date: 2026-06-03

---

## 1. Goal of the project

Build the **Nevag Submission Triage Agent**: an underwriting agent that takes
messy submissions (email + PDFs, scanned PDFs, Excel, Word, images, ZIPs),
extracts every attribute a carrier's Excel **rater** requires, validates each
value against evidence, and safely fills the rater — evidence-backed, governed,
explainable, and reproducible.

Two-part intent:
1. A **reusable agent orchestration platform** (LangGraph executor, agent
   registry, node types, shared workflow state, If/Else routing, HITL, audit).
2. **Nevag as the first workflow template** on that platform — not hardcoded.

Source of truth for the full architecture: the *Reusable Agent Workflow
Architecture* document (19 reusable agents, rater-driven flow, governance model,
config-driven reuse strategy, build order).

## 2. Key constraint — future integration

This standalone agent must later become **one agent inside the "One AI" chatbot**
(Odyssey Group, D&O reinsurance). It must be built so integration is a thin
adapter + a registration, **never a rewrite**.

### What the One AI README told us (decisive facts)

- One AI's repo is a **Next.js frontend + BFF only** — it contains **no agent
  intelligence**. Real agents are **separate backend services** called over HTTP.
- Therefore Nevag = a **standalone backend service** that:
  1. runs its own brain, and
  2. exposes the One AI **data-agent contract**:
     ```
     POST ${AGENT_BACKEND_URL}/{agentSlug}   body: { chat_history, query }
     -> { result: { response: <str | {table_header, table_data} | {chart}>,
                    token_data? } }
     ```
- Slug rule: `AgentName.lower().replace(" ", "_")` →
  "Nevag Submission Triage" → `/nevag_submission_triage`.
- Registering the agent in the One AI **core backend** makes it appear in the
  Agent Marketplace with **no frontend deploy**.
- Org LLM standard: **Azure OpenAI GPT-4.1** (`AZURE_OPENAI_*`, `gpt-4.1`,
  api version `2024-12-01-preview`), low temperature for deterministic extraction.

### Integration traps identified (the data-agent contract vs. Nevag's needs)

1. **Attachments don't fit `{chat_history, query}`** (text only). Solved by
   supporting two intake paths (below).
2. **HITL pause/resume has nowhere to live** in a one-shot response → review
   state must live in Nevag's own store, keyed by `submission_id`, with a
   separate resume call.
3. **LLM standardization** → default to Azure GPT-4.1 behind a pluggable interface.
4. **Long processing + delivering the filled Excel** → stream `status:` progress;
   deliver the workbook via blob link / `<<FILE_DATA>>`.

## 3. Decisions made

| Decision | Choice |
| --- | --- |
| Backend stack | **Python + FastAPI + LangGraph** |
| Intake path | **Both** — manual upload **and** server-side mailbox/blob ingestion |
| UI surface | **Generic runner first** — return extracted fields as a table + summary, zero frontend changes |
| LLM default | **Azure OpenAI GPT-4.1**, kept behind a pluggable interface |

### Integration-safety principles baked in
- One clean boundary: `NevagAgent.run()` / `.resume()`.
- No globals; all dependencies injected (config, LLM, store).
- `NEVAG_*`-prefixed config so it never collides with the host's env vars.
- Typed I/O contracts; host envelope mapping isolated to one place.
- Pluggable LLM / storage.
- HITL as resumable state, not a blocking call.

## 4. What was built (skeleton, verified runnable)

```
nevag-submission-agent/
├── app/
│   ├── main.py        # One AI adapter (the only host-coupled layer)
│   ├── service.py     # NevagAgent.run()/.resume() — the clean boundary
│   ├── contracts.py   # internal I/O models + One AI envelope mapping
│   ├── workflow.py    # WorkflowState + LangGraph graph (+ sequential fallback)
│   ├── agents.py      # 19 agents as deterministic stubs + pipeline + If/Else
│   ├── config.py      # injected NEVAG_* settings
│   ├── llm.py         # pluggable LLM, Azure GPT-4.1 default + mock
│   ├── intake.py      # both intake paths (upload + mailbox)
│   └── store.py       # submission state store (HITL pause/resume)
├── run_demo.py        # end-to-end demo, no server/credentials
├── requirements.txt · env.example · README.md
```

**Verified:** `python run_demo.py` runs the full Start→End pipeline (all nodes,
including the conditional HITL branch) and emits the exact One AI envelope — a
table of extracted rater fields (value/confidence/source/status) plus `token_data`
(summary, missing fields, review flag, rater file URL, deterministic audit run id).

### Known scaffold caveats
- Agents are **stubs** with deterministic mock output (wiring proven, intelligence
  pending).
- `resume` currently re-runs the whole graph rather than continuing from the
  checkpoint; real resume will continue from saved state.
- Demo ran on the **sequential fallback** executor (LangGraph not installed) and a
  **mock LLM**. Full stack recommends Python 3.11/3.12 (3.14 may lack some wheels).

## 5. Next steps (architecture-doc build order)

1. Implement real **Document AI** agent (PyMuPDF / pdfplumber / openpyxl + OCR).
2. Implement real **Semantic Extraction** (Azure GPT-4.1, rater-driven, evidence).
3. Real **Truth Validation** evidence gates.
4. Wire real **LangGraph** + Postgres-backed state store.
5. **Carrier Mapping + Safe Excel Autofill** (preserve formulas/macros).
6. One AI **registration payload** (+ optional custom-UI component) to see it live
   in the marketplace.

## 6. Integration checklist (when wiring into One AI)

- [ ] Register agent: `AgentName="Nevag Submission Triage"`, `AgentSource="agent studio"`.
- [ ] Host: set `AGENT_BACKEND_URL` (replace hard-coded `dachatbot-05` host).
- [ ] Keep the `{result.response / table_data}` envelope (already emitted).
- [ ] Decide filled-rater delivery (blob link / `<<FILE_DATA>>`).
- [ ] Standardize LLM on Azure GPT-4.1 (default).
