"""FastAPI adapter — speaks the One AI data-agent contract.

This is the ONLY host-coupled layer. The One AI frontend calls a data agent as:
    POST ${AGENT_BACKEND_URL}/{agentSlug}   body: { chat_history, query }
and expects: { result: { response, token_data? } }   (README §11 / §10.4)

Slug rule (README §10.1): AgentName.lower().replace(" ", "_").
For "nebag Submission Triage" -> path "/nebag_submission_triage".

Extra routes beyond the base contract:
  - /sql_executor/{slug} : the "comprehensive" pass the router may call.
  - /upload/{slug}       : multipart route for the manual-upload intake path.
  - /resume/{slug}       : HITL resume (host calls after human review).
"""

from __future__ import annotations

import base64
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .contracts import (
    IntakeMode,
    nebagInput,
    OneAIRequest,
    to_oneai_envelope,
)
from .registration import build_registration_payload
from .registry import describe_registry
from .service import nebagAgent

app = FastAPI(title="nebag Submission Triage Agent")

# One process-wide agent (its deps are injected, no globals leak into the brain).
agent = nebagAgent()
SLUG = agent.settings.api_slug


def _check_slug(slug: str) -> None:
    if slug != SLUG:
        raise HTTPException(404, f"Unknown agent slug '{slug}' (expected '{SLUG}').")


def _component_health() -> dict:
    def _avail(obj):
        try:
            return obj.is_available() if hasattr(obj, "is_available") else None
        except Exception:
            return None
    return {
        "llm": {"provider": getattr(agent.llm, "name", "?"), "available": _avail(agent.llm)},
        "ocr": {"provider": getattr(agent._ocr, "name", "?"), "available": getattr(agent._ocr, "available", None)},
        "embeddings": {"provider": agent.embedder.name, "dim": agent.embedder.dim,
                       "available": _avail(agent.embedder)},
        "vector_store": agent.vector_store_status,
        "vector_store_health": agent.vector_store.health_check(),
    }


@app.get("/healthz")
def healthz():
    """Liveness + component status. Components degrade gracefully, so the service
    is 'ok' even when local backends (Ollama/Qdrant) aren't running."""
    return {"status": "ok", "agent": agent.settings.agent_name, "slug": SLUG, **_component_health()}


@app.get("/readyz")
def readyz():
    """Readiness: the service can always serve (deterministic offline fallbacks),
    so ready=true; component availability is reported for visibility."""
    return {"ready": True, "components": _component_health()}


@app.get("/registration")
def registration():
    """The One AI core-backend registration payload for this agent. Ops can GET
    this and register the agent to surface it in the Marketplace (no UI deploy)."""
    return build_registration_payload(agent.settings)


@app.post("/{slug}")
def run_agent(slug: str, body: OneAIRequest):
    """Base One AI data-agent contract. Query may carry a submission_id to
    process/resume an already-ingested submission (mailbox path)."""
    _check_slug(slug)
    inp = nebagInput(
        query=body.query,
        chat_history=body.chat_history,
        submission_id=body.submission_id,
        app_user_id=body.app_user_id,
        mode=IntakeMode.MAILBOX if body.submission_id is None else IntakeMode.REFERENCE,
    )
    result = agent.run(inp)
    return to_oneai_envelope(result, as_table=True)


@app.post("/sql_executor/{slug}")
def run_agent_comprehensive(slug: str, body: OneAIRequest):
    """The 'comprehensive' pass the dynamic-agent router may request. Same brain."""
    _check_slug(slug)
    inp = nebagInput(query=body.query, chat_history=body.chat_history,
                     submission_id=body.submission_id, app_user_id=body.app_user_id)
    result = agent.run(inp)
    return to_oneai_envelope(result, as_table=True)


@app.post("/upload/{slug}")
async def upload_submission(
    slug: str,
    query: str = Form(""),
    app_user_id: str = Form(""),
    files: List[UploadFile] = File(default_factory=list),
):
    """Manual-upload intake path (the custom UI / dedicated BFF route posts here)."""
    _check_slug(slug)
    packaged = []
    for f in files:
        content = await f.read()
        packaged.append(
            {
                "filename": f.filename,
                "content_type": f.content_type,
                "content_b64": base64.b64encode(content).decode(),
            }
        )
    inp = nebagInput(query=query, app_user_id=app_user_id,
                     mode=IntakeMode.UPLOAD, files=packaged)
    result = agent.run(inp)
    return to_oneai_envelope(result, as_table=True)


@app.post("/resume/{slug}")
def resume_submission(slug: str, submission_id: str, corrections: list):
    """HITL resume: host posts human corrections; workflow continues."""
    _check_slug(slug)
    result = agent.resume(submission_id, corrections)
    return to_oneai_envelope(result, as_table=True)


@app.post("/copilot/{slug}")
def copilot(slug: str, submission_id: str, question: str):
    """Underwriter Copilot (agent #17): grounded Q&A about a processed submission."""
    _check_slug(slug)
    return agent.ask(submission_id, question)


@app.get("/registry/catalog")
def registry_catalog():
    """The Agent Registry catalog (capabilities/IO/version/config) for planners/UI."""
    return {"agents": describe_registry()}
