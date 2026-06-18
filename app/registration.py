"""One AI registration (architecture doc §6 step 6).

Registering this agent in the One AI **core backend** makes it appear in the
Agent Marketplace with no frontend deploy. This module builds the registration
payload and (optionally) POSTs it.

What's certain (from the One AI README, captured in the design record):
  - AgentName = "nebag Submission Triage"; AgentSource = "agent studio".
  - Slug rule: AgentName.lower().replace(" ", "_") -> "nebag_submission_triage".
  - The frontend calls a data agent at {AGENT_BACKEND_URL}/{slug} with
    { chat_history, query } and expects { result: { response, token_data? } }.

What's ASSUMED (exact core-backend field names aren't in this repo): the wrapper
field names below (AgentName/AgentSource/AgentSlug/AgentDescription/...). Adjust
to the real registry schema when integrating — the values are correct regardless.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .config import Settings


def build_registration_payload(settings: Settings, base_url: Optional[str] = None) -> Dict[str, Any]:
    """Build the data-agent registration payload for the One AI core backend."""
    slug = settings.api_slug
    base = base_url or settings.agent_backend_url or "${AGENT_BACKEND_URL}"

    return {
        # --- Certain fields (from the README) ---
        "AgentName": settings.agent_name,
        "AgentSource": settings.agent_source,          # "agent studio"
        "AgentSlug": slug,                              # nebag_submission_triage
        "AgentDescription": settings.agent_description,
        # --- Host/contract wiring (field names assumed; values correct) ---
        "AgentBackendUrl": base,                        # frontend calls {base}/{slug}
        "ResponseType": "table",                        # we emit table_header/table_data
        "Endpoints": {
            "run": f"/{slug}",                          # base data-agent contract
            "comprehensive": f"/sql_executor/{slug}",   # router's deep pass
            "upload": f"/upload/{slug}",                # manual-upload intake
            "resume": f"/resume/{slug}",                # HITL resume
            "health": "/healthz",
        },
        "DataAgentContract": {
            "request": {"chat_history": [], "query": "<string>"},
            "responseShape": {
                "result": {
                    "response": {"table_header": ["..."], "table_data": [["..."]]},
                    "token_data": {"submission_id": "...", "rater_file_url": "..."},
                }
            },
        },
        "Llm": {
            "provider": settings.llm_provider,
            "model": settings.azure_openai_deployment,  # gpt-4.1
            "apiVersion": settings.azure_openai_api_version,
        },
        "SampleQueries": [
            "Process this new D&O submission.",
            "What rater attributes are still missing for submission <id>?",
            "Resume submission <id> after my review.",
        ],
    }


def register_agent(
    settings: Settings,
    payload: Optional[Dict[str, Any]] = None,
    register_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """POST the registration payload to the One AI core backend.

    Integration point — NOT exercised here (needs the core-backend URL + auth).
    Uses stdlib urllib so it adds no dependency. Returns a small result dict.
    Timeout is config-driven (nebag_REGISTRATION_TIMEOUT_SECONDS) unless overridden.
    """
    url = register_url or settings.oneai_register_url
    if not url:
        raise RuntimeError("No registration URL (set nebag_ONEAI_REGISTER_URL or pass register_url).")
    payload = payload or build_registration_payload(settings)
    if timeout is None:
        timeout = float(getattr(settings, "registration_timeout_seconds", 30.0))

    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted host)
        body = resp.read().decode("utf-8", errors="replace")
        return {"status": resp.status, "body": body}
