"""Minimal Ollama HTTP client (stdlib urllib — no extra dependency).

Used by the local Ollama LLM and embedding providers. Kept tiny and dependency-
free so the local-first stack adds no Python packages. All calls are best-effort;
callers handle failures and fall back gracefully.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict


def ping(base_url: str, timeout: float = 2.0) -> bool:
    """True if an Ollama server answers at base_url (GET /api/tags). Fails fast
    (connection refused) when nothing is running locally."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def has_model(base_url: str, model: str, timeout: float = 2.0) -> bool:
    """True only if the server is up AND the model is actually pulled. (Server-up
    alone is not enough — calling a missing model errors at request time.)"""
    if not model:
        return False
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=timeout) as resp:
            names = [m.get("name", "") for m in json.loads(resp.read().decode()).get("models", [])]
    except Exception:
        return False
    base = model.split(":")[0]
    return any(n == model or n.split(":")[0] == base for n in names)


def post_json(base_url: str, path: str, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (local trusted host)
        return json.loads(resp.read().decode("utf-8"))
