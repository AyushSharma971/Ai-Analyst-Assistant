"""Connectivity probe for the optional external services in the host .env.
No secrets printed — only reachable/blocked + a short response snippet."""

from __future__ import annotations

import os
import socket
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings, _env_overlay
from app.research import PerplexityResearchSource


def _host_reachable(host: str, port: int = 443, timeout: float = 4.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def main():
    s = Settings(**_env_overlay())

    print("=== Perplexity (agent #10 research) ===")
    print("api key present:", bool(s.perplexity_api_key), "| model:", s.perplexity_model)
    src = PerplexityResearchSource(s)
    print("source.available:", src.available)
    if src.available:
        hit = src.lookup("annual_revenue", {"insured": "Microsoft Corporation"})
        if hit is None:
            print("lookup -> None (blocked/unreachable OR model returned 'unknown')")
        else:
            print("lookup OK -> value snippet:", repr(str(hit.get("value"))[:80]),
                  "| source:", hit.get("source_document"))

    print("\n=== SharePoint host reachability (intake #1, future) ===")
    for var in ("SHAREPOINT_SERVICE_URL", "SH_SITE_URL", "AD_TOKEN_URL"):
        url = os.environ.get(var) or _env_overlay().get(var.lower()) or ""
        # pull from raw .env too (these aren't Nevag settings fields)
        if not url:
            try:
                from dotenv import dotenv_values
                url = dotenv_values(".env").get(var, "") or ""
            except Exception:
                url = ""
        host = url.split("//")[-1].split("/")[0] if url else ""
        print(f"{var}: host={host or '(unset)'} reachable={_host_reachable(host) if host else False}")


if __name__ == "__main__":
    main()
