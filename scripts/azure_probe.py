"""Azure OpenAI connectivity probe — no secrets printed.

Loads settings from .env (host aliases), then makes ONE tiny chat call and ONE
tiny embedding call to confirm reachability + that the deployments exist.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings, _env_overlay
from app.embeddings import AzureOpenAIEmbedding
from app.llm import AzureOpenAILLM


def _host(url):
    if not url:
        return None
    return url.split("//")[-1].split("/")[0]


def main():
    s = Settings(**_env_overlay())
    print("endpoint host:", _host(s.azure_openai_endpoint))
    print("api key present:", bool(s.azure_openai_api_key), "(masked)")
    print("chat deployment:", s.azure_openai_deployment)
    print("embedding deployment:", s.embedding_deployment, "| api_version:", s.azure_openai_api_version)

    print("\n--- chat call ---")
    try:
        llm = AzureOpenAILLM(s)
        print("is_available:", llm.is_available())
        out = llm.complete("You are a connectivity test.", "Reply with exactly: OK", temperature=0)
        print("chat OK -> response:", repr(out[:80]))
    except Exception as exc:
        msg = str(exc).replace(s.azure_openai_api_key or "X", "***") if s.azure_openai_api_key else str(exc)
        print("chat FAILED:", type(exc).__name__, "-", msg[:300])

    print("\n--- embedding call ---")
    try:
        emb = AzureOpenAIEmbedding(s)
        vecs = emb.embed(["connectivity test"])
        print("embed OK -> dim:", len(vecs[0]) if vecs else 0)
    except Exception as exc:
        msg = str(exc).replace(s.azure_openai_api_key or "X", "***") if s.azure_openai_api_key else str(exc)
        print("embed FAILED:", type(exc).__name__, "-", msg[:300])


if __name__ == "__main__":
    main()
