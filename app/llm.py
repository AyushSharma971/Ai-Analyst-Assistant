"""Pluggable LLM layer.

The brain never imports a concrete provider. It depends on the LLMClient
protocol. Default implementation is Azure OpenAI GPT-4.1 (matching the One AI
org standard). A MockLLM is provided so the skeleton runs with zero credentials.

When integrating into One AI, you can either:
  - keep AzureOpenAILLM (reads nebag_AZURE_OPENAI_*), or
  - implement an adapter that wraps the host's existing Azure client and inject
    it — no brain code changes either way.
"""

from __future__ import annotations

from typing import List, Optional, Protocol

from .config import Settings


class LLMClient(Protocol):
    """Minimal interface the brain depends on."""

    def complete(self, system: str, user: str, *, temperature: Optional[float] = None) -> str:
        ...

    def is_available(self) -> bool:
        """Whether this provider can actually serve calls (config present)."""
        ...


class MockLLM:
    """Deterministic stand-in so the pipeline runs without credentials. Same
    input -> same output (reproducibility)."""

    name = "mock"

    def is_available(self) -> bool:
        return True

    def complete(self, system: str, user: str, *, temperature: Optional[float] = None) -> str:
        return "[mock-llm] " + user[:200]


class OllamaLLM:
    """Local LLM via Ollama (Llama 3.1 / Qwen / Mistral). Dependency-free (stdlib
    HTTP). Retries/timeout config-driven. Availability is probed (no server -> the
    brain falls back to the deterministic heuristic extractor)."""

    name = "ollama"

    def __init__(self, settings: Settings):
        self._s = settings
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            from .ollama_runtime import has_model

            # Available only if the server is up AND the model is pulled.
            self._available = has_model(self._s.ollama_base_url, self._s.llm_model,
                                        min(float(self._s.llm_timeout_seconds), 2.0))
        return self._available

    def complete(self, system: str, user: str, *, temperature: Optional[float] = None) -> str:
        from .common import with_retry
        from .ollama_runtime import post_json

        s = self._s
        temp = s.llm_temperature if temperature is None else temperature

        def _call() -> str:
            resp = post_json(
                s.ollama_base_url,
                "/api/chat",
                {
                    "model": s.llm_model,
                    "stream": False,
                    "options": {"temperature": temp},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                float(s.llm_timeout_seconds),
            )
            return (resp.get("message") or {}).get("content", "") or ""

        return with_retry(
            _call,
            attempts=int(s.llm_max_retries) + 1,
            backoff_seconds=float(s.llm_retry_backoff_seconds),
        )


class AzureOpenAILLM:
    """Azure OpenAI GPT-4.1 client. Lazily imports `openai`; retries + timeout are
    config-driven (nebag_LLM_MAX_RETRIES / _TIMEOUT_SECONDS / _RETRY_BACKOFF_SECONDS)."""

    name = "azure_openai"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = None  # lazy

    def is_available(self) -> bool:
        s = self._settings
        return bool(s.azure_openai_api_key and s.azure_openai_endpoint)

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import AzureOpenAI  # lazy import

        s = self._settings
        if not self.is_available():
            raise RuntimeError(
                "Azure OpenAI not configured: set nebag_AZURE_OPENAI_API_KEY and "
                "nebag_AZURE_OPENAI_ENDPOINT, or use llm_provider='mock'."
            )
        self._client = AzureOpenAI(
            api_key=s.azure_openai_api_key,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            timeout=float(s.llm_timeout_seconds),
        )

    def complete(self, system: str, user: str, *, temperature: Optional[float] = None) -> str:
        from .common import with_retry  # local import to keep module dependency-light

        self._ensure_client()
        s = self._settings
        temp = s.llm_temperature if temperature is None else temperature

        def _call() -> str:
            resp = self._client.chat.completions.create(
                model=s.azure_openai_deployment,  # deployment name == "gpt-4.1"
                temperature=temp,
                timeout=float(s.llm_timeout_seconds),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""

        return with_retry(
            _call,
            attempts=int(s.llm_max_retries) + 1,
            backoff_seconds=float(s.llm_retry_backoff_seconds),
        )


def build_llm(settings: Settings) -> LLMClient:
    """Factory chosen by config. Local-first: Ollama by default. Azure OpenAI is
    kept as an OPTIONAL adapter; mock is the deterministic offline floor."""
    provider = (settings.llm_provider or "ollama").lower()
    if provider == "ollama":
        return OllamaLLM(settings)
    if provider == "mock":
        return MockLLM()
    if provider == "azure_openai":  # optional cloud adapter
        return AzureOpenAILLM(settings)
    raise ValueError(f"Unknown llm_provider: {provider}")
