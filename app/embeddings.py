"""Embedding providers behind a clean interface.

Default is Azure OpenAI (text-embedding-ada-002), matching the org standard.
Offline / no credentials -> a DETERMINISTIC mock embedding so retrieval and tests
run with zero keys (same text -> same vector, reproducible). The brain depends on
the EmbeddingProvider protocol, never a concrete provider.
"""

from __future__ import annotations

import json
import math
import os
from typing import List, Optional, Protocol

from .common import sha1_hex, tokenize
from .config import Settings


class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def is_available(self) -> bool: ...
    def embed(self, texts: List[str]) -> List[List[float]]: ...


class MockEmbedding:
    """Deterministic, dependency-free embedding using the FEATURE-HASHING (hashed
    bag-of-words) trick: each token is hashed to a dimension (+/- sign), counts are
    accumulated, then L2-normalized.

    This is deliberately LEXICAL, not semantic: texts sharing words get positive
    cosine similarity, so offline retrieval (and RAG) is meaningful and fully
    reproducible. Identical text -> identical vector (cosine 1.0)."""

    name = "mock"

    def __init__(self, dim: int):
        self.dim = int(dim)

    def is_available(self) -> bool:
        return True

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t or "") for t in texts]

    def _vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = sha1_hex(tok)
            idx = int(h[:8], 16) % self.dim
            sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:  # empty / no tokens -> stable nonzero unit vector
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]


class SqliteVectorCache:
    """Persistent content-hash -> vector cache (stdlib sqlite3). Survives restarts
    and is shared across workers/processes pointing at the same file."""

    def __init__(self, path: str):
        import sqlite3

        self._path = path
        self._sqlite3 = sqlite3
        with sqlite3.connect(self._path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
            conn.commit()

    def get(self, key, default=None):
        with self._sqlite3.connect(self._path) as conn:
            row = conn.execute("SELECT v FROM emb WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def __setitem__(self, key, value):
        with self._sqlite3.connect(self._path) as conn:
            conn.execute("INSERT OR REPLACE INTO emb (k, v) VALUES (?, ?)", (key, json.dumps(value)))
            conn.commit()


class CachedEmbedding:
    """Caches vectors by content hash (avoid re-embedding identical/duplicate
    chunks across docs and re-runs). Wraps any provider; config-gated. The cache
    can be an in-memory dict (default) or a persistent SqliteVectorCache."""

    def __init__(self, inner: "EmbeddingProvider", cache=None):
        self._inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self._cache = cache if cache is not None else {}

    def is_available(self) -> bool:
        return self._inner.is_available()

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[Optional[List[float]]] = [None] * len(texts)
        to_embed: dict = {}            # key -> text (unique within this batch)
        slots: dict = {}               # key -> [indices]
        for i, t in enumerate(texts):
            key = sha1_hex(self.name, t)
            cached = self._cache.get(key)
            if cached is not None:
                out[i] = cached
                continue
            to_embed.setdefault(key, t)
            slots.setdefault(key, []).append(i)
        if to_embed:
            keys = list(to_embed.keys())
            vecs = self._inner.embed([to_embed[k] for k in keys])  # unique texts only
            for k, v in zip(keys, vecs):
                self._cache[k] = v
                for i in slots[k]:
                    out[i] = v
        return out  # type: ignore[return-value]


class OllamaEmbedding:
    """Local embeddings via Ollama (e.g. bge-m3). Dependency-free (stdlib HTTP).
    Dimension comes from config (drives the vector-store collection size)."""

    name = "ollama"

    def __init__(self, settings: Settings):
        self._s = settings
        self.dim = int(settings.embedding_dim)
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            from .ollama_runtime import has_model

            # Available only if the server is up AND the embedding model is pulled.
            self._available = has_model(self._s.ollama_base_url, self._s.embedding_model, 2.0)
        return self._available

    def embed(self, texts: List[str]) -> List[List[float]]:
        from .common import with_retry
        from .ollama_runtime import post_json

        s = self._s

        def _call() -> List[List[float]]:
            resp = post_json(
                s.ollama_base_url,
                "/api/embed",
                {"model": s.embedding_model, "input": list(texts)},
                float(s.llm_timeout_seconds),
            )
            return resp.get("embeddings") or []

        return with_retry(
            _call,
            attempts=int(s.llm_max_retries) + 1,
            backoff_seconds=float(s.llm_retry_backoff_seconds),
        )


class SentenceTransformerEmbedding:
    """Local embeddings via sentence-transformers (e.g. all-MiniLM-L6-v2). Optional
    dependency; degrades to mock when the package isn't installed."""

    name = "sentence_transformers"

    def __init__(self, settings: Settings):
        self._s = settings
        self.dim = int(settings.embedding_dim)
        self._model = None

    def is_available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except Exception:
            return False

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._s.embedding_model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class AzureOpenAIEmbedding:
    """Azure OpenAI embeddings (lazy `openai` import). Retry/timeout config-driven."""

    name = "azure_openai"

    def __init__(self, settings: Settings):
        self._s = settings
        self._client = None
        self.dim = int(settings.embedding_dim)

    def is_available(self) -> bool:
        s = self._s
        return bool(s.azure_openai_api_key and s.azure_openai_endpoint)

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import AzureOpenAI  # lazy

        s = self._s
        if not self.is_available():
            raise RuntimeError(
                "Azure OpenAI embeddings not configured: set NEVAG_AZURE_OPENAI_API_KEY "
                "and NEVAG_AZURE_OPENAI_ENDPOINT, or use NEVAG_EMBEDDING_PROVIDER=mock."
            )
        self._client = AzureOpenAI(
            api_key=s.azure_openai_api_key,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            timeout=float(s.llm_timeout_seconds),
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        from .common import with_retry

        self._ensure_client()
        s = self._s

        def _call() -> List[List[float]]:
            resp = self._client.embeddings.create(model=s.embedding_deployment, input=texts)
            return [d.embedding for d in resp.data]

        return with_retry(
            _call,
            attempts=int(s.llm_max_retries) + 1,
            backoff_seconds=float(s.llm_retry_backoff_seconds),
        )


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory chosen by config. Local-first: Ollama (bge-m3) or sentence-transformers
    (MiniLM); Azure is an OPTIONAL adapter. Any unavailable local/cloud backend
    degrades to the deterministic MockEmbedding so offline stays reproducible.
    Optionally wrapped with a content-hash cache (NEVAG_EMBEDDING_CACHE_ENABLED)."""
    provider = (settings.embedding_provider or "ollama").lower()
    dim = int(settings.embedding_dim)

    def _mock(reason: str) -> EmbeddingProvider:
        import warnings

        warnings.warn(f"Embeddings: {reason} -> deterministic MockEmbedding (offline).",
                      RuntimeWarning, stacklevel=3)
        return MockEmbedding(dim)

    if provider == "mock":
        base: EmbeddingProvider = MockEmbedding(dim)
    elif provider == "ollama":
        emb = OllamaEmbedding(settings)
        base = emb if emb.is_available() else _mock("Ollama not reachable")
    elif provider == "sentence_transformers":
        emb = SentenceTransformerEmbedding(settings)
        base = emb if emb.is_available() else _mock("sentence-transformers not installed")
    elif provider == "azure_openai":  # optional cloud adapter
        emb = AzureOpenAIEmbedding(settings)
        base = emb if emb.is_available() else _mock("Azure OpenAI not configured")
    else:
        raise ValueError(f"Unknown embedding_provider: {provider}")

    if getattr(settings, "embedding_cache_enabled", True):
        cache = None
        if (getattr(settings, "embedding_cache_backend", "memory") or "memory").lower() == "sqlite":
            path = settings.embedding_cache_path or os.path.join(os.getcwd(), "nevag_emb_cache.db")
            try:
                cache = SqliteVectorCache(path)
            except Exception:
                cache = None  # fall back to in-memory cache
        return CachedEmbedding(base, cache)
    return base
