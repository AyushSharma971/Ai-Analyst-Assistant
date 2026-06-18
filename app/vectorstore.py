"""Vector store behind a configurable adapter interface.

The retrieval backend is pluggable (nebag_VECTOR_STORE):
  - qdrant       : QdrantVectorStore (default; this is the migration target)
  - azure_search : AzureAISearchVectorStore (kept as optional fallback)
  - memory       : MemoryVectorStore (deterministic, dependency-free; offline/tests)
  - none         : disabled

Nothing is hardcoded: collection/index names and the vector dimension come from
config. Qdrant collections are created on demand with the configured dimension.
Offline (no Qdrant URL / client), the factory falls back to the in-memory store
with a clear warning so the rest of the system keeps working.

Every adapter implements the same VectorStore interface:
    upsert_documents(chunks, embeddings, metadata) -> int
    search(query_embedding, top_k, filters) -> list[hit]
    delete_by_document_id(document_id) -> int
    health_check() -> dict
"""

from __future__ import annotations

import math
import uuid
import warnings
from typing import Any, Dict, List, Optional, Protocol

from .config import Settings

# Fixed namespace so chunk_id -> point id is stable across runs (reproducible).
_UUID_NS = uuid.UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")


def _point_id(chunk_id: str) -> str:
    """Stable UUID5 from a chunk_id (Qdrant ids must be int or UUID)."""
    return str(uuid.uuid5(_UUID_NS, str(chunk_id)))


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class VectorStore(Protocol):
    backend: str

    def upsert_documents(
        self, chunks: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]
    ) -> int: ...
    def search(
        self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]: ...
    def delete_by_document_id(self, document_id: str) -> int: ...
    def health_check(self) -> Dict[str, Any]: ...


# --------------------------------------------------------------------------- #
# In-memory store (deterministic; offline + tests)
# --------------------------------------------------------------------------- #
class MemoryVectorStore:
    backend = "memory"

    def __init__(self) -> None:
        self._points: Dict[str, Dict[str, Any]] = {}  # chunk_id -> {vector, payload}

    def upsert_documents(self, chunks, embeddings, metadata) -> int:
        for text, vec, meta in zip(chunks, embeddings, metadata):
            cid = meta.get("chunk_id") or _point_id(text)
            self._points[cid] = {"vector": list(vec), "payload": {**meta, "text": text}}
        return len(chunks)

    def search(self, query_embedding, top_k=5, filters=None) -> List[Dict[str, Any]]:
        hits = []
        for cid, pt in self._points.items():
            if filters and any(pt["payload"].get(k) != v for k, v in filters.items()):
                continue
            hits.append({"id": cid, "score": _cosine(query_embedding, pt["vector"]),
                         "payload": pt["payload"], "vector": pt["vector"]})  # vector reused by MMR
        # Deterministic: sort by score desc, then id for ties.
        hits.sort(key=lambda h: (-h["score"], h["id"]))
        return hits[:top_k]

    def delete_by_document_id(self, document_id) -> int:
        before = len(self._points)
        self._points = {
            c: p for c, p in self._points.items() if p["payload"].get("document_id") != document_id
        }
        return before - len(self._points)

    def health_check(self) -> Dict[str, Any]:
        return {"enabled": True, "backend": self.backend, "points": len(self._points)}


# --------------------------------------------------------------------------- #
# Qdrant store (migration target)
# --------------------------------------------------------------------------- #
class QdrantVectorStore:
    backend = "qdrant"

    def __init__(self, settings: Settings, dim: int, client: Any = None):
        self._s = settings
        self._dim = int(dim)
        self._collection = settings.qdrant_collection  # config-driven, not hardcoded
        self._client = client  # injectable (tests / DI)

    # --- client + collection lifecycle ---
    def _ensure_client(self):
        if self._client is not None:
            return self._client
        from qdrant_client import QdrantClient  # lazy

        url = self._s.qdrant_url
        if not url:
            raise RuntimeError("nebag_QDRANT_URL is required for the qdrant vector store.")
        # API key optional -> local unauthenticated Qdrant is allowed.
        self._client = QdrantClient(url=url, api_key=self._s.qdrant_api_key or None)
        return self._client

    def _ensure_collection(self, client) -> None:
        from qdrant_client import models

        try:
            exists = client.collection_exists(self._collection)
        except Exception:
            exists = any(c.name == self._collection for c in client.get_collections().collections)
        if not exists:
            client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
            )

    def upsert_documents(self, chunks, embeddings, metadata) -> int:
        from qdrant_client import models

        client = self._ensure_client()
        self._ensure_collection(client)
        points = [
            models.PointStruct(
                id=_point_id(meta.get("chunk_id")),
                vector=list(vec),
                payload={**meta, "text": text},
            )
            for text, vec, meta in zip(chunks, embeddings, metadata)
        ]
        client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def search(self, query_embedding, top_k=5, filters=None) -> List[Dict[str, Any]]:
        from qdrant_client import models

        client = self._ensure_client()
        self._ensure_collection(client)
        qfilter = None
        if filters:
            qfilter = models.Filter(
                must=[models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in filters.items()]
            )
        res = client.query_points(
            collection_name=self._collection,
            query=list(query_embedding),
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
            with_vectors=True,
        )
        points = getattr(res, "points", res)
        return [{"id": p.id, "score": p.score, "payload": p.payload,
                 "vector": getattr(p, "vector", None)} for p in points]

    def delete_by_document_id(self, document_id) -> int:
        from qdrant_client import models

        client = self._ensure_client()
        self._ensure_collection(client)
        client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                )
            ),
        )
        return 1  # Qdrant delete is fire-and-forget; exact count not returned

    def health_check(self) -> Dict[str, Any]:
        try:
            client = self._ensure_client()
            cols = [c.name for c in client.get_collections().collections]
            return {
                "enabled": True,
                "backend": self.backend,
                "url": self._s.qdrant_url,
                "collection": self._collection,
                "dim": self._dim,
                "collections": cols,
            }
        except Exception as exc:
            return {"enabled": False, "backend": self.backend, "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# Azure AI Search store (kept as optional fallback; integration point)
# --------------------------------------------------------------------------- #
class AzureAISearchVectorStore:
    backend = "azure_search"

    def __init__(self, settings: Settings, dim: int):
        self._s = settings
        self._dim = int(dim)
        self._index = settings.azure_search_index  # config-driven
        self._client = None

    def is_available(self) -> bool:
        s = self._s
        return bool(s.azure_search_endpoint and s.azure_search_api_key)

    def _client_or_raise(self):
        if self._client is not None:
            return self._client
        from azure.core.credentials import AzureKeyCredential  # lazy
        from azure.search.documents import SearchClient

        if not self.is_available():
            raise RuntimeError("Azure AI Search not configured (endpoint/api key missing).")
        self._client = SearchClient(
            endpoint=self._s.azure_search_endpoint,
            index_name=self._index,
            credential=AzureKeyCredential(self._s.azure_search_api_key),
        )
        return self._client

    def upsert_documents(self, chunks, embeddings, metadata) -> int:
        client = self._client_or_raise()
        docs = [
            {"id": _point_id(m.get("chunk_id")), "content": t, "embedding": list(v), **m}
            for t, v, m in zip(chunks, embeddings, metadata)
        ]
        client.upload_documents(documents=docs)
        return len(docs)

    def search(self, query_embedding, top_k=5, filters=None) -> List[Dict[str, Any]]:
        from azure.search.documents.models import VectorizedQuery

        client = self._client_or_raise()
        flt = " and ".join(f"{k} eq '{v}'" for k, v in (filters or {}).items()) or None
        vq = VectorizedQuery(vector=list(query_embedding), k_nearest_neighbors=top_k, fields="embedding")
        res = client.search(search_text=None, vector_queries=[vq], filter=flt, top=top_k)
        return [{"id": r.get("id"), "score": r.get("@search.score"), "payload": r} for r in res]

    def delete_by_document_id(self, document_id) -> int:
        client = self._client_or_raise()
        # Integration point: a production impl would query keys by document_id first.
        client.delete_documents(documents=[{"document_id": document_id}])
        return 1

    def health_check(self) -> Dict[str, Any]:
        return {
            "enabled": self.is_available(),
            "backend": self.backend,
            "index": self._index,
            "note": "integration point — not exercised offline",
        }


# --------------------------------------------------------------------------- #
# Startup validation + factory
# --------------------------------------------------------------------------- #
def validate_vector_store_config(settings: Settings) -> Dict[str, Any]:
    """Decide whether the configured backend is usable and explain why. Used at
    startup to surface 'enabled/disabled' without making network calls."""
    backend = (settings.vector_store or "qdrant").lower()
    if backend in ("none", "disabled"):
        return {"enabled": False, "backend": backend, "reason": "vector store disabled by config"}
    if backend == "memory":
        return {"enabled": True, "backend": "memory", "reason": "in-memory store"}
    if backend == "azure_search":
        ok = bool(settings.azure_search_endpoint and settings.azure_search_api_key)
        return {
            "enabled": ok,
            "backend": "azure_search",
            "reason": "configured" if ok else "nebag_AZURE_SEARCH_ENDPOINT/API_KEY not set",
        }
    if backend == "qdrant":
        if not settings.qdrant_url:
            return {"enabled": False, "backend": "qdrant", "reason": "nebag_QDRANT_URL not set"}
        auth = "authenticated" if settings.qdrant_api_key else "unauthenticated (local)"
        return {
            "enabled": True,
            "backend": "qdrant",
            "reason": f"configured, {auth}",
            "url": settings.qdrant_url,
            "collection": settings.qdrant_collection,
        }
    return {"enabled": False, "backend": backend, "reason": f"unknown backend '{backend}'"}


def build_vector_store(settings: Settings, dim: int, client: Any = None) -> VectorStore:
    """Build the configured vector store, falling back to the in-memory store
    (with a warning) whenever the requested backend isn't usable offline."""
    status = validate_vector_store_config(settings)
    backend = status["backend"]

    if backend == "memory":
        return MemoryVectorStore()
    if backend in ("none", "disabled"):
        return MemoryVectorStore()  # safe no-op-ish store; health_check reports memory

    if backend == "azure_search":
        if status["enabled"]:
            return AzureAISearchVectorStore(settings, dim)
        warnings.warn(f"Azure AI Search disabled ({status['reason']}) -> MemoryVectorStore", RuntimeWarning, stacklevel=2)
        return MemoryVectorStore()

    if backend == "qdrant":
        if not status["enabled"] and client is None:
            warnings.warn(f"Qdrant disabled ({status['reason']}) -> MemoryVectorStore", RuntimeWarning, stacklevel=2)
            return MemoryVectorStore()
        store = QdrantVectorStore(settings, dim, client=client)
        if client is None:
            # Verify connectivity once; fall back cleanly if unreachable / not installed.
            hc = store.health_check()
            if not hc.get("enabled"):
                warnings.warn(f"Qdrant unreachable ({hc.get('error')}) -> MemoryVectorStore", RuntimeWarning, stacklevel=2)
                return MemoryVectorStore()
        return store

    warnings.warn(f"Unknown vector_store '{backend}' -> MemoryVectorStore", RuntimeWarning, stacklevel=2)
    return MemoryVectorStore()
