"""Retrieval flow: chunk -> embed -> upsert -> similarity search.

This is the glue between Document AI output and the vector store. It chunks parsed
documents (with full provenance metadata), embeds the chunks via the configured
EmbeddingProvider, and upserts them into the configured VectorStore. Retrieval
embeds a query and runs similarity search (optionally filtered, e.g. by
submission_id) to feed grounded extraction / copilot answers.

Chunk size/overlap and top_k are config-driven. Metadata stored per chunk:
document_id, file_name, page_number, chunk_id, source_type, submission_id,
created_at, evidence_snippet.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .common import sha1_hex, tokenize
from .config import Settings
from .embeddings import EmbeddingProvider
from .rerank import Reranker, build_reranker
from .vectorstore import VectorStore, _cosine


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split(text: str, size: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]


def chunk_documents(
    parsed_documents: List[Dict[str, Any]], submission_id: str, settings: Settings
) -> List[Dict[str, Any]]:
    """Turn parsed documents into chunks with full provenance metadata."""
    size = int(settings.chunk_size)
    overlap = int(settings.chunk_overlap)
    snippet_cap = int(settings.extract_snippet_max_chars)
    created = _now_iso()
    out: List[Dict[str, Any]] = []

    def emit(doc: Dict[str, Any]) -> None:
        document_id = doc.get("sha1") or sha1_hex(doc.get("filename", ""), length=16)
        file_name = doc.get("filename")
        source_type = doc.get("kind")
        pages = doc.get("pages") or ([{"page": 1, "text": doc.get("text", "")}] if doc.get("text") else [])
        for p in pages:
            page_no = p.get("page", 1)
            for i, chunk in enumerate(_split(p.get("text", "") or "", size, overlap)):
                chunk_id = f"{document_id}:{page_no}:{i}"
                out.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "document_id": document_id,
                            "file_name": file_name,
                            "page_number": page_no,
                            "chunk_id": chunk_id,
                            "source_type": source_type,
                            "submission_id": submission_id,
                            "created_at": created,
                            "evidence_snippet": chunk[:snippet_cap],
                            "ocr": bool(p.get("ocr")),
                            "ocr_confidence": p.get("ocr_confidence"),
                        },
                    }
                )
        # Tables as their own chunks so RAG can retrieve table-resident values.
        for ti, tbl in enumerate(doc.get("tables", []) or []):
            rows = tbl.get("rows", []) or []
            table_text = "\n".join("\t".join("" if c is None else str(c) for c in r) for r in rows)
            if table_text.strip():
                chunk_id = f"{document_id}:t{tbl.get('page', 1)}:{ti}"
                out.append(
                    {
                        "text": table_text,
                        "metadata": {
                            "document_id": document_id,
                            "file_name": file_name,
                            "page_number": tbl.get("page", 1),
                            "chunk_id": chunk_id,
                            "source_type": source_type,
                            "submission_id": submission_id,
                            "created_at": created,
                            "evidence_snippet": table_text[:snippet_cap],
                            "is_table": True,
                        },
                    }
                )

        for att in doc.get("attachments", []) or []:
            emit(att)

    for d in parsed_documents or []:
        emit(d)
    return out


def ingest_documents(
    store: VectorStore,
    embedder: EmbeddingProvider,
    parsed_documents: List[Dict[str, Any]],
    submission_id: str,
    settings: Settings,
) -> Dict[str, Any]:
    """Chunk -> embed -> upsert. Returns a small report (never raises out)."""
    chunks = chunk_documents(parsed_documents, submission_id, settings)
    if not chunks:
        return {"chunks": 0, "backend": getattr(store, "backend", "?"), "embedder": getattr(embedder, "name", "?")}
    texts = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    embeddings = embedder.embed(texts)
    n = store.upsert_documents(texts, embeddings, metas)
    return {
        "chunks": n,
        "backend": getattr(store, "backend", "?"),
        "embedder": getattr(embedder, "name", "?"),
        "dim": getattr(embedder, "dim", None),
    }


def retrieve(
    store: VectorStore,
    embedder: EmbeddingProvider,
    query: str,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Embed the query and run dense similarity search (simple path)."""
    if not query:
        return []
    query_embedding = embedder.embed([query])[0]
    return store.search(query_embedding, top_k=top_k, filters=filters)


# --------------------------------------------------------------------------- #
# Hybrid retrieval primitives: BM25 keyword index, RRF fusion, MMR
# --------------------------------------------------------------------------- #
class KeywordIndex:
    """In-process BM25 over each submission's chunks (small corpora). Deterministic.

    Provides the keyword arm of hybrid retrieval, complementing dense vectors — it
    catches exact label/keyword hits ('MFA', 'Annual Revenue') a paraphrase-tolerant
    dense model can miss. Per-submission isolation by construction.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = float(k1)
        self._b = float(b)
        self._docs: Dict[str, Dict[str, Dict[str, Any]]] = {}  # sid -> chunk_id -> rec

    def has(self, submission_id: str) -> bool:
        return bool(self._docs.get(submission_id))

    def add(self, submission_id: str, items: List[Dict[str, Any]]) -> None:
        bucket = self._docs.setdefault(submission_id, {})
        for it in items:
            cid = it["payload"].get("chunk_id")
            if cid:
                bucket[cid] = {"payload": it["payload"], "tokens": tokenize(it["text"])}

    def search(self, submission_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        bucket = self._docs.get(submission_id) or {}
        if not bucket:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []
        recs = list(bucket.items())  # (chunk_id, rec)
        n = len(recs)
        avgdl = sum(len(r["tokens"]) for _, r in recs) / n if n else 0.0
        # document frequency per query term
        df = {t: sum(1 for _, r in recs if t in set(r["tokens"])) for t in set(q_terms)}
        hits = []
        for cid, r in recs:
            toks = r["tokens"]
            dl = len(toks) or 1
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for t in set(q_terms):
                if df.get(t, 0) == 0 or t not in tf:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                freq = tf[t]
                denom = freq + self._k1 * (1 - self._b + self._b * dl / (avgdl or 1))
                score += idf * (freq * (self._k1 + 1)) / (denom or 1)
            if score > 0:
                hits.append({"id": cid, "score": score, "payload": r["payload"]})
        hits.sort(key=lambda h: (-h["score"], h["id"]))  # deterministic
        return hits[:top_k]


def reciprocal_rank_fusion(rankings: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    """Fuse multiple ranked lists by RRF score = Σ 1/(k + rank). Deterministic."""
    scores: Dict[str, float] = {}
    payloads: Dict[str, Dict[str, Any]] = {}
    vectors: Dict[str, Any] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            cid = (hit.get("payload", {}) or {}).get("chunk_id") or hit.get("id")
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            payloads.setdefault(cid, hit.get("payload", {}))
            if hit.get("vector") is not None:
                vectors.setdefault(cid, hit["vector"])
    fused = [{"id": cid, "score": scores[cid], "payload": payloads[cid], "vector": vectors.get(cid)}
             for cid in scores]
    fused.sort(key=lambda h: (-h["score"], str(h["id"])))
    return fused


def mmr(
    query_vec: List[float],
    candidates: List[Dict[str, Any]],
    embedder: EmbeddingProvider,
    lam: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance — balance relevance vs. diversity so retrieved
    evidence spans different sources, not duplicates of one page. Deterministic."""
    if not candidates:
        return []
    # Reuse stored vectors when the store returned them; embed only the missing.
    vecs = [c.get("vector") for c in candidates]
    missing = [i for i, v in enumerate(vecs) if not v]
    if missing:
        embedded = embedder.embed([(candidates[i].get("payload", {}) or {}).get("text", "") for i in missing])
        for k, i in enumerate(missing):
            vecs[i] = embedded[k]
    selected: List[int] = []
    sel_vecs: List[List[float]] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < top_k:
        best_i, best_score = None, None
        for i in remaining:
            rel = _cosine(query_vec, vecs[i])
            div = max((_cosine(vecs[i], sv) for sv in sel_vecs), default=0.0)
            score = lam * rel - (1 - lam) * div
            if (
                best_score is None
                or score > best_score
                or (score == best_score and str(candidates[i].get("id")) < str(candidates[best_i].get("id")))
            ):
                best_i, best_score = i, score
        selected.append(best_i)
        sel_vecs.append(vecs[best_i])
        remaining.remove(best_i)
    return [candidates[i] for i in selected]


# --------------------------------------------------------------------------- #
# Retriever facade: ingest + full hybrid search pipeline
# --------------------------------------------------------------------------- #
class Retriever:
    """The retrieval entry point: chunk→embed→upsert (+BM25 index) on ingest, and
    dense ⊕ BM25 → RRF → MMR → rerank → evidence-gate → top_k on search. Every
    stage is config-driven and the whole pipeline is deterministic offline."""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: Optional[Reranker] = None,
    ):
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker or build_reranker(settings)
        self.keyword_index = KeywordIndex(k1=settings.bm25_k1, b=settings.bm25_b)

    def ingest(self, parsed_documents: List[Dict[str, Any]], submission_id: str) -> Dict[str, Any]:
        chunks = chunk_documents(parsed_documents, submission_id, self.settings)
        if not chunks:
            return {"chunks": 0, "backend": getattr(self.vector_store, "backend", "?"),
                    "embedder": getattr(self.embedder, "name", "?")}
        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        embeddings = self.embedder.embed(texts)
        n = self.vector_store.upsert_documents(texts, embeddings, metas)
        if self.settings.retrieval_hybrid:
            self.keyword_index.add(
                submission_id, [{"text": t, "payload": m} for t, m in zip(texts, metas)]
            )
        return {
            "chunks": n,
            "backend": getattr(self.vector_store, "backend", "?"),
            "embedder": getattr(self.embedder, "name", "?"),
            "dim": getattr(self.embedder, "dim", None),
            "hybrid": bool(self.settings.retrieval_hybrid),
        }

    def search(self, query: str, top_k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not query:
            return []
        s = self.settings
        final_k = int(top_k or s.retrieval_top_k)
        submission_id = (filters or {}).get("submission_id")

        qv = self.embedder.embed([query])[0]
        dense = self.vector_store.search(qv, top_k=int(s.retrieval_prefetch_k), filters=filters)

        if s.retrieval_hybrid and submission_id:
            if not self.keyword_index.has(submission_id):
                from .observability import get_logger

                get_logger("retrieval").warning(
                    "hybrid retrieval requested but BM25 index empty for submission "
                    "%s -> dense-only (re-ingest to restore keyword arm)", submission_id,
                )
            kw = self.keyword_index.search(submission_id, query, int(s.retrieval_prefetch_k))
            candidates = reciprocal_rank_fusion([dense, kw], k=int(s.rrf_k)) if kw else dense
        else:
            candidates = dense

        # Evidence-quality gate (config-driven floors).
        min_score = float(s.retrieval_min_score)
        ocr_floor = float(s.ocr_confidence_floor)
        gated = []
        for h in candidates:
            p = h.get("payload", {}) or {}
            if float(h.get("score", 0.0)) < min_score:
                continue
            conf = p.get("ocr_confidence")
            if p.get("ocr") and conf is not None and float(conf) < ocr_floor:
                continue  # drop low-confidence OCR evidence (when confidence is known)
            gated.append(h)
        if not gated:
            gated = candidates  # gate removed everything -> don't starve the caller

        # MMR diversification over a pool, then rerank, then final_k.
        if s.mmr_enabled and len(gated) > final_k:
            pool = gated[: int(s.mmr_pool)]
            gated = mmr(qv, pool, self.embedder, float(s.mmr_lambda), max(final_k, int(s.mmr_pool) // 2))

        reranked = self.reranker.rerank(query, gated)
        return reranked[:final_k]
