"""Rerankers behind a clean interface.

Offline default is a DETERMINISTIC heuristic reranker (config-weighted blend of
the fused retrieval score and lexical query/chunk overlap) — no model needed, so
it's reproducible. A cross-encoder reranker is provided as an integration point
(needs a model/key) and degrades to pass-through when unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

from .common import tokenize
from .config import Settings


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]: ...


class NoOpReranker:
    name = "none"

    def rerank(self, query, hits):
        return hits


class HeuristicReranker:
    name = "heuristic"

    def __init__(self, settings: Settings):
        self._w_base = float(settings.rerank_weight_base)
        self._w_lex = float(settings.rerank_weight_lexical)

    def rerank(self, query, hits):
        if not hits:
            return hits
        qtok = set(tokenize(query))
        max_base = max((float(h.get("score", 0.0)) for h in hits), default=1.0) or 1.0
        out = []
        for h in hits:
            text = (h.get("payload", {}) or {}).get("text", "")
            ctok = set(tokenize(text))
            lexical = (len(qtok & ctok) / len(qtok)) if qtok else 0.0
            rs = self._w_base * (float(h.get("score", 0.0)) / max_base) + self._w_lex * lexical
            item = dict(h)
            item["rerank_score"] = rs
            out.append(item)
        out.sort(key=lambda h: (-h["rerank_score"], str(h.get("id", ""))))  # deterministic
        return out


class CrossEncoderReranker:
    """Integration point for a cross-encoder/LLM reranker. Not exercised offline
    (no model); degrades to pass-through so the pipeline keeps working."""

    name = "cross_encoder"

    def __init__(self, settings: Settings):
        self._settings = settings
        self.available = False  # set True once a model client is wired

    def rerank(self, query, hits):
        return hits


def build_reranker(settings: Settings) -> Reranker:
    provider = (settings.rerank_provider or "heuristic").lower()
    if provider == "none":
        return NoOpReranker()
    if provider == "cross_encoder":
        return CrossEncoderReranker(settings)
    return HeuristicReranker(settings)
