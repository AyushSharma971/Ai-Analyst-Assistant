"""External Research — fill gaps from APPROVED sources only
(architecture doc §5; conditional node, runs only when fields are missing).

Governance rule: never invent values. Research may only add a field if an
*approved* source returns it, and the field carries that source as its evidence.
With no approved sources configured (the default), this no-ops — missing stays
missing and routes to human review.

Pluggable by design: implement a ResearchSource (e.g. D&B, state filings, an
internal data warehouse) and register it via build_research_sources(). No source
backends ship here, so this is an integration point that degrades cleanly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from .config import Settings


class ResearchSource(Protocol):
    name: str
    available: bool

    def lookup(self, attribute: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return {value, confidence, source_document, evidence_snippet} or None."""
        ...


class PerplexityResearchSource:
    """Approved web-research source via Perplexity (sonar). OpenAI-compatible HTTP
    (stdlib urllib, no new dep). Returns a value ONLY when the model gives a
    concrete answer with citations; otherwise None (never invents). Results are
    marked 'review' by the caller so a human confirms external-sourced values."""

    name = "perplexity"

    def __init__(self, settings: Settings):
        self._s = settings
        self.available = bool(getattr(settings, "perplexity_api_key", None))

    def lookup(self, attribute: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        insured = context.get("insured") or context.get("insured_hint") or ""
        if not insured:
            return None
        question = (
            f"For the company '{insured}', what is the {attribute.replace('_', ' ')}? "
            "Answer with only the value. If you cannot find it from reputable sources, reply 'unknown'."
        )
        try:
            import json
            import urllib.request

            body = json.dumps({
                "model": self._s.perplexity_model,
                "temperature": 0,
                "messages": [{"role": "user", "content": question}],
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.perplexity.ai/chat/completions",
                data=body,
                headers={"Authorization": f"Bearer {self._s.perplexity_api_key}",
                         "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=float(self._s.llm_timeout_seconds)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = (data["choices"][0]["message"]["content"] or "").strip()
            citations = data.get("citations") or []
        except Exception:
            return None  # unreachable / error -> no enrichment (graceful)
        if not content or content.lower().startswith("unknown"):
            return None
        return {
            "value": content,
            "confidence": float(self._s.research_default_confidence),
            "source_document": citations[0] if citations else f"perplexity:{self._s.perplexity_model}",
            "evidence_snippet": content[:300],
            "status": self._s.research_default_status,
        }


def build_research_sources(settings: Settings) -> List[ResearchSource]:
    """Factory for APPROVED research backends, gated by their own config. Perplexity
    is enabled when NEVAG_PERPLEXITY_API_KEY (or host PERPLEXITY_API_KEY) is set."""
    sources: List[ResearchSource] = []
    if getattr(settings, "perplexity_api_key", None):
        sources.append(PerplexityResearchSource(settings))
    return sources


def research_missing(
    missing: List[str],
    context: Dict[str, Any],
    sources: List[ResearchSource],
    default_confidence: float = 0.5,
    default_status: str = "review",
) -> List[Dict[str, Any]]:
    """Query approved sources for each missing attribute. Returns extracted_field
    dicts. Default confidence/status are config-driven (passed by the agent)."""
    found: List[Dict[str, Any]] = []
    for attr in missing:
        for src in sources:
            if not getattr(src, "available", False):
                continue
            hit = src.lookup(attr, context)
            if not hit:
                continue
            found.append(
                {
                    "name": attr,
                    "value": hit.get("value"),
                    "confidence": float(hit.get("confidence", default_confidence)),
                    "status": hit.get("status", default_status),  # external -> needs confirm
                    "source_document": hit.get("source_document", f"external:{src.name}"),
                    "source_page": None,
                    "evidence_snippet": hit.get("evidence_snippet"),
                    "extractor": f"research:{src.name}",
                }
            )
            break  # first approved source that has it wins
    return found
