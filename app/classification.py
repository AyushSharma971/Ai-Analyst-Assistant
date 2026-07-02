"""Industry & Product Detection (architecture doc §6 agent #6).

Config-driven taxonomy (config/industry_taxonomy.json): keyword->industry and
industry->likely product. Classification is evidence-based (keyword hits over the
submission text + the extracted industry value); unknown routes to a default
rather than blocking. LLM is optional and not required here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .common import bundled_config, load_json_config
from .config import Settings

_BUNDLED = bundled_config("industry_taxonomy.json")


def load_taxonomy(settings: Optional[Settings] = None) -> Dict[str, Any]:
    path = (getattr(settings, "industry_taxonomy_path", None) if settings else None) or _BUNDLED
    return load_json_config(path, default={"industries": {}, "industry_to_product": {"_default": "D&O"}})


def classify(text: str, approved_industry: Optional[str], taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    industries = taxonomy.get("industries", {})
    products = taxonomy.get("industry_to_product", {})
    blob = (text or "").lower()
    if approved_industry:
        blob += " " + str(approved_industry).lower()

    scores = {ind: sum(1 for kw in kws if kw in blob) for ind, kws in industries.items()}
    scores = {ind: s for ind, s in scores.items() if s}
    best = max(scores, key=scores.get) if scores else None

    # Prefer the extracted industry value (multilingual, evidence-backed) over a
    # keyword guess; use the taxonomy only to map to a product (falling back to
    # the default). Data-driven — nothing about the value is hardcoded.
    if approved_industry:
        product = products.get(best) or products.get("_default", "D&O")
        return {
            "industry": approved_industry,
            "product": product,
            "confidence": 1.0,
            "evidence": "extracted industry value" + (f" (+taxonomy:{best})" if best else ""),
        }
    if not best:
        return {"industry": "Unknown", "product": products.get("_default", "D&O"),
                "confidence": 0.0, "evidence": None}
    total = sum(scores.values()) or 1
    return {
        "industry": best,
        "product": products.get(best) or products.get("_default", "D&O"),
        "confidence": round(scores[best] / total, 2),
        "evidence": f"matched {scores[best]} taxonomy keyword(s)",
    }
