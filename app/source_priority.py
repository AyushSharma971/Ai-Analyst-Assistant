"""Source-priority resolution for cross-document contradictions (architecture §7).

When the same attribute appears with different values in different documents, a
config-driven priority order (e.g. Financials > Application > Email) decides the
authoritative value, so a field agreeing with the top source isn't flagged for
review. Fully config-driven; deterministic; decisions stay rule-based (this only
RESOLVES conflicts, it never invents values).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common import bundled_config, load_json_config, normalize_text
from .config import Settings


def load_source_priority(settings: Optional[Settings] = None) -> Dict[str, Any]:
    path = (getattr(settings, "source_priority_path", None) if settings else None) or bundled_config(
        "source_priority.json"
    )
    cfg = load_json_config(path, default={})
    return {
        "default": cfg.get("default", []),
        "by_attribute": cfg.get("by_attribute", {}),
        "filename_classes": cfg.get("filename_classes", {}),
        "degraded_source_types": cfg.get("degraded_source_types", []),
    }


def degraded(candidate: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    """1 if the candidate's source is a degraded kind (OCR/scanned/image), else 0.
    Used as a secondary tie-break so clean digital sources beat noisy scans."""
    return 1 if candidate.get("source_type") in cfg.get("degraded_source_types", []) else 0


def sort_key(attribute: str, c: Dict[str, Any], cfg: Dict[str, Any]):
    """(priority class rank, degraded penalty) — the authoritative-ordering key."""
    return (
        priority_rank(attribute, c.get("source_document"), c.get("source_type"), cfg),
        degraded(c, cfg),
    )


def classify_source(file_name: Optional[str], source_type: Optional[str], cfg: Dict[str, Any]) -> str:
    """Map a candidate's source to a priority class: filename keyword first, then
    the document kind, else 'other'."""
    fname = normalize_text(file_name or "")
    for cls, keywords in cfg.get("filename_classes", {}).items():
        if any(normalize_text(kw) in fname for kw in keywords):
            return cls
    return source_type or "other"


def priority_rank(attribute: str, file_name: Optional[str], source_type: Optional[str], cfg: Dict[str, Any]) -> int:
    """Lower = higher priority. Unknown classes sort last (stable)."""
    order: List[str] = cfg.get("by_attribute", {}).get(attribute) or cfg.get("default", [])
    cls = classify_source(file_name, source_type, cfg)
    return order.index(cls) if cls in order else len(order)


def resolve(attribute: str, candidates: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick the authoritative candidate by priority. Returns the winning candidate
    annotated with `resolved` (True only when the highest-priority class is
    internally consistent — i.e. a clear winner)."""
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda c: (*sort_key(attribute, c, cfg), str(c.get("source_document") or ""), str(c.get("value"))),
    )
    top = sort_key(attribute, ranked[0], cfg)
    # The top group = highest priority class AND least-degraded (clean digital first).
    top_values = {str(c.get("value")) for c in candidates if sort_key(attribute, c, cfg) == top}
    winner = dict(ranked[0])
    winner["resolved"] = len(top_values) == 1  # the authoritative source agrees with itself
    return winner
