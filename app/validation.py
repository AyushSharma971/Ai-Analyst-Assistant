"""Truth Validation — evidence gates (architecture doc §5 step 3).

The governance core: a value is only safe to autofill if it is evidence-backed,
internally consistent, and within the rules. This module runs a sequence of
GATES over each extracted field and assigns a final status:

  - approved : passed every gate; safe for Safe-Excel-Autofill to write.
  - review   : soft issue a human should eyeball (low confidence, weak/missing
               evidence, value not grounded in its snippet, cross-doc conflict).
  - rejected : hard rule break (type mismatch, constraint violation). Never
               autofilled.

Gates, in order:
  1. evidence      — must cite a source document + a non-empty evidence snippet.
  2. grounding     — the value must actually appear in its cited snippet
                     (anti-hallucination check).
  3. confidence    — must be >= settings.validation_min_confidence.
  4. type          — value must match the attribute's declared dtype.
  5. constraints   — config-driven per-attribute rules from the catalog
                     (min / max / allowed / pattern). Violations are hard.
  6. contradiction — if other documents state a different value for the same
                     attribute, flag the conflict for human resolution.

Everything that can be configured lives in the catalog or settings — no rules
are baked into this code.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .common import digits, load_value_tokens, normalize_text, num_digits, tokenize
from .config import Settings
from .extraction import (
    AttributeSpec,
    build_candidate_map,
    coerce,
    load_catalog,
    spec_for,
)

# Status constants (mirror contracts.FieldStatus values; kept as plain strings so
# this module stays dependency-light and state remains JSON-friendly).
APPROVED = "approved"
REVIEW = "review"
REJECTED = "rejected"

# Shared text helpers now live in app.common (DRY).
_norm = normalize_text


def _is_grounded(
    value: Any, dtype: str, snippet: str, true_tokens, false_tokens,
    *, scale_tolerant: bool = True, min_sig: int = 2, str_overlap: float = 0.34,
) -> bool:
    """Is the value supported by its cited snippet? Tolerant of (a) scale — a value
    reported in thousands/millions still grounds via significant-digit match — and
    (b) translation/paraphrase — string values ground by token overlap, not exact
    substring. All thresholds are config-driven; a true mismatch still fails."""
    if not snippet:
        return False
    snip = _norm(snippet)
    if dtype in ("number", "integer"):
        want = num_digits(value)
        if want and want in digits(snippet):
            return True
        if scale_tolerant:  # significant digits (drop scaling zeros): 639,480,000 -> '63948'
            sig = want.rstrip("0")
            if len(sig) >= min_sig and sig in digits(snippet):
                return True
        sval = coerce(snippet, dtype, true_tokens, false_tokens)  # locale-aware re-parse
        return sval is not None and num_digits(sval) == want
    if dtype == "boolean":
        toks = true_tokens if value else false_tokens
        return any(t in snip for t in toks)
    # strings: token overlap (LLM may translate/paraphrase); fall back to substring
    value_tokens = set(tokenize(value))
    if value_tokens:
        overlap = sum(1 for t in value_tokens if t in set(tokenize(snippet))) / len(value_tokens)
        if overlap >= str_overlap:
            return True
    return _norm(value) in snip


def _check_type(value: Any, dtype: str) -> bool:
    if dtype == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if dtype == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if dtype == "boolean":
        return isinstance(value, bool)
    return isinstance(value, str)


def _check_constraints(value: Any, spec: AttributeSpec) -> List[str]:
    """Return a list of human-readable constraint violations (empty == ok)."""
    c = spec.constraints or {}
    problems: List[str] = []
    if "min" in c and isinstance(value, (int, float)) and value < c["min"]:
        problems.append(f"below minimum {c['min']}")
    if "max" in c and isinstance(value, (int, float)) and value > c["max"]:
        problems.append(f"above maximum {c['max']}")
    if "allowed" in c and c["allowed"]:
        allowed = [_norm(a) for a in c["allowed"]]
        if _norm(value) not in allowed:
            problems.append(f"not in allowed set {c['allowed']}")
    if "pattern" in c and c["pattern"]:
        if not re.search(c["pattern"], str(value)):
            problems.append(f"does not match pattern /{c['pattern']}/")
    return problems


def _values_conflict(a: Any, b: Any, dtype: str, tolerance: float) -> bool:
    if dtype in ("number", "integer"):
        try:
            return abs(float(a) - float(b)) > tolerance
        except Exception:
            return _norm(a) != _norm(b)
    if dtype == "boolean":
        return bool(a) != bool(b)
    return _norm(a) != _norm(b)


def _supply_evidence(value, dtype, retriever, submission_id, true_tokens, false_tokens, snippet_cap, ground_kw):
    """Evidence-supply: fetch a retrieved chunk that ACTUALLY contains the value and
    return it as a citable snippet. Returns None if nothing grounds the value — it
    never invents evidence, so the rule-based gate still decides."""
    try:
        hits = retriever.search(str(value), top_k=3, filters={"submission_id": submission_id})
    except Exception:
        return None
    for h in hits:
        p = h.get("payload", {}) or {}
        text = p.get("text", "") or ""
        if _is_grounded(value, dtype, text, true_tokens, false_tokens, **ground_kw):
            return {
                "snippet": (p.get("evidence_snippet") or text)[:snippet_cap],
                "file_name": p.get("file_name"),
                "page_number": p.get("page_number"),
            }
    return None


def validate_fields(
    fields: List[Dict[str, Any]],
    documents: List[Dict[str, Any]],
    settings: Settings,
    catalog: Optional[Dict[str, AttributeSpec]] = None,
    retriever: Optional[Any] = None,
    submission_id: Optional[str] = None,
    candidate_map: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Run every gate over each extracted field. Mutates each field's `status`,
    attaches `validation_reasons`, and sets `notes`. Returns (results, review).

    `results` is a compact per-field summary; `review` is True when any field is
    review/rejected (so the workflow routes to the HITL node)."""
    catalog = catalog or load_catalog(settings)
    min_conf = float(settings.validation_min_confidence)
    tolerance = float(settings.validation_numeric_tolerance)
    snippet_cap = int(settings.extract_snippet_max_chars)
    true_tokens, false_tokens = load_value_tokens(getattr(settings, "value_tokens_path", None))
    ground_kw = {
        "scale_tolerant": bool(settings.grounding_numeric_scale_tolerant),
        "min_sig": int(settings.grounding_numeric_min_significant_digits),
        "str_overlap": float(settings.grounding_string_min_overlap),
    }
    # Cross-document candidates: reuse the shared map (computed once in Semantic
    # Extraction) or build it here for standalone callers (text + tables).
    if candidate_map is None:
        specs = [spec_for(f["name"], catalog) for f in fields]
        candidate_map = build_candidate_map(specs, documents, settings)
    use_retrieval = bool(
        getattr(settings, "validation_use_retrieval", False) and retriever is not None and submission_id
    )

    results: List[Dict[str, Any]] = []
    needs_human = False

    for f in fields:
        spec = spec_for(f["name"], catalog)
        value = f.get("value")
        dtype = spec.dtype
        reasons: List[str] = []
        hard_fail = False

        # 1. evidence presence (with optional retrieval evidence-supply — additive)
        snippet = f.get("evidence_snippet") or ""
        if not snippet and use_retrieval:
            supplied = _supply_evidence(
                value, dtype, retriever, submission_id, true_tokens, false_tokens, snippet_cap, ground_kw
            )
            if supplied:
                f["evidence_snippet"] = supplied["snippet"]
                if not f.get("source_document"):
                    f["source_document"] = supplied["file_name"]
                f["evidence_source"] = "retrieval"
                snippet = supplied["snippet"]
        has_evidence = bool(f.get("source_document")) and bool(snippet)
        if not has_evidence:
            reasons.append("missing evidence (no source document or snippet)")

        # 2. grounding (anti-hallucination) — only meaningful if we have a snippet
        if snippet and not _is_grounded(value, dtype, snippet, true_tokens, false_tokens, **ground_kw):
            reasons.append("value not grounded in cited evidence")

        # 3. confidence
        if float(f.get("confidence", 0.0)) < min_conf:
            reasons.append(f"confidence {f.get('confidence', 0.0):.2f} < {min_conf:.2f}")

        # 4. type
        if not _check_type(value, dtype):
            reasons.append(f"type mismatch (expected {dtype})")
            hard_fail = True

        # 5. config-driven constraints (hard)
        for problem in _check_constraints(value, spec):
            reasons.append(f"constraint: {problem}")
            hard_fail = True

        # 5b. reconciliation conflict (multiple materially different values) -> review
        rec = f.get("reconciliation") or {}
        if rec.get("conflict") and getattr(settings, "reconcile_conflict_to_review", True):
            cand = ", ".join(f"{c.get('value')!r}@{c.get('source_document')}" for c in rec.get("candidates", [])[:4])
            reasons.append(f"multiple values for this attribute, confirm which applies: {cand}")

        # 6. cross-document contradiction (uses the shared candidate map).
        # Source-priority can RESOLVE a conflict: if a higher-priority source
        # agrees with this field's value, it's not flagged (decision stays rule-based).
        candidates = candidate_map.get(f["name"], [])
        conflicting = [
            c for c in candidates if _values_conflict(c["value"], value, dtype, tolerance)
        ]
        if conflicting:
            resolved = False
            if getattr(settings, "contradiction_use_source_priority", True):
                from .source_priority import load_source_priority, resolve as _resolve

                winner = _resolve(f["name"], candidates, load_source_priority(settings))
                if winner and winner.get("resolved") and not _values_conflict(
                    winner["value"], value, dtype, tolerance
                ):
                    resolved = True
                    f["resolved_by"] = "source_priority"
            if not resolved:
                others = ", ".join(
                    f"{c['value']!r}@{c.get('source_document')}" for c in conflicting[:3]
                )
                reasons.append(f"conflicting values in other documents: {others}")

        # Decide final status.
        if hard_fail:
            status = REJECTED
        elif reasons:
            status = REVIEW
        else:
            status = APPROVED

        f["status"] = status
        f["validation_reasons"] = reasons
        if reasons:
            f["notes"] = "; ".join(reasons)
        if status in (REVIEW, REJECTED):
            needs_human = True

        results.append({"field": f["name"], "status": status, "reasons": reasons})

    return results, needs_human
