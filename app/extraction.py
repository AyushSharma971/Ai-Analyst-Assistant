"""Semantic Extraction — rater-driven, evidence-backed attribute extraction
(architecture doc §5 step 2).

Given (a) the rater's required attributes and (b) the structured parsed_documents
from Document AI, produce one ExtractedField per attribute we can support with
evidence: value + confidence + source_document + source_page + evidence_snippet.

Two strategies, in order:
  1. LLM (default): Azure OpenAI GPT-4.1, low temperature, asked for STRICT JSON.
     Every value must be grounded in the supplied document text — the model is
     told never to guess and to omit attributes it cannot evidence.
  2. Heuristic fallback (deterministic): label/alias scan over the parsed text.
     Runs when no real LLM is configured (MockLLM) or the LLM output won't parse,
     so the pipeline stays runnable and reproducible with zero credentials.

Config-driven, not hardcoded: extraction behavior comes from an attribute catalog
(name -> description/type/aliases). The catalog ships with a canonical D&O set and
falls back to a sensible default spec for any attribute name the rater introduces.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .common import bundled_config, load_json_config, load_value_tokens, values_equivalent
from .config import Settings
from .llm import LLMClient


# --------------------------------------------------------------------------- #
# Attribute specs (the config that drives extraction)
# --------------------------------------------------------------------------- #
@dataclass
class AttributeSpec:
    name: str
    description: str = ""
    dtype: str = "string"          # string | number | integer | boolean
    aliases: List[str] = field(default_factory=list)
    # Optional, config-driven validation rules (consumed by validation.py):
    #   {required, min, max, allowed: [...], pattern: "<regex>"}
    constraints: Dict[str, Any] = field(default_factory=dict)

    def all_labels(self) -> List[str]:
        # primary label derived from the canonical name, plus configured aliases
        primary = self.name.replace("_", " ")
        seen, out = set(), []
        for label in [primary, *self.aliases]:
            key = label.lower()
            if key not in seen:
                seen.add(key)
                out.append(label)
        return out

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "AttributeSpec":
        return cls(
            name=name,
            description=data.get("description", ""),
            dtype=data.get("dtype", "string"),
            aliases=list(data.get("aliases", [])),
            constraints=dict(data.get("constraints", {})),
        )


# Last-resort built-in seed if no catalog file is found. The real catalog lives
# in config/attribute_catalog.json (or NEVAG_ATTRIBUTE_CATALOG_PATH) — this dict
# is only a safety net so the engine never hard-fails with no config at all.
_BUILTIN_SEED: Dict[str, AttributeSpec] = {
    "insured_name": AttributeSpec(
        "insured_name", "Legal name of the insured entity", "string",
        ["insured name", "named insured", "applicant", "insured"],
    ),
}

# Path to the catalog bundled with the repo (project_root/config/...).
_BUNDLED_CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "attribute_catalog.json",
)


def load_catalog(settings: Optional[Settings] = None) -> Dict[str, AttributeSpec]:
    """Load the attribute catalog from config (the 'how' of extraction).

    Order: NEVAG_ATTRIBUTE_CATALOG_PATH -> bundled config file -> built-in seed.
    Nothing about which attributes exist or how they're typed is baked into code.
    """
    path = (getattr(settings, "attribute_catalog_path", None) if settings else None)
    path = path or _BUNDLED_CATALOG_PATH
    raw = load_json_config(path, default=None)
    if raw is None:
        return dict(_BUILTIN_SEED)
    attrs = raw.get("attributes", raw) if isinstance(raw, dict) else {}
    catalog: Dict[str, AttributeSpec] = {}
    for name, data in attrs.items():
        if name.startswith("_") or not isinstance(data, dict):
            continue
        catalog[name] = AttributeSpec.from_dict(name, data)
    return catalog or dict(_BUILTIN_SEED)


def spec_for(name: str, catalog: Dict[str, AttributeSpec]) -> AttributeSpec:
    """Return the spec for an attribute. Unknown attributes (e.g. a brand-new
    rater field) get a sensible default spec so extraction still works for them."""
    return catalog.get(name) or AttributeSpec(name=name, aliases=[name.replace("_", " ")])


# --------------------------------------------------------------------------- #
# Document text access
# --------------------------------------------------------------------------- #
def iter_text_blocks(documents: List[Dict[str, Any]]) -> Iterable[Tuple[str, int, str]]:
    """Yield (filename, page, text) for every page of every document, recursing
    into email attachments. Tables are serialized into the page text so that
    label/value pairs living in tables are searchable too."""
    for doc in documents or []:
        fname = doc.get("filename", "unknown")
        pages = doc.get("pages") or []
        # Index tables by page so we can append them to the matching block.
        tables_by_page: Dict[int, List[List[List[Any]]]] = {}
        for t in doc.get("tables", []):
            tables_by_page.setdefault(t.get("page", 1), []).append(t.get("rows", []))

        if not pages and doc.get("text"):
            yield fname, 1, doc["text"]
        for p in pages:
            page_no = p.get("page", 1)
            text = p.get("text", "") or ""
            extra = tables_by_page.get(page_no, [])
            if extra:
                table_text = "\n".join(
                    "\t".join(str(c) for c in row) for rows in extra for row in rows
                )
                text = (text + "\n" + table_text).strip()
            if text:
                yield fname, page_no, text

        # Email attachments are real submission documents — recurse.
        for att in doc.get("attachments", []) or []:
            yield from iter_text_blocks([att])


# --------------------------------------------------------------------------- #
# Value coercion / formatting
# --------------------------------------------------------------------------- #
# A grouped numeric run (digits + thousands/decimal separators). Locale is resolved
# ALGORITHMICALLY in _normalize_number — no hardcoded locale assumption.
_NUM_TOKEN = re.compile(r"[-+]?\d(?:[\d.,' ]*\d)?")
_UNIT_RE = re.compile(r"\s*([a-zA-Z]{1,10})")    # magnitude suffix right after a number


def _normalize_number(token: str) -> Optional[float]:
    """Disambiguate decimal vs thousands separators dynamically (handles en '1,234.56'
    AND de '1.234,56' AND grouped '639.480'), then return a float. No locale hardcoded:
      - both '.' and ',' present  -> the LAST one is the decimal separator
      - one separator, repeated    -> all thousands
      - one separator, once        -> decimal unless it groups exactly 3 trailing digits
    """
    token = token.strip()
    has_dot, has_comma = "." in token, "," in token
    if has_dot and has_comma:
        dec = "." if token.rfind(".") > token.rfind(",") else ","
    elif token.count(".") > 1 or token.count(",") > 1:
        dec = None  # repeated separator -> grouping only
    elif has_dot or has_comma:
        sep = "." if has_dot else ","
        dec = sep if len(token.rsplit(sep, 1)[-1]) != 3 else None
    else:
        dec = None
    out = []
    for ch in token:
        if ch.isdigit() or ch in "+-":
            out.append(ch)
        elif dec and ch == dec:
            out.append(".")
        # any other separator char is dropped (thousands grouping)
    try:
        return float("".join(out))
    except ValueError:
        return None


def load_number_units(settings: Optional[Settings] = None) -> Dict[str, float]:
    """Magnitude multipliers from config/number_units.json (k/m/bn -> factor)."""
    path = (getattr(settings, "number_units_path", None) if settings else None) or bundled_config(
        "number_units.json"
    )
    data = load_json_config(path, default={})
    units = data.get("units", data) if isinstance(data, dict) else {}
    return {str(k).lower(): float(v) for k, v in units.items()}


def coerce(
    value: Any,
    dtype: str,
    true_tokens: Optional[List[str]] = None,
    false_tokens: Optional[List[str]] = None,
    number_units: Optional[Dict[str, float]] = None,
) -> Optional[Any]:
    """Coerce a raw extracted value to the spec's type. Returns None if the value
    cannot be represented as that type (treated as 'not found'). Boolean tokens and
    magnitude units are config-driven; loaded lazily if not supplied."""
    if value is None:
        return None
    if dtype in ("number", "integer"):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            num: Any = value
        else:
            s = str(value)
            m = _NUM_TOKEN.search(s)
            if not m:
                return None
            num = _normalize_number(m.group(0))
            if num is None:
                return None
            # Magnitude suffix immediately after the number ($125M, 1,2 Mrd, 639 Tsd).
            unit_match = _UNIT_RE.match(s[m.end():])
            if unit_match:
                if number_units is None:
                    number_units = load_number_units()
                num *= number_units.get(unit_match.group(1).lower(), 1.0)
        return int(round(num)) if dtype == "integer" else num
    if dtype == "boolean":
        if true_tokens is None or false_tokens is None:
            true_tokens, false_tokens = load_value_tokens()
        s = str(value).strip().lower()
        if s in true_tokens:
            return True
        if s in false_tokens:
            return False
        return None
    s = str(value).strip()
    return s or None


# --------------------------------------------------------------------------- #
# LLM strategy
# --------------------------------------------------------------------------- #
_FALLBACK_SYSTEM = (
    "You are an insurance submission data-extraction engine. Extract ONLY values "
    "explicitly supported by the provided documents. Never guess. Respond with "
    "STRICT JSON only."
)


def load_extraction_prompt(settings: Optional[Settings] = None) -> str:
    """Extraction system prompt from config/prompts.json (config-driven)."""
    path = (getattr(settings, "prompts_path", None) if settings else None) or bundled_config("prompts.json")
    return load_json_config(path, default=_FALLBACK_SYSTEM, key="extraction_system")


def _build_user_prompt(specs: List[AttributeSpec], blocks: List[Tuple[str, int, str]]) -> str:
    attr_lines = [
        f'- {s.name} ({s.dtype}): {s.description or "n/a"}; aliases: {", ".join(s.aliases) or "—"}'
        for s in specs
    ]
    doc_sections = [f"=== {fname} (page {page}) ===\n{text}" for fname, page, text in blocks]
    schema = (
        '[{"name": "<attribute>", "value": <extracted value>, '
        '"confidence": <0..1>, "source_document": "<filename>", '
        '"source_page": <int>, "evidence_snippet": "<short verbatim quote>"}]'
    )
    return (
        "Extract the following rater attributes from the documents.\n\n"
        "ATTRIBUTES:\n" + "\n".join(attr_lines) + "\n\n"
        "DOCUMENTS:\n" + "\n\n".join(doc_sections) + "\n\n"
        "Return a JSON array. One object per attribute you can evidence; omit the "
        "rest. Use this exact shape:\n" + schema
    )


def _parse_llm_json(raw: str) -> Optional[List[Dict[str, Any]]]:
    """Tolerantly pull a JSON array out of an LLM response."""
    if not raw:
        return None
    text = raw.strip()
    # strip ```json ... ``` fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        data = json.loads(text)
    except Exception:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except Exception:
            return None
    if isinstance(data, dict):
        data = data.get("fields") or data.get("attributes") or [data]
    return data if isinstance(data, list) else None


def _extract_with_llm(
    specs: List[AttributeSpec],
    blocks: List[Tuple[str, int, str]],
    llm: LLMClient,
    settings: Settings,
) -> Optional[List[Dict[str, Any]]]:
    # Skip the LLM entirely when the provider can't serve calls (e.g. Ollama not
    # running) -> go straight to the deterministic heuristic. Avoids retry hangs.
    if hasattr(llm, "is_available") and not llm.is_available():
        return None

    prompt = _build_user_prompt(specs, blocks)
    system = load_extraction_prompt(settings)
    try:
        raw = llm.complete(system, prompt, temperature=float(settings.llm_temperature))
    except Exception:
        return None
    items = _parse_llm_json(raw)
    if items is None:
        return None

    default_conf = float(settings.extract_default_confidence)
    snippet_cap = int(settings.extract_snippet_max_chars)
    true_tokens, false_tokens = load_value_tokens(getattr(settings, "value_tokens_path", None))
    by_name = {s.name: s for s in specs}
    fields: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name")
        spec = by_name.get(name)
        if spec is None:
            continue
        value = coerce(it.get("value"), spec.dtype, true_tokens, false_tokens)
        if value is None:
            continue
        try:
            conf = max(0.0, min(1.0, float(it.get("confidence", default_conf))))
        except Exception:
            conf = default_conf
        fields.append(
            {
                "name": name,
                "value": value,
                "confidence": conf,
                "status": "approved",  # provisional; TruthValidation is the gate
                "source_document": it.get("source_document"),
                "source_page": it.get("source_page"),
                "evidence_snippet": (it.get("evidence_snippet") or "")[:snippet_cap] or None,
                "extractor": "llm",
            }
        )
    return fields


# --------------------------------------------------------------------------- #
# Heuristic fallback strategy (deterministic, no credentials)
# --------------------------------------------------------------------------- #
def _find_value_after_label(text: str, label: str, gap_max: int) -> Optional[Tuple[str, str]]:
    """Find 'Label: value' / 'Label - value' style pairs. Returns (value, snippet).

    The separator must be a colon or a SPACE-PADDED dash, and the short gap before
    it (max `gap_max` chars, config-driven) may not cross a colon/newline. This
    prevents matching a label word buried mid-sentence and stops an intra-word
    hyphen ('multi-factor') from acting as a separator."""
    pattern = re.compile(
        r"(?i)\b" + re.escape(label) + r"\b[^\n:]{0," + str(int(gap_max)) + r"}?(?::|\s[-–—]\s)\s*(?P<val>[^\n]+)"
    )
    m = pattern.search(text)
    if not m:
        return None
    value = re.split(r"\s{2,}", m.group("val").strip())[0].strip()
    return value, m.group(0).strip()


# --------------------------------------------------------------------------- #
# Candidate gathering — text (Label: value) + tables (row label -> next cell).
# Computed ONCE per run and shared by extraction, missing-detection, and the
# contradiction gate (DRY; avoids re-scanning the corpus per agent).
# --------------------------------------------------------------------------- #
def _kind_by_file(documents):
    """filename -> document kind (source_type), recursing into attachments."""
    out: Dict[str, Any] = {}

    def walk(d):
        if d.get("filename"):
            out[d["filename"]] = d.get("kind")
        for a in d.get("attachments", []) or []:
            walk(a)

    for d in documents or []:
        walk(d)
    return out


def _text_candidates(spec, blocks, settings, true_tokens, false_tokens, units, kind_by_file=None):
    gap = int(settings.extract_label_gap_max_chars)
    snippet_cap = int(settings.extract_snippet_max_chars)
    conf_primary = float(settings.extract_confidence_primary)
    conf_alias = float(settings.extract_confidence_alias)
    kind_by_file = kind_by_file or {}
    labels = spec.all_labels()
    out: List[Dict[str, Any]] = []
    for fname, page, text in blocks:
        for label in labels:
            hit = _find_value_after_label(text, label, gap)
            if not hit:
                continue
            raw_val, snippet = hit
            value = coerce(raw_val, spec.dtype, true_tokens, false_tokens, units)
            if value is None:
                continue
            out.append({
                "name": spec.name, "value": value,
                "confidence": conf_primary if label == labels[0] else conf_alias,
                "status": "approved", "source_document": fname, "source_page": page,
                "source_type": kind_by_file.get(fname),
                "evidence_snippet": snippet[:snippet_cap], "extractor": "heuristic", "origin": "text",
            })
    return out


def _find_in_tables(spec, documents, settings, true_tokens, false_tokens, units):
    """Row-label -> adjacent-cell extraction (handles 2-column forms / tables that
    the colon scanner misses)."""
    from .common import normalize_text

    snippet_cap = int(settings.extract_snippet_max_chars)
    conf_table = float(settings.extract_confidence_table)
    labels = [normalize_text(lbl) for lbl in spec.all_labels()]
    out: List[Dict[str, Any]] = []

    def scan(doc):
        fname = doc.get("filename")
        kind = doc.get("kind")
        for tbl in doc.get("tables", []) or []:
            page = tbl.get("page", 1)
            for row in tbl.get("rows", []) or []:
                cells = ["" if c is None else str(c) for c in row]
                for i, cell in enumerate(cells):
                    nc = normalize_text(cell)
                    if not nc or not any(nc == lbl or lbl in nc for lbl in labels):
                        continue
                    nxt = next((cells[j] for j in range(i + 1, len(cells)) if cells[j].strip()), None)
                    if nxt is None:
                        continue
                    value = coerce(nxt, spec.dtype, true_tokens, false_tokens, units)
                    if value is None:
                        continue
                    out.append({
                        "name": spec.name, "value": value, "confidence": conf_table,
                        "status": "approved", "source_document": fname, "source_page": page,
                        "source_type": kind,
                        "evidence_snippet": f"{cell}: {nxt}"[:snippet_cap],
                        "extractor": "heuristic", "origin": "table",
                    })
                    break
        for att in doc.get("attachments", []) or []:
            scan(att)

    for d in documents or []:
        scan(d)
    return out


def build_candidate_map(specs, documents, settings):
    """Per-attribute candidate values from text + tables, computed once per run."""
    blocks = list(iter_text_blocks(documents))
    tt, ft = load_value_tokens(getattr(settings, "value_tokens_path", None))
    units = load_number_units(settings)
    kinds = _kind_by_file(documents)
    return {
        spec.name: _text_candidates(spec, blocks, settings, tt, ft, units, kinds)
        + _find_in_tables(spec, documents, settings, tt, ft, units)
        for spec in specs
    }


def _best_candidate(cands, settings):
    """Deterministic best pick. When source-priority is enabled, the authoritative
    source ranks first; otherwise highest confidence wins. Stable tie-breaks."""
    if not cands:
        return None
    if getattr(settings, "contradiction_use_source_priority", True):
        from .source_priority import load_source_priority, sort_key

        cfg = load_source_priority(settings)
        return sorted(
            cands,
            key=lambda c: (
                *sort_key(c.get("name"), c, cfg),  # priority class, then degraded(OCR) penalty
                -float(c.get("confidence", 0)), str(c.get("source_document") or ""),
                int(c.get("source_page") or 0), str(c.get("value")),
            ),
        )[0]
    return sorted(
        cands,
        key=lambda c: (-float(c.get("confidence", 0)), str(c.get("source_document") or ""),
                       int(c.get("source_page") or 0), str(c.get("value"))),
    )[0]


def _fields_from_candidates(specs, candidate_map, settings):
    out: List[Dict[str, Any]] = []
    for spec in specs:
        best = _best_candidate(candidate_map.get(spec.name, []), settings)
        if best:
            out.append({k: v for k, v in best.items() if k != "origin"})
    return out


def reconcile_fields(fields, settings, catalog=None):
    """Collapse multiple values for the same canonical attribute into ONE.

    The authoritative value is chosen dynamically by source priority -> degraded
    (OCR/scanned) penalty -> confidence -> provenance (all config-driven, nothing
    company-specific). If materially different values remain, the single chosen
    field is annotated `reconciliation.conflict=True` (with all candidates) so the
    validation layer routes it to review instead of approving several values."""
    catalog = catalog or load_catalog(settings)
    abs_tol = float(settings.validation_numeric_tolerance)
    rel_tol = float(getattr(settings, "reconcile_numeric_rel_tolerance", 0.0))
    use_priority = bool(getattr(settings, "contradiction_use_source_priority", True))
    cfg = None
    if use_priority:
        from .source_priority import load_source_priority
        cfg = load_source_priority(settings)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for f in fields:
        if f["name"] not in groups:
            order.append(f["name"])
        groups.setdefault(f["name"], []).append(f)

    def rank_key(name, f):
        if cfg is not None:
            from .source_priority import sort_key
            base = sort_key(name, f, cfg)
        else:
            base = (0, 0)
        return (*base, -float(f.get("confidence", 0)), str(f.get("source_document") or ""),
                int(f.get("source_page") or 0), str(f.get("value")))

    out: List[Dict[str, Any]] = []
    for name in order:
        group = groups[name]
        if len(group) == 1:
            out.append(group[0])
            continue
        spec = spec_for(name, catalog)
        winner = dict(sorted(group, key=lambda f: rank_key(name, f))[0])
        # distinct (materially-different) values across the group
        distinct: List[Any] = []
        for f in group:
            if not any(values_equivalent(f.get("value"), d, spec.dtype, abs_tol, rel_tol) for d in distinct):
                distinct.append(f.get("value"))
        if len(distinct) > 1:
            winner["reconciliation"] = {
                "conflict": True,
                "chosen": winner.get("value"),
                "candidates": [
                    {"value": f.get("value"), "source_document": f.get("source_document"),
                     "source_type": f.get("source_type"), "confidence": f.get("confidence")}
                    for f in group
                ],
            }
        else:
            winner["reconciliation"] = {"conflict": False, "merged_from": len(group)}
        out.append(winner)
    return out


def _extract_heuristic(specs, blocks, settings):
    """Text-only heuristic over a block set (used by the RAG union fallback)."""
    tt, ft = load_value_tokens(getattr(settings, "value_tokens_path", None))
    units = load_number_units(settings)
    cmap = {s.name: _text_candidates(s, blocks, settings, tt, ft, units) for s in specs}
    return _fields_from_candidates(specs, cmap, settings)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def extract_fields(
    required_attributes: List[str],
    documents: List[Dict[str, Any]],
    llm: LLMClient,
    settings: Settings,
    catalog: Optional[Dict[str, AttributeSpec]] = None,
    candidate_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Extract rater-required attributes from parsed documents, with evidence.

    LLM-first; falls back to the deterministic text+table heuristic. A prebuilt
    candidate_map can be passed to avoid recomputing the corpus scan (DRY)."""
    catalog = catalog or load_catalog(settings)
    specs = [spec_for(name, catalog) for name in required_attributes]
    blocks = list(iter_text_blocks(documents))
    has_content = bool(blocks) or any((d.get("tables") for d in documents or []))
    if not has_content:
        return []

    fields = _extract_with_llm(specs, blocks, llm, settings)
    if fields:  # LLM produced at least one grounded field
        return fields
    if not settings.extract_enable_heuristic_fallback:
        return []  # fallback disabled by config
    cmap = candidate_map or build_candidate_map(specs, documents, settings)
    return _fields_from_candidates(specs, cmap, settings)


# --------------------------------------------------------------------------- #
# Retrieval-augmented extraction (RAG) — config-gated; falls back to full text
# --------------------------------------------------------------------------- #
def build_attribute_query(spec: AttributeSpec) -> str:
    """Query for retrieving an attribute's evidence: canonical name + aliases +
    description. Keeps retrieval rater-driven and config-driven (catalog)."""
    parts = [spec.name.replace("_", " ")] + list(spec.aliases)
    if spec.description:
        parts.append(spec.description)
    return " ".join(parts)


def extract_fields_rag(
    required_attributes: List[str],
    documents: List[Dict[str, Any]],
    llm: LLMClient,
    settings: Settings,
    retriever: Any,
    submission_id: str,
    catalog: Optional[Dict[str, AttributeSpec]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Retrieve focused per-attribute context via the hybrid Retriever (dense ⊕
    BM25 → RRF → MMR → rerank → gate), extract over that reduced context, and
    attach citations. Returns None to signal fallback to full-text extraction."""
    catalog = catalog or load_catalog(settings)
    specs = [spec_for(name, catalog) for name in required_attributes]
    top_k = int(getattr(settings, "rag_top_k", settings.retrieval_top_k))

    attr_hits: Dict[str, List[Dict[str, Any]]] = {}
    union: Dict[str, Tuple[str, int, str]] = {}  # chunk_id -> (file, page, text)
    for spec in specs:
        # The Retriever applies fusion/MMR/rerank + the evidence-quality gate.
        hits = retriever.search(
            build_attribute_query(spec), top_k=top_k, filters={"submission_id": submission_id}
        )
        attr_hits[spec.name] = hits
        for h in hits:
            p = h.get("payload", {})
            cid = p.get("chunk_id")
            if cid and cid not in union:
                union[cid] = (p.get("file_name"), p.get("page_number", 1), p.get("text", ""))

    blocks = [(fn, pg, tx) for (fn, pg, tx) in union.values() if tx]
    if not blocks:
        return None  # nothing retrieved -> caller falls back to full text

    if getattr(settings, "rag_per_attribute_prompts", False):
        # Field-specific prompts: extract each attribute over ONLY its own
        # retrieved chunks (more precise; one focused LLM call per attribute).
        fields = []
        by_name = {s.name: s for s in specs}
        for name, hits in attr_hits.items():
            attr_blocks = [
                (h["payload"].get("file_name"), h["payload"].get("page_number", 1), h["payload"].get("text", ""))
                for h in hits if h.get("payload", {}).get("text")
            ]
            if not attr_blocks:
                continue
            got = _extract_with_llm([by_name[name]], attr_blocks, llm, settings)
            if not got and settings.extract_enable_heuristic_fallback:
                got = _extract_heuristic([by_name[name]], attr_blocks, settings)
            fields.extend(got or [])
    else:
        # Batched: extract all attributes over the focused union (one LLM call).
        fields = _extract_with_llm(specs, blocks, llm, settings)
        if not fields:
            fields = _extract_heuristic(specs, blocks, settings) if settings.extract_enable_heuristic_fallback else []

    # Source tracing: attach the retrieved evidence (citations) per field.
    for f in fields:
        hits = attr_hits.get(f["name"], [])
        f["retrieval_method"] = "rag"
        f["citations"] = [
            {
                "chunk_id": h["payload"].get("chunk_id"),
                "file_name": h["payload"].get("file_name"),
                "page_number": h["payload"].get("page_number"),
                "score": round(float(h.get("score", 0.0)), 4),
            }
            for h in hits[:3]
        ]
    return fields
