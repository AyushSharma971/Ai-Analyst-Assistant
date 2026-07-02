"""Safe Excel Autofill — write approved values into the carrier rater
(architecture doc §5 step 5, part 2).

Safety properties (the whole point of "safe"):
  - Opens the rater WITHOUT data_only, so existing formulas are kept as formulas
    (Excel recalculates on open). Macros are preserved via keep_vba for .xlsm.
  - Writes ONLY the cells in the mapping plan (approved + mapped). It never clears
    or rewrites any other cell, so the carrier's layout, formulas, and macros are
    left intact.
  - For cells with a DROPDOWN (data-validation list), the canonical value is mapped
    to the carrier's allowed vocabulary (e.g. boolean True -> "Yes") so the fill is
    a valid dropdown option. Mapping is config-driven (config/value_mappings.json).
  - Re-checks status defensively and writes to a NEW output file, never the
    template in place.

openpyxl is imported lazily; a missing wheel degrades to a clear error rather
than crashing the pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from .common import bundled_config, load_json_config, normalize_text
from .config import Settings


def load_value_mappings(settings: Optional[Settings] = None) -> Dict[str, Any]:
    path = (getattr(settings, "value_mappings_path", None) if settings else None) or bundled_config(
        "value_mappings.json"
    )
    return load_json_config(path, default={"boolean": {}, "by_attribute": {}})


def _parse_list_formula(formula1: Optional[str]) -> List[str]:
    """Parse a data-validation list source. Inline lists ('"Yes,No"') are parsed;
    range references (=Sheet!$A$1:$A$3) can't be resolved here -> []."""
    if not formula1 or str(formula1).startswith("="):
        return []
    return [opt.strip() for opt in str(formula1).strip().strip('"').split(",") if opt.strip()]


def _dropdown_cells(ws) -> Dict[str, List[str]]:
    """Map each cell coordinate that has a list data-validation to its options."""
    from openpyxl.utils.cell import get_column_letter, range_boundaries

    out: Dict[str, List[str]] = {}
    try:
        validations = ws.data_validations.dataValidation
    except Exception:
        return out
    for dv in validations:
        if dv.type != "list":
            continue
        options = _parse_list_formula(dv.formula1)
        if not options:
            continue
        for token in str(dv.sqref).split():
            if ":" in token:
                c0, r0, c1, r1 = range_boundaries(token)
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        out[f"{get_column_letter(c)}{r}"] = options
            else:
                out[token] = options
    return out


def _map_to_option(value: Any, options: List[str], mappings: Dict[str, Any], attribute: str) -> Any:
    """Map a canonical value to a matching dropdown option; leave unchanged if no
    confident match (so we never write an arbitrary wrong option)."""
    by_attr = (mappings.get("by_attribute", {}) or {}).get(attribute, {})
    norm_opts = {normalize_text(o): o for o in options}

    # explicit per-attribute mapping wins
    label = by_attr.get(str(value)) if not isinstance(value, bool) else by_attr.get(str(value).lower())
    if label and normalize_text(label) in norm_opts:
        return norm_opts[normalize_text(label)]

    if isinstance(value, bool):
        label = (mappings.get("boolean", {}) or {}).get("true" if value else "false")
        if label and normalize_text(label) in norm_opts:
            return norm_opts[normalize_text(label)]
        return value

    # direct (case-insensitive) match to an allowed option
    if normalize_text(value) in norm_opts:
        return norm_opts[normalize_text(value)]
    return value


def fill_rater(
    template_path: str,
    mapping_plan: Dict[str, Any],
    output_path: str,
    value_mappings: Optional[Dict[str, Any]] = None,
) -> Tuple[int, List[Tuple[str, str]]]:
    """Write the mapping plan's approved values into a copy of the rater template,
    mapping dropdown cells to the carrier's allowed vocabulary.

    Returns (cells_written, [(attribute, cell), ...]). Raises on hard I/O errors;
    callers wrap this so one failure doesn't kill the run.
    """
    import openpyxl  # lazy

    keep_vba = template_path.lower().endswith(".xlsm")
    # data_only=False (default) keeps formulas as formulas -> preserved on save.
    wb = openpyxl.load_workbook(template_path, keep_vba=keep_vba)

    sheet = mapping_plan.get("sheet")
    ws = wb[sheet] if sheet and sheet in wb.sheetnames else wb.active

    mappings = value_mappings if value_mappings is not None else load_value_mappings()
    dropdowns = _dropdown_cells(ws)

    written: List[Tuple[str, str]] = []
    for name, spec in (mapping_plan.get("cells") or {}).items():
        cell = spec.get("cell")
        if not cell:
            continue
        value = spec.get("value")
        if cell in dropdowns:  # map canonical value to the carrier's dropdown vocabulary
            value = _map_to_option(value, dropdowns[cell], mappings, name)
        ws[cell] = value
        written.append((name, cell))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    wb.save(output_path)
    wb.close()
    return len(written), written
