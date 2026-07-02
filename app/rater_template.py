"""Rater Template analysis — rater-driven attribute discovery
(architecture doc §5 step 5 / the 'rater-driven' premise).

The rater is the source of truth for WHICH attributes a carrier needs. This
module reads the blank carrier rater workbook and discovers:
  - the required attributes (by matching label cells against the attribute
    catalog's names/aliases), and
  - the target input cell for each (the cell next to the label), which doubles
    as a rater-derived carrier cell map for Safe Excel Autofill.

This replaces the previously hardcoded attribute list with genuine, config-free
discovery. openpyxl is imported lazily; if the rater can't be read the caller
falls back to the catalog's attribute set.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from .extraction import AttributeSpec


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _build_label_index(catalog: Dict[str, AttributeSpec]) -> Dict[str, str]:
    """normalized label/alias -> canonical attribute name."""
    index: Dict[str, str] = {}
    for name, spec in catalog.items():
        for label in spec.all_labels():
            index[_norm(label)] = name
    return index


def _match_attribute(cell_text: str, index: Dict[str, str]) -> Optional[str]:
    key = _norm(cell_text)
    if not key:
        return None
    if key in index:                       # exact label match
        return index[key]
    # whole-phrase containment (e.g. cell "Insured Name:" contains "insured name")
    for label, name in index.items():
        if label and (label in key or key in label):
            return name
    return None


def discover_rater_requirements(
    path: str, catalog: Dict[str, AttributeSpec], sheet: Optional[str] = None
) -> Tuple[List[str], Dict[str, str], str]:
    """Return (required_attributes, {attr: target_cell}, sheet_title).

    Scans label cells in rater order; the value cell is taken as the neighbour to
    the right of each matched label.
    """
    import openpyxl  # lazy

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet] if sheet and sheet in wb.sheetnames else wb.active
    index = _build_label_index(catalog)

    required: List[str] = []
    cell_map: Dict[str, str] = {}
    seen = set()
    for row in ws.iter_rows():
        for cell in row:
            if not isinstance(cell.value, str):
                continue
            attr = _match_attribute(cell.value, index)
            if attr and attr not in seen:
                seen.add(attr)
                required.append(attr)
                target = ws.cell(row=cell.row, column=cell.column + 1).coordinate
                cell_map[attr] = target
    title = ws.title
    wb.close()
    return required, cell_map, title


def analyze_rater_workbook(path: str) -> Dict[str, Any]:
    """Excel-dependency intelligence (agent #14): inspect the workbook BEFORE
    filling so autofill can protect it. Detects sheets, hidden tabs, macros,
    formulas + their cell dependencies, dropdown (data-validation) cells, and a
    template fingerprint for version detection. Best-effort; never raises."""
    import openpyxl

    keep_vba = path.lower().endswith(".xlsm")
    info: Dict[str, Any] = {
        "sheets": [],
        "hidden_sheets": [],
        "has_macros": keep_vba,
        "formulas": [],
        "dependencies": [],
        "dropdowns": [],
        "fingerprint": None,
    }
    try:
        wb = openpyxl.load_workbook(path, keep_vba=keep_vba)  # NOT data_only -> see formulas
    except Exception:
        return info

    cell_ref = re.compile(r"[A-Z]{1,3}\d+")
    for ws in wb.worksheets:
        info["sheets"].append(ws.title)
        if getattr(ws, "sheet_state", "visible") != "visible":
            info["hidden_sheets"].append(ws.title)
        for row in ws.iter_rows():
            for cell in row:
                v = cell.value
                if isinstance(v, str) and v.startswith("="):
                    info["formulas"].append({"sheet": ws.title, "cell": cell.coordinate, "formula": v})
                    info["dependencies"].append(
                        {"cell": f"{ws.title}!{cell.coordinate}", "depends_on": cell_ref.findall(v)}
                    )
        # Dropdowns / data validations
        try:
            for dv in ws.data_validations.dataValidation:
                if dv.type == "list":
                    info["dropdowns"].append(
                        {"sheet": ws.title, "range": str(dv.sqref), "allowed": dv.formula1}
                    )
        except Exception:
            pass

    seed = (str(info["sheets"]) + str(sorted(f["cell"] for f in info["formulas"]))).encode()
    info["fingerprint"] = hashlib.sha1(seed).hexdigest()[:12]
    wb.close()
    return info
