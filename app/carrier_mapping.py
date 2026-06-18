"""Carrier Mapping — canonical attributes -> carrier rater cells
(architecture doc §5 step 5, part 1).

Decides WHERE each approved value goes in the carrier's Excel rater. Fully
config-driven: the attribute->cell map lives in config/carrier_mapping.json (or
nebag_CARRIER_MAPPING_PATH), keyed by carrier, so onboarding a new carrier or
moving a cell is a config edit, not a code change.

Only APPROVED fields are mapped — review/rejected/missing never reach autofill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .common import bundled_config, load_json_config
from .config import Settings

_BUNDLED_MAPPING_PATH = bundled_config("carrier_mapping.json")


@dataclass
class CarrierMapping:
    carrier: str
    sheet: Optional[str] = None
    workbook: Optional[str] = None              # optional default template path
    cells: Dict[str, str] = field(default_factory=dict)  # attribute -> cell ref


def load_carrier_mapping(settings: Optional[Settings] = None, carrier: Optional[str] = None) -> CarrierMapping:
    """Load the cell map for a carrier from config. Returns an empty mapping
    (no cells) if the file or carrier is missing — autofill then no-ops safely."""
    path = (getattr(settings, "carrier_mapping_path", None) if settings else None) or _BUNDLED_MAPPING_PATH
    carrier = carrier or (getattr(settings, "default_carrier", "default") if settings else "default")
    raw = load_json_config(path, default=None)
    if raw is None:
        return CarrierMapping(carrier=carrier)
    carriers = raw.get("carriers", raw) if isinstance(raw, dict) else {}
    spec = carriers.get(carrier) or {}
    return CarrierMapping(
        carrier=carrier,
        sheet=spec.get("sheet"),
        workbook=spec.get("workbook"),
        cells=dict(spec.get("cells", {})),
    )


def build_mapping_plan(fields: List[Dict[str, Any]], mapping: CarrierMapping) -> Dict[str, Any]:
    """Build the write plan for the autofill agent. Includes only approved fields
    that have a cell in the carrier map; everything else is reported as skipped."""
    cells: Dict[str, Dict[str, Any]] = {}
    unmapped: List[str] = []
    for f in fields:
        if f.get("status") != "approved":
            continue
        ref = mapping.cells.get(f["name"])
        if not ref:
            unmapped.append(f["name"])
            continue
        cells[f["name"]] = {"cell": ref, "value": f.get("value")}
    return {
        "carrier": mapping.carrier,
        "sheet": mapping.sheet,
        "workbook": mapping.workbook,
        "cells": cells,
        "unmapped_approved": unmapped,
    }
