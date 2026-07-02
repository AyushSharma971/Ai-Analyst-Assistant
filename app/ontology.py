"""Ontology & Knowledge Graph (architecture doc §6 agent #8).

Config-driven relationship rules (config/ontology_rules.json). Each rule fires on
an attribute/value condition and asserts an evidence-backed fact
(subject -> relation -> object). Only facts grounded in the canonical record are
emitted; nothing is inferred without a backing value. Reusable by swapping the
ontology config for another domain.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common import bundled_config, condition_matches, load_json_config
from .config import Settings

_BUNDLED = bundled_config("ontology_rules.json")


def load_ontology(settings: Optional[Settings] = None) -> List[Dict[str, Any]]:
    path = (getattr(settings, "ontology_rules_path", None) if settings else None) or _BUNDLED
    return load_json_config(path, default={}, key="rules") or []


def derive_facts(record: Dict[str, Any], rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for rule in rules:
        attr = rule.get("attribute")
        if attr in record and condition_matches(rule.get("when", {}), record[attr]):
            facts.append(
                {
                    "subject": rule["subject"],
                    "relation": rule["relation"],
                    "object": rule["object"],
                    "evidence": {attr: record[attr]},  # grounded in canonical data
                }
            )
    return facts
