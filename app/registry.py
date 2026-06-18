"""Agent Registry (architecture doc §3 build item 1, §4).

A catalog of the reusable agents with their capabilities, version, input/output
contracts, and config keys. A workflow planner can use this to select and wire
agents automatically instead of hardcoding a pipeline. The nebag template is one
ordered selection from this registry.

This formalizes "agents as a reusable catalog": the same agent code is reused
across carriers/products/domains; only config (schema/catalog/ontology/mapping/
rules) changes (doc §7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from . import agents as A


@dataclass
class AgentSpec:
    name: str
    instance: A.Agent
    version: str
    capabilities: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    config_keys: List[str] = field(default_factory=list)
    reuse: str = ""


def _spec(inst, version, capabilities, inputs, outputs, config_keys=None, reuse=""):
    return AgentSpec(
        name=inst.name,
        instance=inst,
        version=version,
        capabilities=capabilities,
        inputs=inputs,
        outputs=outputs,
        config_keys=config_keys or [],
        reuse=reuse,
    )


# One instance per agent (reused by the registry and the pipeline).
_BY_NAME: Dict[str, A.Agent] = {a.name: a for a in A.AGENT_PIPELINE}


AGENT_REGISTRY: Dict[str, AgentSpec] = {
    "submission_intake": _spec(
        _BY_NAME["submission_intake"], "1.0",
        "Package submission: expand ZIPs, hash dedup, version lineage, submission id",
        ["files", "query"], ["files", "file_lineage", "submission_id"],
        reuse="any email/file intake workflow",
    ),
    "pre_submission_context": _spec(
        _BY_NAME["pre_submission_context"], "1.0",
        "Early routing context from raw email: channel/broker/insured-hint/new-vs-renewal",
        ["files", "query"], ["submission_context"],
        reuse="change routing rules/labels for other domains",
    ),
    "rater_template": _spec(
        _BY_NAME["rater_template"], "1.0",
        "Rater-driven discovery + Excel dependency analysis (formulas/macros/dropdowns/fingerprint)",
        ["rater_template_path"], ["rater_required_attributes", "rater_cell_map", "rater_analysis"],
        ["rater_template_path", "attribute_catalog_path"], "any Excel automation workflow",
    ),
    "document_ai": _spec(
        _BY_NAME["document_ai"], "1.0",
        "Parse PDF/scanned+OCR/Excel/Word/email into page-level structured docs",
        ["files"], ["parsed_documents"],
        ["ocr_provider", "scanned_text_threshold"], "any document-heavy workflow",
    ),
    "semantic_extraction": _spec(
        _BY_NAME["semantic_extraction"], "1.0",
        "Rater-driven, evidence-backed extraction (LLM + heuristic fallback)",
        ["parsed_documents", "rater_required_attributes"], ["extracted_fields"],
        ["attribute_catalog_path", "llm_provider"], "different schemas/prompts/source-priority",
    ),
    "post_extraction_context": _spec(
        _BY_NAME["post_extraction_context"], "1.0",
        "Confirm business context from extracted evidence; flag conflicts",
        ["extracted_fields", "submission_context"], ["confirmed_context"],
        reuse="any workflow needing context/entity confirmation",
    ),
    "industry_product_detection": _spec(
        _BY_NAME["industry_product_detection"], "1.0",
        "Classify industry + likely product from config taxonomy + evidence",
        ["parsed_documents", "extracted_fields"], ["classification"],
        ["industry_taxonomy_path"], "change taxonomy for other domains",
    ),
    "canonical_schema": _spec(
        _BY_NAME["canonical_schema"], "1.0",
        "Normalize to a versioned canonical record with type coercion + provenance",
        ["extracted_fields"], ["canonical_record", "canonical_provenance", "schema_version"],
        ["canonical_schema_version"], "swap schema config for different projects",
    ),
    "ontology_knowledge_graph": _spec(
        _BY_NAME["ontology_knowledge_graph"], "1.0",
        "Derive evidence-backed relationship facts from config ontology rules",
        ["canonical_record"], ["graph_facts"],
        ["ontology_rules_path"], "project-specific ontology",
    ),
    "missing_attribute_detection": _spec(
        _BY_NAME["missing_attribute_detection"], "1.0",
        "Split gaps: Not Found / OCR Failed / Low Confidence / Contradiction",
        ["rater_required_attributes", "extracted_fields", "parsed_documents"],
        ["gap_report", "missing_fields"], reuse="any required-field completeness check",
    ),
    "external_research": _spec(
        _BY_NAME["external_research"], "1.0",
        "Fill gaps from APPROVED sources only; never invents (conditional)",
        ["missing_fields", "submission_context"], ["extracted_fields"],
        reuse="different source lists/search policies",
    ),
    "truth_validation": _spec(
        _BY_NAME["truth_validation"], "1.0",
        "Evidence gates: evidence/grounding/confidence/type/constraints/contradiction",
        ["extracted_fields", "parsed_documents"], ["validation_results", "review_required"],
        ["validation_min_confidence", "attribute_catalog_path"], "domain-specific rules",
    ),
    "human_review": _spec(
        _BY_NAME["human_review"], "1.0",
        "HITL pause/resume checkpoint (conditional)",
        ["review_required"], ["status_trail"], reuse="any human-governed workflow",
    ),
    "risk_reasoning": _spec(
        _BY_NAME["risk_reasoning"], "1.0",
        "Evidence-constrained risk signals + deterministic scoring bands",
        ["extracted_fields", "graph_facts"], ["risk_assessment"],
        reuse="different risk models/rules",
    ),
    "carrier_mapping": _spec(
        _BY_NAME["carrier_mapping"], "1.0",
        "Map canonical -> carrier rater cells (rater-derived or config); approved only",
        ["extracted_fields", "rater_cell_map"], ["mapping_plan"],
        ["carrier_mapping_path", "default_carrier"], "generic mapping; only config changes",
    ),
    "safe_excel_autofill": _spec(
        _BY_NAME["safe_excel_autofill"], "1.0",
        "Write only approved/mapped values; preserve formulas/macros; execution report",
        ["mapping_plan", "rater_template_path"], ["rater_file_url", "autofill_report"],
        ["rater_template_path", "rater_output_dir"], "structured Excel filling across domains",
    ),
    "explainability_audit": _spec(
        _BY_NAME["explainability_audit"], "1.0",
        "Full reproducible decision record (deterministic run id)",
        ["*"], ["audit", "audit_run_id"], reuse="all enterprise AI workflows",
    ),
    "underwriter_copilot": _spec(
        _BY_NAME["underwriter_copilot"], "1.0",
        "Grounded evidence-backed Q&A over the submission (no fabrication)",
        ["extracted_fields", "audit"], ["copilot_ready", "copilot_sample"],
        reuse="any document Q&A/review workflow",
    ),
    "human_review_workbench": _spec(
        _BY_NAME["human_review_workbench"], "1.0",
        "Green/Yellow/Red review packet with evidence + suggested actions",
        ["extracted_fields", "validation_results", "missing_fields"], ["review_workbench"],
        reuse="any human-governed review framework",
    ),
    "learning": _spec(
        _BY_NAME["learning"], "1.0",
        "Improvement proposals from run signals (advisory; nothing auto-promotes)",
        ["gap_report", "parsed_documents", "mapping_plan"], ["learning_signals"],
        reuse="feedback/improvement loop across projects",
    ),
}


def get_agent(name: str) -> A.Agent:
    return AGENT_REGISTRY[name].instance


def describe_registry() -> List[Dict[str, Any]]:
    """Serializable catalog view (capabilities/IO/version/config) for planners/UI."""
    return [
        {
            "name": s.name,
            "version": s.version,
            "capabilities": s.capabilities,
            "inputs": s.inputs,
            "outputs": s.outputs,
            "config_keys": s.config_keys,
            "reuse": s.reuse,
        }
        for s in AGENT_REGISTRY.values()
    ]
