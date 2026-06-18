"""The 19 reusable agents (architecture doc §6) as pluggable workflow nodes.

Every agent is generic code whose behavior comes from CONFIG (schema/catalog/
ontology/taxonomy/mapping/rules), takes run(state, ctx) -> state, never reaches
for globals, and appends a status line so progress can be streamed. Credential-
dependent intelligence (LLM/OCR/external APIs) is optional and degrades to
deterministic, evidence-grounded behavior.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .audit import build_audit
from .carrier_mapping import CarrierMapping, build_mapping_plan, load_carrier_mapping
from .classification import classify, load_taxonomy
from .common import bundled_config, condition_matches, load_json_config
from .config import Settings
from .copilot import answer as copilot_answer
from .document_ai import OCREngine, parse_documents
from .excel_autofill import fill_rater, load_value_mappings
from .extraction import (
    build_candidate_map,
    coerce,
    extract_fields,
    extract_fields_rag,
    load_catalog,
    reconcile_fields,
    spec_for,
)
from .intake import expand_and_fingerprint
from .llm import LLMClient
from .ontology import derive_facts, load_ontology
from .rater_template import analyze_rater_workbook, discover_rater_requirements
from .research import build_research_sources, research_missing
from .validation import validate_fields


@dataclass
class AgentContext:
    """Injected dependencies shared by all agents (no globals)."""

    settings: Settings
    llm: LLMClient
    ocr: Optional[OCREngine] = None       # OCR backend for Document AI (auto-built if None)
    embedder: Optional[Any] = None        # embedding provider (retrieval/RAG)
    vector_store: Optional[Any] = None    # vector store (retrieval/RAG)
    retriever: Optional[Any] = None       # hybrid Retriever facade (ingest + search)
    # store, mapping_registry ... injected here later.


class Agent:
    """Base node. Subclasses set `name` and implement `_run`."""

    name: str = "agent"

    def run(self, state: Dict[str, Any], ctx: AgentContext) -> Dict[str, Any]:
        state.setdefault("status_trail", []).append(f"run {self.name}")
        return self._run(state, ctx)

    def _run(self, state: Dict[str, Any], ctx: AgentContext) -> Dict[str, Any]:  # noqa
        return state


# --------------------------------------------------------------------------- #
# 1. Submission Intake & Email Orchestration
# --------------------------------------------------------------------------- #
class SubmissionIntakeAgent(Agent):
    """Packages the submission: expands ZIPs, fingerprints files by hash, detects
    duplicates / version lineage, and assigns a deterministic submission id."""

    name = "submission_intake"

    def _run(self, state, ctx):
        files = state.get("files", [])
        expanded, lineage = expand_and_fingerprint(files)
        state["files"] = expanded
        state["file_lineage"] = lineage
        if not state.get("submission_id"):
            shas = sorted(l["sha1"] for l in lineage if l.get("sha1"))
            seed = (state.get("query", "") + "|".join(shas)).encode()
            state["submission_id"] = "sub_" + hashlib.sha1(seed).hexdigest()[:12]
        dups = sum(1 for l in lineage if l["duplicate"])
        zipped = sum(1 for l in lineage if l.get("from_zip"))
        state.setdefault("status_trail", []).append(
            f"submission_intake: {len(expanded)} files ({dups} duplicate, {zipped} from zip)"
        )
        return state


# --------------------------------------------------------------------------- #
# 2. Pre-Submission Context & Routing (early, from raw email)
# --------------------------------------------------------------------------- #
class PreSubmissionContextAgent(Agent):
    """Lightweight early understanding from the RAW email (before parsing):
    channel, broker domain, insured hint, new vs renewal. Marks uncertainty
    rather than forcing a decision."""

    name = "pre_submission_context"

    def _run(self, state, ctx):
        info = {"channel": "upload", "kind": "new_business", "broker": None, "insured_hint": None}
        blob = (state.get("query", "") or "").lower()
        for f in state.get("files", []):
            fname = (f.get("filename") or "").lower()
            if fname.endswith((".eml", ".msg")):
                info["channel"] = "email"
                hdr = _peek_email_headers(f)
                frm = hdr.get("from", "")
                if "@" in frm:
                    info["broker"] = frm.split("@")[-1].strip(" >")
                info["insured_hint"] = hdr.get("subject") or info["insured_hint"]
                blob += " " + (hdr.get("subject", "") or "").lower()
        if "renewal" in blob:
            info["kind"] = "renewal"
        state["submission_context"] = info
        state.setdefault("status_trail", []).append(
            f"pre_submission_context: channel={info['channel']}, kind={info['kind']}"
            + (f", broker={info['broker']}" if info["broker"] else "")
        )
        return state


def _peek_email_headers(f: Dict[str, Any]) -> Dict[str, str]:
    """Read just the From/Subject from a raw .eml without full parsing."""
    from .intake import _file_bytes

    data = _file_bytes(f)
    if not data or (f.get("filename") or "").lower().endswith(".msg"):
        return {}
    try:
        import email
        from email import policy

        msg = email.message_from_bytes(data, policy=policy.default)
        return {"from": str(msg.get("From", "")), "subject": str(msg.get("Subject", ""))}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# 14. Rater Template & Excel Dependency Intelligence (analyzed EARLY)
# --------------------------------------------------------------------------- #
class RaterTemplateAgent(Agent):
    """Rater-driven: reads the blank carrier rater to DISCOVER required attributes
    + their target cells, and analyzes workbook behavior (formulas, macros, hidden
    tabs, dropdowns, dependency graph, fingerprint) so autofill can protect it.
    Falls back to the catalog attribute set when no rater workbook is configured."""

    name = "rater_template"

    def _run(self, state, ctx):
        catalog = load_catalog(ctx.settings)
        path = ctx.settings.rater_template_path or state.get("rater_template_path")
        trail = state.setdefault("status_trail", [])
        if path and os.path.exists(path):
            try:
                required, cell_map, sheet = discover_rater_requirements(path, catalog)
                state["rater_analysis"] = analyze_rater_workbook(path)
                if required:
                    state["rater_required_attributes"] = required
                    state["rater_cell_map"] = {"sheet": sheet, "cells": cell_map}
                    state["rater_fill_order"] = list(cell_map.keys())  # inputs first; formulas recalc
                    fa = state["rater_analysis"]
                    trail.append(
                        f"rater_template: discovered {len(required)} attributes (sheet '{sheet}'); "
                        f"{len(fa['formulas'])} formulas, {len(fa['dropdowns'])} dropdowns, "
                        f"macros={fa['has_macros']}, fingerprint={fa['fingerprint']}"
                    )
                    return state
                trail.append("rater_template: no labels matched; using catalog set")
            except Exception as exc:
                trail.append(f"rater_template: rater parse failed ({exc}); using catalog set")
        else:
            trail.append("rater_template: no rater workbook configured; using catalog set")
        state["rater_required_attributes"] = list(catalog.keys())
        return state


# --------------------------------------------------------------------------- #
# 3. Intelligent Document AI
# --------------------------------------------------------------------------- #
class DocumentAIAgent(Agent):
    """Turns raw files into structured, page-level parsed_documents (digital PDF,
    scanned PDF/image OCR, Excel, Word, email). Each parser degrades gracefully."""

    name = "document_ai"

    def _run(self, state, ctx):
        files = state.get("files", [])
        parsed = parse_documents(files, ctx.settings, ctx.ocr)
        state["parsed_documents"] = parsed
        ok = sum(1 for d in parsed if d.get("parse_status") == "ok")
        attn = [d["filename"] for d in parsed if d.get("parse_status") in ("failed", "partial", "skipped")]
        trail = state.setdefault("status_trail", [])
        trail.append(f"document_ai: parsed {ok}/{len(parsed)} documents ok")
        if attn:
            trail.append("document_ai: needs attention -> " + ", ".join(attn))
        return state


# --------------------------------------------------------------------------- #
# 4. Semantic Extraction & Cross-Document Intelligence
# --------------------------------------------------------------------------- #
class SemanticExtractionAgent(Agent):
    """Rater-driven, attribute-by-attribute extraction with evidence. LLM (Azure
    GPT-4.1) with a strict-JSON grounded prompt; deterministic heuristic fallback
    with zero credentials. Every field carries value/confidence/source/snippet."""

    name = "semantic_extraction"

    def _run(self, state, ctx):
        required = state.get("rater_required_attributes", [])
        documents = state.get("parsed_documents", [])
        trail = state.setdefault("status_trail", [])
        catalog = load_catalog(ctx.settings)
        specs = [spec_for(n, catalog) for n in required]

        # Shared candidate map (text + tables), computed ONCE per run and reused
        # by Missing-Attribute + Truth-Validation for contradiction (no re-scan).
        cmap = build_candidate_map(specs, documents, ctx.settings)
        state["attribute_candidates"] = {
            name: [
                {"value": c["value"], "source_document": c.get("source_document"),
                 "source_page": c.get("source_page"), "source_type": c.get("source_type")}
                for c in cands
            ]
            for name, cands in cmap.items()
        }

        mode = "full_text"
        fields = None
        if ctx.settings.rag_extraction_enabled and ctx.retriever is not None:
            try:
                report = ctx.retriever.ingest(documents, state.get("submission_id", ""))
                state["_ingested"] = True
                state["retrieval"] = report
                fields = extract_fields_rag(
                    required, documents, ctx.llm, ctx.settings, ctx.retriever, state.get("submission_id", ""),
                )
                if fields is not None:
                    mode = "rag"
            except Exception as exc:  # never fail extraction on a retrieval problem
                trail.append(f"semantic_extraction: RAG failed ({type(exc).__name__}: {exc}); full-text fallback")
                fields = None

        if fields is None:  # deterministic full-text path (reuses the shared map)
            fields = extract_fields(required, documents, ctx.llm, ctx.settings, catalog=catalog, candidate_map=cmap)

        # Reconcile duplicate values per attribute -> one authoritative field
        # (conflicts annotated for the validation layer to route to review).
        fields = reconcile_fields(fields, ctx.settings, catalog)
        state["extracted_fields"] = fields
        found = {f["name"] for f in fields}
        via = {f.get("extractor") for f in fields}
        trail.append(
            f"semantic_extraction: extracted {len(found)}/{len(required)} attributes "
            f"[{mode}]" + (f" (via {', '.join(sorted(v for v in via if v))})" if via else "")
        )
        return state


# --------------------------------------------------------------------------- #
# 5. Post-Extraction Context Intelligence
# --------------------------------------------------------------------------- #
class PostExtractionContextAgent(Agent):
    """Confirms business context using actual extracted evidence (not just the
    email subject). Flags conflicts between the early hint and the evidence."""

    name = "post_extraction_context"

    def _run(self, state, ctx):
        by = {f["name"]: f for f in state.get("extracted_fields", [])}
        pre = state.get("submission_context", {}) or {}
        insured = by.get("insured_name", {}).get("value")
        industry = by.get("industry", {}).get("value")
        conflicts: List[str] = []
        hint = (pre.get("insured_hint") or "")
        if insured and hint and str(insured).split()[0].lower() not in hint.lower():
            conflicts.append(f"email hint '{hint}' vs extracted insured '{insured}'")
        state["confirmed_context"] = {
            "insured": insured,
            "industry": industry,
            "kind": pre.get("kind"),
            "evidence_backed": insured is not None,
            "conflicts": conflicts,
        }
        state.setdefault("status_trail", []).append(
            f"post_extraction_context: insured={insured!r}"
            + (f", {len(conflicts)} conflict(s)" if conflicts else "")
        )
        return state


# --------------------------------------------------------------------------- #
# 6. Industry & Insurance Product Detection
# --------------------------------------------------------------------------- #
class IndustryProductDetectionAgent(Agent):
    """Classifies industry + likely product from a config-driven taxonomy and the
    extracted evidence. Unknown routes to a default rather than blocking."""

    name = "industry_product_detection"

    def _run(self, state, ctx):
        taxonomy = load_taxonomy(ctx.settings)
        industry_field = next(
            (f.get("value") for f in state.get("extracted_fields", []) if f["name"] == "industry"),
            None,
        )
        text = " ".join((d.get("text", "") or "") for d in state.get("parsed_documents", []))
        result = classify(text, industry_field, taxonomy)
        state["classification"] = result
        state.setdefault("status_trail", []).append(
            f"industry_product_detection: {result['industry']} -> {result['product']} "
            f"(conf {result['confidence']})"
        )
        return state


# --------------------------------------------------------------------------- #
# 7. Canonical Schema
# --------------------------------------------------------------------------- #
class CanonicalSchemaAgent(Agent):
    """Normalizes extracted values into one standardized record with type coercion
    and source provenance, under a versioned schema."""

    name = "canonical_schema"

    def _run(self, state, ctx):
        catalog = load_catalog(ctx.settings)
        record: Dict[str, Any] = {}
        provenance: Dict[str, Any] = {}
        for f in state.get("extracted_fields", []):
            spec = spec_for(f["name"], catalog)
            record[f["name"]] = coerce(f.get("value"), spec.dtype)
            provenance[f["name"]] = {
                "source_document": f.get("source_document"),
                "source_page": f.get("source_page"),
                "confidence": f.get("confidence"),
                "dtype": spec.dtype,
            }
        state["canonical_record"] = record
        state["canonical_provenance"] = provenance
        state["schema_version"] = ctx.settings.canonical_schema_version
        state.setdefault("status_trail", []).append(
            f"canonical_schema: {len(record)} fields normalized (schema {state['schema_version']})"
        )
        return state


# --------------------------------------------------------------------------- #
# 8. Ontology & Knowledge Graph
# --------------------------------------------------------------------------- #
class OntologyKnowledgeGraphAgent(Agent):
    """Derives evidence-backed relationship facts from config-driven ontology
    rules (e.g. MFA -> reduces -> account takeover risk)."""

    name = "ontology_knowledge_graph"

    def _run(self, state, ctx):
        rules = load_ontology(ctx.settings)
        facts = derive_facts(state.get("canonical_record", {}), rules)
        state["graph_facts"] = facts
        state.setdefault("status_trail", []).append(
            f"ontology_knowledge_graph: {len(facts)} evidence-backed fact(s)"
        )
        return state


# --------------------------------------------------------------------------- #
# 9. Missing Attribute Detection
# --------------------------------------------------------------------------- #
class MissingAttributeAgent(Agent):
    """Separates required-field gaps into Not Found / OCR Failed / Low Confidence /
    Contradiction so downstream routing and review are precise."""

    name = "missing_attribute_detection"

    def _run(self, state, ctx):
        required = state.get("rater_required_attributes", [])
        by = {f["name"]: f for f in state.get("extracted_fields", [])}
        docs = state.get("parsed_documents", [])
        candidates = state.get("attribute_candidates", {})  # shared map (text+tables)
        min_conf = float(ctx.settings.validation_min_confidence)
        ocr_issue = any(
            d.get("parse_status") in ("partial", "failed") and not d.get("text") for d in docs
        )

        not_found, ocr_failed, low_conf, contradiction = [], [], [], []
        for attr in required:
            f = by.get(attr)
            if f is None:
                (ocr_failed if ocr_issue else not_found).append(attr)
                continue
            if float(f.get("confidence", 0)) < min_conf:
                low_conf.append(attr)
            if len({str(c["value"]) for c in candidates.get(attr, [])}) > 1:
                contradiction.append(attr)

        state["gap_report"] = {
            "not_found": not_found,
            "ocr_failed": ocr_failed,
            "low_confidence": low_conf,
            "contradiction": contradiction,
        }
        # Truly-absent fields route to research/review.
        state["missing_fields"] = sorted(set(not_found) | set(ocr_failed))
        state.setdefault("status_trail", []).append(
            f"missing_attribute_detection: not_found={len(not_found)}, ocr_failed={len(ocr_failed)}, "
            f"low_conf={len(low_conf)}, contradiction={len(contradiction)}"
        )
        return state


# --------------------------------------------------------------------------- #
# 10. External Intelligence & Research (conditional)
# --------------------------------------------------------------------------- #
class ExternalResearchAgent(Agent):
    """Runs only when fields are missing. Queries APPROVED sources only and never
    invents values — researched values come back as 'review'."""

    name = "external_research"

    def _run(self, state, ctx):
        missing = state.get("missing_fields", [])
        if not missing:
            return state
        sources = build_research_sources(ctx.settings)
        trail = state.setdefault("status_trail", [])
        if not sources:
            trail.append(
                f"external_research: {len(missing)} missing, no approved sources configured "
                "(no values invented)"
            )
            return state
        found = research_missing(
            missing,
            state.get("submission_context", {}),
            sources,
            default_confidence=float(ctx.settings.research_default_confidence),
            default_status=ctx.settings.research_default_status,
        )
        if found:
            state.setdefault("extracted_fields", []).extend(found)
            present = {f["name"] for f in state["extracted_fields"]}
            state["missing_fields"] = [m for m in missing if m not in present]
        trail.append(f"external_research: resolved {len(found)}/{len(missing)} from approved sources")
        return state


# --------------------------------------------------------------------------- #
# 11. Truth Validation & Hallucination Prevention
# --------------------------------------------------------------------------- #
class TruthValidationAgent(Agent):
    """Evidence gates: evidence presence, grounding (anti-hallucination), confidence,
    type, config-driven constraints, cross-document contradiction. Only safe values
    stay approved; the rest route to review/rejected."""

    name = "truth_validation"

    def _run(self, state, ctx):
        fields = state.get("extracted_fields", [])
        documents = state.get("parsed_documents", [])
        results, review = validate_fields(
            fields, documents, ctx.settings,
            retriever=ctx.retriever, submission_id=state.get("submission_id"),
            candidate_map=state.get("attribute_candidates"),
        )
        if state.get("missing_fields"):
            review = True
        state["validation_results"] = results
        state["review_required"] = review
        n_rejected = sum(1 for r in results if r["status"] == "rejected")
        n_review = sum(1 for r in results if r["status"] == "review")
        n_ok = sum(1 for r in results if r["status"] == "approved")
        state.setdefault("status_trail", []).append(
            f"truth_validation: {n_ok} approved, {n_review} review, {n_rejected} rejected"
        )
        return state


# --------------------------------------------------------------------------- #
# (HITL node) Human Review pause
# --------------------------------------------------------------------------- #
class HumanReviewAgent(Agent):
    """Conditional HITL node: pauses at a resumable checkpoint instead of blocking."""

    name = "human_review"

    def _run(self, state, ctx):
        state.setdefault("status_trail", []).append("HITL: review pending (resumable)")
        return state


# --------------------------------------------------------------------------- #
# 12. Risk Analysis & Reasoning
# --------------------------------------------------------------------------- #
class RiskReasoningAgent(Agent):
    """Transparent, evidence-constrained risk signals from approved values (and
    ontology facts). Each flag is explainable; deterministic scoring bands."""

    name = "risk_reasoning"

    def _run(self, state, ctx):
        approved = {
            f["name"]: f.get("value")
            for f in state.get("extracted_fields", [])
            if f.get("status") == "approved"
        }
        cfg_path = ctx.settings.risk_rules_path or bundled_config("risk_rules.json")
        cfg = load_json_config(cfg_path, default={"rules": [], "bands": []})
        rules = cfg.get("rules", [])
        bands = sorted(cfg.get("bands", []), key=lambda b: b.get("min_score", 0), reverse=True)

        flags: List[str] = []
        score = 0
        for rule in rules:
            attr = rule.get("attribute")
            if attr in approved and condition_matches(rule.get("when", {}), approved[attr]):
                flags.append(rule.get("flag", attr))
                score += int(rule.get("weight", 1))
        level = next((b["level"] for b in bands if score >= int(b.get("min_score", 0))), "standard")

        state["risk_assessment"] = {
            "level": level,
            "score": score,
            "flags": flags,
            "graph_facts": state.get("graph_facts", []),
            "rationale": "; ".join(flags) or "no elevated risk signals in approved data",
        }
        state.setdefault("status_trail", []).append(
            f"risk_reasoning: level={level} (score {score})" + (f" ({', '.join(flags)})" if flags else "")
        )
        return state


# --------------------------------------------------------------------------- #
# 13. Carrier Translation & Mapping
# --------------------------------------------------------------------------- #
class CarrierMappingAgent(Agent):
    """Canonical attributes -> carrier rater cells. Prefers the rater-derived cell
    map (fully rater-driven); falls back to the carrier config. Approved only."""

    name = "carrier_mapping"

    def _run(self, state, ctx):
        discovered = state.get("rater_cell_map")
        if discovered and discovered.get("cells"):
            mapping = CarrierMapping(
                carrier="rater-derived",
                sheet=discovered.get("sheet"),
                cells=dict(discovered["cells"]),
            )
        else:
            mapping = load_carrier_mapping(ctx.settings)
        plan = build_mapping_plan(state.get("extracted_fields", []), mapping)
        state["mapping_plan"] = plan
        trail = state.setdefault("status_trail", [])
        trail.append(
            f"carrier_mapping: {len(plan['cells'])} approved fields mapped (carrier '{plan['carrier']}')"
        )
        if plan["unmapped_approved"]:
            trail.append("carrier_mapping: no cell for -> " + ", ".join(plan["unmapped_approved"]))
        return state


# --------------------------------------------------------------------------- #
# 15. Safe Excel Autofill Execution
# --------------------------------------------------------------------------- #
class SafeExcelAutofillAgent(Agent):
    """Writes ONLY approved, mapped values into a COPY of the rater, preserving
    formulas/macros and producing an execution report. Skips if no template."""

    name = "safe_excel_autofill"

    def _run(self, state, ctx):
        sid = state.get("submission_id", "submission")
        plan = state.get("mapping_plan") or {}
        trail = state.setdefault("status_trail", [])
        template = ctx.settings.rater_template_path or state.get("rater_template_path")

        if not template or not os.path.exists(template):
            state["rater_file_url"] = None
            state["autofill_report"] = {
                "written": 0,
                "skipped_unmapped": len(plan.get("unmapped_approved", [])),
                "note": "no rater template configured -> write skipped",
            }
            trail.append(
                f"safe_excel_autofill: no rater template -> skipped (plan ready: "
                f"{len(plan.get('cells', {}))} cells)"
            )
            return state

        out_dir = ctx.settings.rater_output_dir or os.path.join(os.getcwd(), "output")
        ext = os.path.splitext(template)[1] or ".xlsx"
        out_path = os.path.join(out_dir, f"{sid}_filled{ext}")
        try:
            n, written = fill_rater(template, plan, out_path, load_value_mappings(ctx.settings))
            state["rater_file_url"] = "file://" + os.path.abspath(out_path).replace("\\", "/")
            state["autofill_report"] = {
                "written": n,
                "cells": [{"attribute": a, "cell": c} for a, c in written],
                "skipped_unmapped": len(plan.get("unmapped_approved", [])),
                "formulas_preserved": len((state.get("rater_analysis") or {}).get("formulas", [])),
                "recalc_note": "values written; Excel recalculates formulas on open",
            }
            trail.append(f"safe_excel_autofill: wrote {n} approved values -> {out_path}")
        except Exception as exc:
            state["rater_file_url"] = None
            state["autofill_report"] = {"written": 0, "error": f"{type(exc).__name__}: {exc}"}
            trail.append(f"safe_excel_autofill: FAILED ({type(exc).__name__}: {exc})")
        return state


# --------------------------------------------------------------------------- #
# 16. Explainability & Audit
# --------------------------------------------------------------------------- #
class ExplainabilityAuditAgent(Agent):
    """Compiles the full, reproducible decision record (provenance, per-field
    extraction+validation+write outcomes, counts, context, risk)."""

    name = "explainability_audit"

    def _run(self, state, ctx):
        audit = build_audit(state, ctx.settings)
        state["audit"] = audit
        state["audit_run_id"] = audit["run_id"]
        state.setdefault("status_trail", []).append(
            f"explainability_audit: run {audit['run_id']} "
            f"({audit['counts']['approved']} approved, {audit['counts']['cells_written']} written)"
        )
        return state


# --------------------------------------------------------------------------- #
# 17. Underwriter Copilot & Conversational AI
# --------------------------------------------------------------------------- #
class UnderwriterCopilotAgent(Agent):
    """Prepares the grounded Q&A surface. Real questions go through
    nebagAgent.ask(); here we mark readiness and a sample grounded answer."""

    name = "underwriter_copilot"

    def _run(self, state, ctx):
        state["copilot_ready"] = True
        # A grounded sample answer to confirm the surface works offline.
        state["copilot_sample"] = copilot_answer("what is missing?", state, ctx.llm)
        state.setdefault("status_trail", []).append("underwriter_copilot: ready (grounded Q&A)")
        return state


# --------------------------------------------------------------------------- #
# 18. Human Review Workbench
# --------------------------------------------------------------------------- #
class HumanReviewWorkbenchAgent(Agent):
    """Compiles the Green/Yellow/Red review packet for the human workbench:
    approved (green), review (yellow), rejected + missing (red), each with
    evidence and a suggested action. Corrections are applied via resume()."""

    name = "human_review_workbench"

    def _run(self, state, ctx):
        green, yellow, red = [], [], []
        for f in state.get("extracted_fields", []):
            item = {
                "attribute": f["name"],
                "value": f.get("value"),
                "confidence": f.get("confidence"),
                "evidence_snippet": f.get("evidence_snippet"),
                "source_document": f.get("source_document"),
                "reasons": f.get("validation_reasons", []),
            }
            status = f.get("status")
            if status == "approved":
                green.append(item)
            elif status == "rejected":
                item["suggested_action"] = "correct or reject"
                red.append(item)
            else:
                item["suggested_action"] = "confirm or correct"
                yellow.append(item)
        for attr in state.get("missing_fields", []):
            red.append({"attribute": attr, "value": None, "suggested_action": "provide value"})

        state["review_workbench"] = {
            "green": green,
            "yellow": yellow,
            "red": red,
            "requires_human": bool(yellow or red),
        }
        state.setdefault("status_trail", []).append(
            f"human_review_workbench: green={len(green)}, yellow={len(yellow)}, red={len(red)}"
        )
        return state


# --------------------------------------------------------------------------- #
# 19. Learning & Self-Optimization
# --------------------------------------------------------------------------- #
class LearningAgent(Agent):
    """Turns this run's signals (low-confidence fields, parse failures, unmapped
    attributes, contradictions, human corrections) into improvement PROPOSALS.
    Nothing auto-promotes — proposals are advisory only."""

    name = "learning"

    def _run(self, state, ctx):
        gap = state.get("gap_report", {}) or {}
        proposals: List[Dict[str, Any]] = []
        for attr in gap.get("low_confidence", []):
            proposals.append({"type": "alias_or_prompt", "attribute": attr,
                              "suggestion": f"add aliases / tune prompt for '{attr}' (low confidence)"})
        for attr in gap.get("contradiction", []):
            proposals.append({"type": "source_priority", "attribute": attr,
                              "suggestion": f"define source-priority rule for '{attr}' (conflicting values)"})
        for d in state.get("parsed_documents", []):
            if d.get("parse_status") in ("partial", "failed"):
                proposals.append({"type": "ocr_routing", "document": d.get("filename"),
                                  "suggestion": f"review OCR/parse for '{d.get('filename')}'"})
        for attr in (state.get("mapping_plan") or {}).get("unmapped_approved", []):
            proposals.append({"type": "mapping_gap", "attribute": attr,
                              "suggestion": f"add carrier cell mapping for '{attr}'"})
        state["learning_signals"] = {"proposals": proposals, "auto_promote": False}
        state.setdefault("status_trail", []).append(
            f"learning: {len(proposals)} improvement proposal(s) (advisory, no auto-promote)"
        )
        return state


# --------------------------------------------------------------------------- #
# nebag workflow template order (doc §8 + registry agents #5/#6/#8 wired in).
# external_research and human_review are gated by conditional edges in workflow.py.
# --------------------------------------------------------------------------- #
AGENT_PIPELINE: List[Agent] = [
    SubmissionIntakeAgent(),
    PreSubmissionContextAgent(),
    RaterTemplateAgent(),
    DocumentAIAgent(),
    SemanticExtractionAgent(),
    PostExtractionContextAgent(),
    IndustryProductDetectionAgent(),
    CanonicalSchemaAgent(),
    OntologyKnowledgeGraphAgent(),
    MissingAttributeAgent(),
    ExternalResearchAgent(),
    TruthValidationAgent(),
    HumanReviewAgent(),
    RiskReasoningAgent(),
    CarrierMappingAgent(),
    SafeExcelAutofillAgent(),
    ExplainabilityAuditAgent(),
    UnderwriterCopilotAgent(),
    HumanReviewWorkbenchAgent(),
    LearningAgent(),
]
