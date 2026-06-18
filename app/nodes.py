"""Node Types + declarative nebag workflow template (architecture doc §3 items
2, §4, §8).

Defines the reusable workflow building blocks — Start, Agent, Tool, If/Else,
HITL, End — and expresses the nebag template as a declarative list of typed nodes
(like an n8n/Langflow graph) while the LangGraph executor controls execution.

The executor (workflow.py) consumes the agent ORDER from this template; the
If/Else predicates and HITL pause are referenced here by name so the graph is
fully described in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class NodeKind(str, Enum):
    START = "start"
    AGENT = "agent"
    TOOL = "tool"
    IF_ELSE = "if_else"
    HITL = "hitl"
    END = "end"


@dataclass
class Node:
    kind: NodeKind
    name: str
    agent: Optional[str] = None       # registry key for AGENT/HITL nodes
    predicate: Optional[str] = None   # routing predicate name for IF_ELSE nodes
    on_true: Optional[str] = None     # agent to run when predicate is true
    note: str = ""


# The nebag template (doc §8), expressed with explicit node types. IF_ELSE nodes
# name the predicate (defined in workflow.py) and the branch agent.
nebag_TEMPLATE: List[Node] = [
    Node(NodeKind.START, "start"),
    Node(NodeKind.AGENT, "submission_intake", agent="submission_intake"),
    Node(NodeKind.AGENT, "pre_submission_context", agent="pre_submission_context"),
    Node(NodeKind.AGENT, "rater_template", agent="rater_template"),
    Node(NodeKind.AGENT, "document_ai", agent="document_ai"),
    Node(NodeKind.AGENT, "semantic_extraction", agent="semantic_extraction"),
    Node(NodeKind.AGENT, "post_extraction_context", agent="post_extraction_context"),
    Node(NodeKind.AGENT, "industry_product_detection", agent="industry_product_detection"),
    Node(NodeKind.AGENT, "canonical_schema", agent="canonical_schema"),
    Node(NodeKind.AGENT, "ontology_knowledge_graph", agent="ontology_knowledge_graph"),
    Node(NodeKind.AGENT, "missing_attribute_detection", agent="missing_attribute_detection"),
    Node(NodeKind.IF_ELSE, "if_missing", predicate="has_missing_fields", on_true="external_research",
         note="if missing fields -> External Research, else continue"),
    Node(NodeKind.AGENT, "truth_validation", agent="truth_validation"),
    Node(NodeKind.IF_ELSE, "if_review", predicate="needs_human_review", on_true="human_review",
         note="if validation fails/low confidence -> Human Review (HITL), else continue"),
    Node(NodeKind.HITL, "human_review", agent="human_review"),
    Node(NodeKind.AGENT, "risk_reasoning", agent="risk_reasoning"),
    Node(NodeKind.AGENT, "carrier_mapping", agent="carrier_mapping"),
    Node(NodeKind.AGENT, "safe_excel_autofill", agent="safe_excel_autofill"),
    Node(NodeKind.AGENT, "explainability_audit", agent="explainability_audit"),
    Node(NodeKind.AGENT, "underwriter_copilot", agent="underwriter_copilot"),
    Node(NodeKind.AGENT, "human_review_workbench", agent="human_review_workbench"),
    Node(NodeKind.AGENT, "learning", agent="learning"),
    Node(NodeKind.END, "end"),
]


def template_agent_order() -> List[str]:
    """Agent names in template order (excludes Start/End/If-Else marker nodes)."""
    return [n.agent for n in nebag_TEMPLATE if n.agent and n.kind in (NodeKind.AGENT, NodeKind.HITL)]
