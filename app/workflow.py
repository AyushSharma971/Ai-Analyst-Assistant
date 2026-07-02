"""Workflow state + the Nevag workflow template (LangGraph) with checkpointed HITL.

Reusable orchestration skeleton from the architecture doc:
Start -> Intake -> Rater/Doc/Extraction/... -> If/Else (missing) ->
Validation -> If/Else (review) -> [HITL pause] -> Mapping -> Excel Autofill ->
Audit -> End.

Two executors behind one interface (`.run(state)` / `.resume(state)`):
  - LangGraph (intended): compiled with a checkpointer + `interrupt_before` on the
    human_review node, so a submission needing review PAUSES at a durable
    checkpoint; `resume()` continues from there (no re-running the whole graph).
  - Sequential fallback: same pause/continue semantics implemented directly, so
    the scaffold runs (and HITL works) before `pip install langgraph` and on
    Pythons where wheels may lag.

Checkpoint persistence: MemorySaver by default, PostgresSaver when
NEVAG_DATABASE_URL + langgraph's postgres extra are available (integration point;
not exercised without a live DB).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, TypedDict

from .agents import AgentContext, AGENT_PIPELINE


class WorkflowState(TypedDict, total=False):
    """Shared object passed node->node (architecture doc §4 'Workflow State')."""

    submission_id: str
    thread_id: str
    query: str
    chat_history: List[Dict[str, Any]]
    app_user_id: str

    files: List[Dict[str, Any]]
    file_lineage: List[Dict[str, Any]]
    parsed_documents: List[Dict[str, Any]]
    rater_required_attributes: List[str]
    rater_cell_map: Dict[str, Any]
    rater_analysis: Dict[str, Any]
    rater_fill_order: List[str]
    extracted_fields: List[Dict[str, Any]]
    canonical_record: Dict[str, Any]
    canonical_provenance: Dict[str, Any]
    schema_version: str
    classification: Dict[str, Any]
    graph_facts: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    gap_report: Dict[str, Any]
    missing_fields: List[str]
    review_required: bool
    workflow_paused: bool

    submission_context: Dict[str, Any]
    confirmed_context: Dict[str, Any]
    risk_assessment: Dict[str, Any]

    mapping_plan: Dict[str, Any]
    rater_file_url: str
    autofill_report: Dict[str, Any]
    audit_run_id: str
    audit: Dict[str, Any]
    copilot_ready: bool
    copilot_sample: Dict[str, Any]
    review_workbench: Dict[str, Any]
    learning_signals: Dict[str, Any]
    attribute_candidates: Dict[str, Any]
    metrics: Dict[str, Any]

    status_trail: List[str]


# --------------------------------------------------------------------------- #
# Conditional routing predicates (the If/Else nodes)
# --------------------------------------------------------------------------- #
def has_missing_fields(state: WorkflowState) -> bool:
    return bool(state.get("missing_fields"))


def needs_human_review(state: WorkflowState) -> bool:
    return bool(state.get("review_required"))


# Nodes that run AFTER the HITL checkpoint (used by both executors for resume).
_POST_REVIEW = (
    "risk_reasoning",
    "carrier_mapping",
    "safe_excel_autofill",
    "explainability_audit",
    "underwriter_copilot",
    "human_review_workbench",
    "learning",
)


# --------------------------------------------------------------------------- #
# Sequential fallback executor (used when langgraph is not installed)
# --------------------------------------------------------------------------- #
class _SequentialWorkflow:
    """Minimal stand-in for a checkpointed LangGraph graph.

    `run` executes Start->...->(pause at human_review if review needed).
    `resume` continues from just after human_review to End. Same input -> same
    output (deterministic), matching the doc's reproducibility requirement.
    """

    def __init__(self, ctx: AgentContext):
        self._ctx = ctx

    def _run_node(self, agent, state: WorkflowState) -> WorkflowState:
        """Run one node with timing + structured logging (metrics into state)."""
        from .observability import get_logger, now_ms, record_node

        start = now_ms()
        state = agent.run(state, self._ctx)
        dur = now_ms() - start
        record_node(state, agent.name, dur)
        get_logger("workflow").info(
            "node complete",
            extra={"node": agent.name, "duration_ms": round(dur, 2),
                   "submission_id": state.get("submission_id"), "event": "node"},
        )
        return state

    def run(self, state: WorkflowState, thread_id: Optional[str] = None) -> WorkflowState:
        for agent in AGENT_PIPELINE:
            if agent.name == "external_research" and not has_missing_fields(state):
                state.setdefault("status_trail", []).append(
                    "skip external_research (no missing fields)"
                )
                continue
            if agent.name == "human_review":
                if needs_human_review(state):
                    state["workflow_paused"] = True
                    state.setdefault("status_trail", []).append(
                        "PAUSE: awaiting human review (checkpoint saved)"
                    )
                    return state
                state.setdefault("status_trail", []).append(
                    "skip human_review (validation passed)"
                )
                continue
            state = self._run_node(agent, state)
        state["workflow_paused"] = False
        return state

    def resume(self, state: WorkflowState, thread_id: Optional[str] = None) -> WorkflowState:
        state.setdefault("status_trail", []).append(
            "RESUME: human review applied, continuing from checkpoint"
        )
        for agent in AGENT_PIPELINE:
            if agent.name in _POST_REVIEW:
                state = self._run_node(agent, state)
        state["workflow_paused"] = False
        return state


# --------------------------------------------------------------------------- #
# LangGraph executor wrapper (intended production path)
# --------------------------------------------------------------------------- #
class _LangGraphWorkflow:
    """Adapts a compiled, checkpointed LangGraph graph to run()/resume()."""

    def __init__(self, compiled):
        self._g = compiled

    def _cfg(self, state: WorkflowState, thread_id: Optional[str]) -> Dict[str, Any]:
        tid = thread_id or state.get("thread_id") or state.get("submission_id") or "default"
        return {"configurable": {"thread_id": tid}}

    def run(self, state: WorkflowState, thread_id: Optional[str] = None) -> WorkflowState:
        out = self._g.invoke(state, self._cfg(state, thread_id))
        # With interrupt_before=["human_review"], invoke returns the state at the
        # interrupt; downstream nodes (incl. audit) have not run yet.
        out["workflow_paused"] = needs_human_review(out) and not out.get("audit_run_id")
        return out

    def resume(self, state: WorkflowState, thread_id: Optional[str] = None) -> WorkflowState:
        cfg = self._cfg(state, thread_id)
        # Persist human corrections into the checkpoint, then continue past HITL.
        self._g.update_state(cfg, state)
        out = self._g.invoke(None, cfg)
        out["workflow_paused"] = False
        return out


# --------------------------------------------------------------------------- #
# Checkpointer + graph builders
# --------------------------------------------------------------------------- #
def build_checkpointer(settings):
    """Return a LangGraph checkpointer, or None if langgraph isn't installed.

    PostgresSaver when NEVAG_DATABASE_URL + langgraph-postgres are available
    (integration point), otherwise MemorySaver.
    """
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except Exception:
        return None

    if getattr(settings, "database_url", None):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            saver = PostgresSaver.from_conn_string(settings.database_url)
            saver.setup()
            return saver
        except Exception as exc:  # pragma: no cover - optional dep / no DB
            warnings.warn(
                f"PostgresSaver unavailable, using MemorySaver: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return MemorySaver()


def build_workflow(ctx: AgentContext, checkpointer: Any = None):
    """Build the executable workflow. Prefers LangGraph; falls back to sequential.

    Returns an object exposing `.run(state, thread_id)` and `.resume(state, thread_id)`.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:
        return _SequentialWorkflow(ctx)

    if checkpointer is None:
        checkpointer = build_checkpointer(ctx.settings)

    g = StateGraph(WorkflowState)

    def make_node(agent) -> Callable[[WorkflowState], WorkflowState]:
        return lambda state: agent.run(state, ctx)

    linear = [a for a in AGENT_PIPELINE if a.name not in ("external_research", "human_review")]
    for agent in linear:
        g.add_node(agent.name, make_node(agent))

    g.add_node("external_research", make_node(_by_name("external_research")))
    g.add_node("human_review", make_node(_by_name("human_review")))

    g.add_edge(START, linear[0].name)
    for i in range(len(linear) - 1):
        cur, nxt = linear[i].name, linear[i + 1].name
        if cur == "missing_attribute_detection":
            g.add_conditional_edges(
                cur,
                lambda s: "external_research" if has_missing_fields(s) else nxt,
                {"external_research": "external_research", nxt: nxt},
            )
            g.add_edge("external_research", nxt)
        elif cur == "truth_validation":
            g.add_conditional_edges(
                cur,
                lambda s: "human_review" if needs_human_review(s) else nxt,
                {"human_review": "human_review", nxt: nxt},
            )
            g.add_edge("human_review", nxt)
        else:
            g.add_edge(cur, nxt)
    g.add_edge(linear[-1].name, END)

    # Pause before HITL so review state can live in the store between calls.
    compiled = g.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"] if checkpointer else None,
    )
    return _LangGraphWorkflow(compiled)


def _by_name(name: str):
    for a in AGENT_PIPELINE:
        if a.name == name:
            return a
    raise KeyError(name)
