"""NevagAgent — THE single clean boundary.

Everything the host needs goes through this class. Host integration = construct
one NevagAgent (with injected Settings + optional LLM/store) and call run()/resume().
No globals, no host coupling here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agents import AgentContext
from .config import Settings, get_settings
from .copilot import answer as copilot_answer
from .document_ai import build_ocr_engine
from .contracts import (
    ExtractedField,
    FieldStatus,
    IntakeMode,
    NevagInput,
    NevagResult,
)
from .embeddings import EmbeddingProvider, build_embedding_provider
from .intake import gather_files
from .llm import LLMClient, build_llm
from .observability import configure_logging, get_logger, now_ms
from .retrieval import Retriever
from .store import StateStore, build_state_store
from .vectorstore import VectorStore, build_vector_store, validate_vector_store_config
from .workflow import WorkflowState, build_checkpointer, build_workflow


class NevagAgent:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[LLMClient] = None,
        store: Optional[StateStore] = None,
        embedder: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.settings = settings or get_settings()
        configure_logging(self.settings)
        self._log = get_logger("service")
        self.llm = llm or build_llm(self.settings)
        self.store = store or build_state_store(self.settings)
        # Build the OCR engine once (probes for Azure DI / Tesseract); injected.
        self._ocr = build_ocr_engine(self.settings)
        # Retrieval subsystem: embeddings + vector store (Qdrant/Azure/memory).
        self.embedder = embedder or build_embedding_provider(self.settings)
        self.vector_store = vector_store or build_vector_store(self.settings, self.embedder.dim)
        self.vector_store_status = validate_vector_store_config(self.settings)
        # Hybrid retriever facade (dense ⊕ BM25 → RRF → MMR → rerank → gate).
        self.retriever = Retriever(self.settings, self.embedder, self.vector_store)
        # Inject retrieval deps so agents (Semantic Extraction RAG, Validation) use them.
        self._ctx = AgentContext(
            settings=self.settings, llm=self.llm, ocr=self._ocr,
            embedder=self.embedder, vector_store=self.vector_store, retriever=self.retriever,
        )
        # One checkpointer shared by the graph (Postgres if configured, else Memory).
        self._checkpointer = build_checkpointer(self.settings)
        self._workflow = build_workflow(self._ctx, checkpointer=self._checkpointer)

    def _ingest(self, out: WorkflowState) -> None:
        """Chunk + embed + upsert parsed docs into the vector store + BM25 index."""
        if out.get("_ingested"):  # already ingested during RAG extraction
            out.setdefault("retrieval", out.get("retrieval", {}))
            return
        try:
            report = self.retriever.ingest(out.get("parsed_documents", []), out["submission_id"])
            out["retrieval"] = report
            out.setdefault("status_trail", []).append(
                f"retrieval: ingested {report['chunks']} chunks "
                f"({report['backend']} store, {report['embedder']} embeddings)"
            )
        except Exception as exc:  # never fail the run because of retrieval
            out["retrieval"] = {"chunks": 0, "error": f"{type(exc).__name__}: {exc}"}
            out.setdefault("status_trail", []).append(f"retrieval: skipped ({exc})")

    # --- main entry --------------------------------------------------------- #
    def run(self, inp: NevagInput) -> NevagResult:
        files = gather_files(inp, self.settings)
        state: WorkflowState = {
            "query": inp.query,
            "chat_history": inp.chat_history,
            "app_user_id": inp.app_user_id or "",
            "files": files,
            "status_trail": [],
        }
        if inp.submission_id:
            state["submission_id"] = inp.submission_id
            state["thread_id"] = inp.submission_id

        start_ms = now_ms()
        out: WorkflowState = self._workflow.run(state, thread_id=state.get("thread_id"))
        # Carry a stable thread id for a later resume to continue the checkpoint.
        out.setdefault("thread_id", out["submission_id"])
        # Retrieval ingestion: chunk -> embed -> upsert (Qdrant/memory/...).
        self._ingest(out)
        self._summarize_metrics(out, start_ms)
        self._persist(out["submission_id"], out)
        self._log.info(
            "run complete",
            extra={"submission_id": out["submission_id"],
                   "duration_ms": out["metrics"]["total_ms"], "event": "run"},
        )
        return self._to_result(out)

    def _persist(self, submission_id: str, out: WorkflowState) -> None:
        """Save state, slimmed by default: drop raw file bytes and full document
        text (rehydratable / not needed for resume), keep doc metadata + decisions."""
        if not getattr(self.settings, "state_persist_slim", True):
            self.store.save(submission_id, out)
            return
        slim = dict(out)
        slim.pop("files", None)                  # raw bytes / base64 — never persist
        slim.pop("attribute_candidates", None)   # transient (recomputed pre-review)
        slim["parsed_documents"] = [
            {
                "filename": d.get("filename"), "kind": d.get("kind"),
                "parse_status": d.get("parse_status"), "parser": d.get("parser"),
                "sha1": d.get("sha1"), "page_count": len(d.get("pages", []) or []),
            }
            for d in (out.get("parsed_documents") or [])
        ]
        self.store.save(submission_id, slim)

    @staticmethod
    def _summarize_metrics(out: WorkflowState, start_ms: float) -> None:
        results = out.get("validation_results", [])
        metrics = out.setdefault("metrics", {})
        metrics["total_ms"] = round(now_ms() - start_ms, 2)
        metrics["counts"] = {
            "documents": len(out.get("parsed_documents", [])),
            "extracted": len(out.get("extracted_fields", [])),
            "approved": sum(1 for r in results if r.get("status") == "approved"),
            "review": sum(1 for r in results if r.get("status") == "review"),
            "rejected": sum(1 for r in results if r.get("status") == "rejected"),
            "missing": len(out.get("missing_fields", [])),
            "review_required": bool(out.get("review_required")),
            "chunks_indexed": (out.get("retrieval") or {}).get("chunks", 0),
        }

    # --- retrieval (similarity search over the submission's chunks) --------- #
    def retrieve(self, submission_id: str, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Grounded hybrid search, scoped to one submission by default."""
        k = int(top_k or self.settings.retrieval_top_k)
        return self.retriever.search(query, top_k=k, filters={"submission_id": submission_id})

    # --- HITL resume -------------------------------------------------------- #
    def resume(self, submission_id: str, corrections: List[Dict[str, Any]]) -> NevagResult:
        """Apply human corrections and CONTINUE from the HITL checkpoint.

        Unlike a full re-run, this picks up just after the human_review node and
        runs the remaining nodes (risk -> mapping -> autofill -> audit)."""
        state = self.store.load(submission_id)
        if state is None:
            raise KeyError(f"Unknown submission_id: {submission_id}")
        by_name = {c["name"]: c for c in corrections}
        for f in state.get("extracted_fields", []):
            if f["name"] in by_name:
                c = by_name[f["name"]]
                f["value"] = c.get("value", f.get("value"))
                f["status"] = "approved"
                f["confidence"] = 1.0
                f["notes"] = "human override"
        state["review_required"] = False
        out = self._workflow.resume(state, thread_id=state.get("thread_id"))
        self._persist(submission_id, out)
        return self._to_result(out)

    # --- copilot (agent #17): grounded Q&A over a processed submission ------ #
    def ask(self, submission_id: str, question: str) -> Dict[str, Any]:
        """Answer an evidence-backed question about a processed submission. Reads
        only this submission's stored state; never fabricates."""
        state = self.store.load(submission_id)
        if state is None:
            raise KeyError(f"Unknown submission_id: {submission_id}")
        retrieved = self.retrieve(submission_id, question)
        return copilot_answer(question, state, self.llm, retrieved=retrieved)

    # --- mapping ------------------------------------------------------------ #
    @staticmethod
    def _to_result(state: WorkflowState) -> NevagResult:
        fields = [
            ExtractedField(
                name=f["name"],
                value=f.get("value"),
                confidence=f.get("confidence", 0.0),
                status=FieldStatus(f.get("status", "missing")),
                source_document=f.get("source_document"),
                source_page=f.get("source_page"),
                evidence_snippet=f.get("evidence_snippet"),
                notes=f.get("notes"),
            )
            for f in state.get("extracted_fields", [])
        ]
        missing = state.get("missing_fields", [])
        review = state.get("review_required", False)
        n_ok = sum(1 for f in fields if f.status == FieldStatus.APPROVED)
        summary = (
            f"Processed submission {state.get('submission_id')}: "
            f"{n_ok}/{len(state.get('rater_required_attributes', []))} rater attributes "
            f"approved, {len(missing)} missing. "
            + ("Human review required." if review else "Ready for autofill.")
        )
        return NevagResult(
            submission_id=state["submission_id"],
            summary=summary,
            fields=fields,
            missing_fields=missing,
            review_required=review,
            rater_file_url=state.get("rater_file_url"),
            audit_run_id=state.get("audit_run_id"),
            submission_context=state.get("submission_context"),
            confirmed_context=state.get("confirmed_context"),
            classification=state.get("classification"),
            graph_facts=state.get("graph_facts", []),
            gap_report=state.get("gap_report"),
            risk_assessment=state.get("risk_assessment"),
            review_workbench=state.get("review_workbench"),
            learning_signals=state.get("learning_signals"),
            retrieval=state.get("retrieval"),
            metrics=state.get("metrics"),
            audit=state.get("audit"),
            status_trail=state.get("status_trail", []),
        )
