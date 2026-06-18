"""Centralized, injected configuration for the nebag agent.

Integration rule: NOTHING in this service reads os.environ directly. Everything
flows through a Settings object that is *passed in* at construction time. This is
what lets nebag drop into the One AI chatbot (or any host) without colliding with
the host's own config / DB pools / clients.

All variables are namespaced with the nebag_ prefix so they never clash with the
host app's variables (BACKEND_API_URL, AZURE_OPENAI_*, etc.).

Single source of truth: fields are declared ONCE on the pydantic model below.
`get_settings()` overlays nebag_* environment variables (and an optional .env)
generically by iterating model fields — so there is no duplicated field list to
drift. Tests/hosts construct Settings(**kwargs) directly (no env coupling).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    """Runtime settings. Construct with kwargs, or via get_settings() for env."""

    model_config = ConfigDict(extra="ignore")

    # --- Service identity -----------------------------------------------------
    agent_name: str = "nebag Submission Triage"
    environment: str = "dev"

    # --- One AI registration (marketplace) ------------------------------------
    agent_source: str = "agent studio"
    agent_description: str = (
        "Underwriting submission triage: ingests messy submissions (email, PDF, "
        "scanned PDF, Excel, Word), extracts every rater-required attribute with "
        "evidence, validates it, and safely fills the carrier rater. Evidence-"
        "backed, governed, explainable."
    )
    agent_backend_url: Optional[str] = None
    oneai_register_url: Optional[str] = None

    # --- LLM (local-first: Ollama. azure_openai is an OPTIONAL adapter) -------
    llm_provider: str = "ollama"               # ollama | mock | azure_openai
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1"                # llama3.1 | qwen | mistral
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = "2024-12-01-preview"
    azure_openai_deployment: str = "gpt-4.1"
    llm_temperature: float = 0.0               # deterministic extraction
    llm_max_retries: int = 2
    llm_timeout_seconds: float = 30.0
    llm_retry_backoff_seconds: float = 0.0

    # --- Document AI / OCR ----------------------------------------------------
    ocr_provider: str = "auto"                 # auto | azure_doc_intel | tesseract | none
    azure_doc_intel_endpoint: Optional[str] = None
    azure_doc_intel_key: Optional[str] = None
    tesseract_cmd: Optional[str] = None
    scanned_text_threshold: int = 20           # < this many chars/page => OCR
    ocr_dpi: int = 200
    ocr_enable_fallback: bool = True
    doc_ai_max_workers: int = 1                 # parallel per-file parsing (1 = sequential)

    # --- Extraction (config-driven) -------------------------------------------
    attribute_catalog_path: Optional[str] = None
    prompts_path: Optional[str] = None
    value_tokens_path: Optional[str] = None
    number_units_path: Optional[str] = None    # magnitude multipliers (k/m/bn)
    extract_confidence_primary: float = 0.85
    extract_confidence_alias: float = 0.80
    extract_confidence_table: float = 0.82     # value found in a table row
    extract_default_confidence: float = 0.50
    extract_snippet_max_chars: int = 300
    extract_label_gap_max_chars: int = 40
    extract_enable_heuristic_fallback: bool = True
    # Duplicate-attribute reconciliation
    reconcile_numeric_rel_tolerance: float = 0.0   # >0 => treat near values as equal
    reconcile_conflict_to_review: bool = True       # materially different -> review

    # --- Validation -----------------------------------------------------------
    validation_min_confidence: float = 0.80
    validation_numeric_tolerance: float = 1e-9
    # Grounding tolerance (so correct scaled/translated values aren't false-flagged).
    grounding_string_min_overlap: float = 0.34       # token-overlap for string values
    grounding_numeric_scale_tolerant: bool = True    # match significant digits (scale-agnostic)
    grounding_numeric_min_significant_digits: int = 2
    # Source-priority resolution of contradictions (config-driven; see §7).
    contradiction_use_source_priority: bool = True
    source_priority_path: Optional[str] = None

    # --- Classification + ontology + risk -------------------------------------
    industry_taxonomy_path: Optional[str] = None
    ontology_rules_path: Optional[str] = None
    risk_rules_path: Optional[str] = None
    canonical_schema_version: str = "nebag-canonical-v1"
    prompt_version: str = "extraction-v1"

    # --- External research ----------------------------------------------------
    research_default_confidence: float = 0.50
    research_default_status: str = "review"
    perplexity_api_key: Optional[str] = None       # web research (agent #10)
    perplexity_model: str = "sonar-pro"

    # --- Registration ---------------------------------------------------------
    registration_timeout_seconds: float = 30.0

    # --- Carrier mapping + Safe Excel Autofill --------------------------------
    carrier_mapping_path: Optional[str] = None
    default_carrier: str = "default"
    rater_template_path: Optional[str] = None
    rater_output_dir: Optional[str] = None
    value_mappings_path: Optional[str] = None   # canonical -> carrier dropdown vocab

    # --- Retrieval: vector store + embeddings ---------------------------------
    vector_store: str = "qdrant"               # qdrant | azure_search | memory | none
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "nebag_documents"
    azure_search_endpoint: Optional[str] = None
    azure_search_api_key: Optional[str] = None
    azure_search_index: str = "nebag-documents"
    embedding_provider: str = "ollama"         # ollama | sentence_transformers | mock | azure_openai
    embedding_model: str = "bge-m3"
    embedding_deployment: str = "chatbot-text-embedding-ada-002"  # azure-only
    embedding_dim: int = 1024
    embedding_cache_enabled: bool = True
    embedding_cache_backend: str = "memory"    # memory | sqlite (persistent)
    embedding_cache_path: Optional[str] = None  # sqlite file for the vector cache
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieval_top_k: int = 5
    rag_extraction_enabled: bool = False
    rag_top_k: int = 4
    rag_per_attribute_prompts: bool = False     # one focused LLM call per attribute
    retrieval_min_score: float = 0.0
    ocr_confidence_floor: float = 0.0
    retrieval_hybrid: bool = True
    retrieval_prefetch_k: int = 20
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    mmr_enabled: bool = True
    mmr_lambda: float = 0.5
    mmr_pool: int = 12
    rerank_provider: str = "heuristic"         # heuristic | none | cross_encoder
    rerank_weight_base: float = 1.0
    rerank_weight_lexical: float = 0.5
    validation_use_retrieval: bool = False

    # --- Storage / DB ---------------------------------------------------------
    database_url: Optional[str] = None         # sqlite:///… | postgresql+psycopg://…
    blob_container_url: Optional[str] = None
    state_persist_slim: bool = True            # drop raw bytes/full text from stored state

    # --- Observability --------------------------------------------------------
    log_enabled: bool = True
    log_level: str = "WARNING"                 # DEBUG | INFO | WARNING | ERROR
    log_format: str = "text"                   # text | json

    # --- Intake ---------------------------------------------------------------
    mailbox_enabled: bool = False
    upload_enabled: bool = True
    # SharePoint document-library intake (ACS app-only auth). Config-driven.
    sharepoint_service_url: Optional[str] = None
    sharepoint_site_url: Optional[str] = None
    sharepoint_site_name: Optional[str] = None
    sharepoint_library_name: str = "Documents"
    sharepoint_tenant_id: Optional[str] = None
    sharepoint_client_id: Optional[str] = None
    sharepoint_client_secret: Optional[str] = None
    sharepoint_token_url: Optional[str] = None

    @property
    def api_slug(self) -> str:
        return self.agent_name.lower().replace(" ", "_")


# Host-app (One AI) env names that nebag fields may fall back to when the
# nebag_-prefixed var isn't set — so an existing host .env works without renaming.
# (Values are read at runtime; nothing is stored in code.)
_HOST_ALIASES: Dict[str, str] = {
    "azure_openai_endpoint": "AZURE_OPENAI_ENDPOINT",
    "azure_openai_api_key": "AZURE_OPENAI_API_KEY",
    "azure_openai_api_version": "TXTEMBD_API_VERSION",
    "embedding_deployment": "TXTEMBD_DEPLOYMENT_NAME",
    "embedding_model": "TXTEMBD_MODEL_NAME",
    "azure_search_endpoint": "AZURE_SEARCH_ENDPOINT",
    "azure_search_api_key": "AZURE_SEARCH_KEY",
    "azure_search_index": "AZURE_SEARCH_INDEX",
    "tesseract_cmd": "TESSERACT_CMD",
    "perplexity_api_key": "PERPLEXITY_API_KEY",
    "perplexity_model": "PERPLEXITY_MODEL_NAME",
    "sharepoint_service_url": "SHAREPOINT_SERVICE_URL",
    "sharepoint_site_url": "SH_SITE_URL",
    "sharepoint_site_name": "SHAREPOINT_SITE_NAME",
    "sharepoint_library_name": "SHAREPOINT_LIBRARY_NAME",
    "sharepoint_tenant_id": "SH_TENANT_ID",
    "sharepoint_client_id": "SH_CLIENT_ID",
    "sharepoint_client_secret": "SH_CLIENT_SECRET",
    "sharepoint_token_url": "AD_TOKEN_URL",
}


def _env_overlay() -> Dict[str, Any]:
    """Collect nebag_* values from .env (if python-dotenv present) + os.environ,
    mapped to field names. Falls back to known host-app names (One AI) when a
    nebag_ var is absent. Pydantic coerces the string values to field types."""
    sources: Dict[str, str] = {}
    try:  # optional .env support; os.environ always wins
        from dotenv import dotenv_values

        sources.update({k: v for k, v in dotenv_values(".env").items() if v is not None})
    except Exception:
        pass
    sources.update(os.environ)

    out: Dict[str, Any] = {}
    for name in Settings.model_fields:
        env_key = "nebag_" + name.upper()
        if env_key in sources and sources[env_key] != "":
            out[name] = sources[env_key]
        elif name in _HOST_ALIASES and sources.get(_HOST_ALIASES[name]):
            out[name] = sources[_HOST_ALIASES[name]]
    return out


@lru_cache
def get_settings() -> "Settings":
    """Default settings singleton for standalone runs — overlays nebag_* env vars.

    The host app should NOT rely on this; it should construct Settings(...) with
    its own values and inject them into nebagAgent. This cache exists only for
    convenient local/standalone execution.
    """
    return Settings(**_env_overlay())
