"""
app/core/config.py
Application settings loaded from environment variables via pydantic-settings.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── IBM watsonx.ai ──────────────────────────────────────────────────────────
    ibm_cloud_api_key: str = ""
    ibm_cloud_url: str = "https://us-south.ml.cloud.ibm.com"
    ibm_project_id: str = ""

    # ── IBM Granite models ──────────────────────────────────────────────────────
    granite_model_id: str = "ibm/granite-4-h-small"
    granite_instruct_model_id: str = "ibm/granite-4-h-small"
    granite_embedding_model_id: str = "ibm/slate-125m-english-rtrvr-v2"

    # ── Vector store ────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    embedding_chunk_size: int = 512
    embedding_chunk_overlap: int = 64

    # ── Application ─────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 5050
    app_debug: bool = True
    app_title: str = "NAAC Accreditation AI Assistant"
    app_version: str = "1.0.0"

    # ── RAG ─────────────────────────────────────────────────────────────────────
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.3
    rag_max_tokens: int = 1024
    rag_temperature: float = 0.3

    # ── Document paths ──────────────────────────────────────────────────────────
    naac_docs_dir: str = "./data/naac_documents"
    ssr_templates_dir: str = "./data/ssr_templates"
    reports_dir: str = "./data/reports"


@lru_cache
def get_settings() -> Settings:
    return Settings()
