# app/services/__init__.py
from .ibm_granite import get_granite_client, GraniteClient
from .document_ingestion import ingest_all, ingest_directory, get_retriever
from .rag_pipeline import answer_question, draft_ssr_section, get_criteria_inputs, document_best_practice

__all__ = [
    "get_granite_client",
    "GraniteClient",
    "ingest_all",
    "ingest_directory",
    "get_retriever",
    "answer_question",
    "draft_ssr_section",
    "get_criteria_inputs",
    "document_best_practice",
]
