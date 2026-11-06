"""
app/services/document_ingestion.py
Ingests NAAC documents (PDF, DOCX, TXT) into ChromaDB using
LangChain text splitters and local sentence-transformer embeddings.

Fallback: when IBM Slate embeddings are not available on the Lite plan,
the service transparently falls back to the open-source
`all-MiniLM-L6-v2` model from sentence-transformers.
"""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.core import get_settings, logger

_settings = get_settings()

# ── Loader map by extension ────────────────────────────────────────────────────
_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns sentence-transformer embeddings.
    Uses the compact all-MiniLM-L6-v2 model (runs locally, no quota).
    For production: swap to IBM Slate embeddings via langchain-ibm.
    """
    logger.info("Loading sentence-transformer embedding model (all-MiniLM-L6-v2)")
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Load (or create) persistent ChromaDB vector store."""
    return Chroma(
        collection_name="naac_knowledge_base",
        embedding_function=embeddings,
        persist_directory=_settings.chroma_persist_dir,
    )


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def ingest_directory(directory: str) -> int:
    """
    Load all supported documents from *directory*, split, embed, and
    upsert into ChromaDB.  Returns the number of new chunks added.
    """
    doc_dir = Path(directory)
    if not doc_dir.exists():
        logger.warning(f"Document directory not found: {doc_dir}")
        return 0

    documents = []
    for ext, loader_cls in _LOADER_MAP.items():
        for file_path in doc_dir.rglob(f"*{ext}"):
            try:
                logger.info(f"Loading document: {file_path.name}")
                loader = loader_cls(str(file_path))
                docs = loader.load()
                # Tag source metadata
                for doc in docs:
                    doc.metadata.update({
                        "source_file": file_path.name,
                        "source_dir": str(file_path.parent),
                        "file_hash": _file_hash(file_path),
                    })
                documents.extend(docs)
            except Exception as exc:
                logger.error(f"Failed to load {file_path}: {exc}")

    if not documents:
        logger.info("No documents found to ingest.")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_settings.embedding_chunk_size,
        chunk_overlap=_settings.embedding_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")

    embeddings = _get_embeddings()
    vector_store = _get_vector_store(embeddings)
    vector_store.add_documents(chunks)
    logger.info(f"Ingested {len(chunks)} chunks into ChromaDB")
    return len(chunks)


def ingest_all() -> dict:
    """Ingest NAAC docs, SSR templates, and past reports."""
    results = {}
    for label, path in [
        ("naac_guidelines", _settings.naac_docs_dir),
        ("ssr_templates", _settings.ssr_templates_dir),
        ("past_reports", _settings.reports_dir),
    ]:
        results[label] = ingest_directory(path)
    return results


def get_retriever(top_k: int | None = None):
    """Return a LangChain retriever backed by ChromaDB."""
    k = top_k or _settings.rag_top_k
    embeddings = _get_embeddings()
    vector_store = _get_vector_store(embeddings)
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
