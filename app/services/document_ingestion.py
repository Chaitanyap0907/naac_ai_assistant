"""
app/services/document_ingestion.py
Ingests NAAC documents (PDF, DOCX, TXT) into a FAISS vector store using
LangChain text splitters and sentence-transformer embeddings.

Uses FAISS (faiss-cpu) instead of ChromaDB to avoid sqlite3 >= 3.35
requirement that fails on Streamlit Cloud (Ubuntu 20.04 / sqlite 3.31).
"""
from __future__ import annotations

import os
import pickle
import hashlib
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

from langchain_community.vectorstores import FAISS

from app.core import get_settings, logger

_settings = get_settings()

# ── Loader map by extension ────────────────────────────────────────────────────
_LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".md":   TextLoader,
}

# FAISS index stored relative to this file's project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FAISS_DIR    = str(_PROJECT_ROOT / "data" / "faiss_index")
_FAISS_PATH   = str(_PROJECT_ROOT / "data" / "faiss_index" / "naac_index")


def _get_embeddings() -> HuggingFaceEmbeddings:
    logger.info("Loading sentence-transformer embedding model (all-MiniLM-L6-v2)")
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_or_create_store(embeddings) -> FAISS:
    """Load existing FAISS index from disk, or return None if not yet created."""
    index_file = _FAISS_PATH + ".faiss"
    if os.path.exists(index_file):
        logger.info("Loading existing FAISS index from disk")
        return FAISS.load_local(
            _FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return None


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def ingest_directory(directory: str) -> int:
    """
    Load all supported documents from *directory*, split, embed,
    and upsert into FAISS.  Returns number of new chunks added.
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
                for doc in docs:
                    doc.metadata.update({
                        "source_file": file_path.name,
                        "source_dir":  str(file_path.parent),
                        "file_hash":   _file_hash(file_path),
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
    existing = _load_or_create_store(embeddings)

    Path(_FAISS_DIR).mkdir(parents=True, exist_ok=True)

    if existing is None:
        store = FAISS.from_documents(chunks, embeddings)
    else:
        store = existing
        store.add_documents(chunks)

    store.save_local(_FAISS_PATH)
    logger.info(f"Saved FAISS index with {len(chunks)} new chunks")
    return len(chunks)


def ingest_all() -> dict:
    """Ingest NAAC docs, SSR templates, and past reports."""
    results = {}
    for label, path in [
        ("naac_guidelines", _settings.naac_docs_dir),
        ("ssr_templates",   _settings.ssr_templates_dir),
        ("past_reports",    _settings.reports_dir),
    ]:
        results[label] = ingest_directory(path)
    return results


def get_retriever(top_k: Optional[int] = None):
    """Return a LangChain retriever backed by FAISS."""
    k = top_k or _settings.rag_top_k
    embeddings = _get_embeddings()
    index_file = _FAISS_PATH + ".faiss"
    if not os.path.exists(index_file):
        logger.warning("FAISS index not found — running ingest_all() first")
        ingest_all()
    store = FAISS.load_local(
        _FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
