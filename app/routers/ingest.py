"""
app/routers/ingest.py
Document ingestion endpoints — upload and index NAAC documents.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel

from app.services.document_ingestion import ingest_directory, ingest_all
from app.core import get_settings, logger

router = APIRouter(prefix="/api/ingest", tags=["Document Ingestion"])
_settings = get_settings()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class IngestResponse(BaseModel):
    success: bool
    message: str
    chunks_added: int


@router.post("/upload", response_model=IngestResponse, summary="Upload a NAAC document")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(default="naac_guidelines", description="naac_guidelines | ssr_templates | reports"),
) -> IngestResponse:
    """
    Upload a PDF/DOCX/TXT document and ingest it into the knowledge base.
    Category controls which sub-directory it lands in.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    category_map = {
        "naac_guidelines": _settings.naac_docs_dir,
        "ssr_templates": _settings.ssr_templates_dir,
        "reports": _settings.reports_dir,
    }
    target_dir = category_map.get(category, _settings.naac_docs_dir)
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(target_dir) / file.filename

    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Uploaded: {file.filename} → {target_dir}")
    finally:
        await file.close()

    chunks = ingest_directory(target_dir)
    return IngestResponse(
        success=True,
        message=f"'{file.filename}' ingested successfully.",
        chunks_added=chunks,
    )


@router.post("/ingest-all", response_model=dict, summary="Re-ingest all documents")
async def ingest_all_documents() -> dict:
    """
    Re-scan all document directories and re-index everything into ChromaDB.
    """
    try:
        results = ingest_all()
        total = sum(results.values())
        return {"success": True, "results": results, "total_chunks": total}
    except Exception as exc:
        logger.error(f"Ingest-all error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
