"""
app/routers/chat.py
REST endpoints for NAAC AI assistant interactions.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.rag_pipeline import (
    answer_question,
    draft_ssr_section,
    get_criteria_inputs,
    document_best_practice,
)
from app.core import logger

router = APIRouter(prefix="/api/chat", tags=["NAAC Assistant"])


# ── Request / Response Models ──────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000, description="Faculty's NAAC question")


class SSRDraftRequest(BaseModel):
    criterion: str = Field(..., description="NAAC Criterion number (e.g., '1', '2', '3')")
    section: str = Field(..., description="Section name (e.g., 'Curricular Aspects')")
    institution_data: str = Field(default="", description="Institution-specific data to include")


class CriteriaRequest(BaseModel):
    criterion_number: str = Field(..., description="NAAC Criterion number (1–7)")


class BestPracticeRequest(BaseModel):
    domain: str = Field(..., description="Domain or activity (e.g., 'Student Mentoring Program')")


class AssistantResponse(BaseModel):
    success: bool
    mode: str
    content: str
    sources: list[dict]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AssistantResponse, summary="Ask a NAAC Question")
async def ask_naac_question(req: QuestionRequest) -> AssistantResponse:
    """
    General-purpose NAAC Q&A endpoint.
    Retrieves relevant context from the knowledge base and generates
    a grounded answer using IBM Granite.
    """
    try:
        result = answer_question(req.question)
        return AssistantResponse(
            success=True,
            mode=result["mode"],
            content=result["answer"],
            sources=result["sources"],
        )
    except Exception as exc:
        logger.error(f"/ask error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/draft-ssr", response_model=AssistantResponse, summary="Draft SSR Section")
async def draft_ssr(req: SSRDraftRequest) -> AssistantResponse:
    """
    Generates a complete SSR section draft for the specified criterion
    using institution data + NAAC guidelines retrieved from ChromaDB.
    """
    try:
        result = draft_ssr_section(req.criterion, req.section, req.institution_data)
        return AssistantResponse(
            success=True,
            mode=result["mode"],
            content=result["draft"],
            sources=result["sources"],
        )
    except Exception as exc:
        logger.error(f"/draft-ssr error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/criteria-inputs", response_model=AssistantResponse, summary="Criteria-Wise Inputs")
async def criteria_inputs(req: CriteriaRequest) -> AssistantResponse:
    """
    Returns detailed criteria-wise inputs including key indicators,
    required evidence, and tips for NAAC compliance.
    """
    try:
        result = get_criteria_inputs(req.criterion_number)
        return AssistantResponse(
            success=True,
            mode=result["mode"],
            content=result["inputs"],
            sources=result["sources"],
        )
    except Exception as exc:
        logger.error(f"/criteria-inputs error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/best-practice", response_model=AssistantResponse, summary="Best Practice Documentation")
async def best_practice(req: BestPracticeRequest) -> AssistantResponse:
    """
    Generates structured best practice documentation for a given
    institutional domain, formatted per NAAC best-practice template.
    """
    try:
        result = document_best_practice(req.domain)
        return AssistantResponse(
            success=True,
            mode=result["mode"],
            content=result["documentation"],
            sources=result["sources"],
        )
    except Exception as exc:
        logger.error(f"/best-practice error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
