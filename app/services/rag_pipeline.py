"""
app/services/rag_pipeline.py
Core RAG pipeline that combines:
  1. ChromaDB retrieval (semantic search over NAAC knowledge base)
  2. IBM Granite LLM (ibm/granite-13b-chat-v2) for grounded generation

NAAC-specific prompt templates are defined here for:
  - General Q&A
  - SSR section drafting
  - Criteria-wise input generation
  - Best practice documentation
"""
from __future__ import annotations

from typing import Optional

from app.core import get_settings, logger
from app.services.document_ingestion import get_retriever
from app.services.ibm_granite import get_granite_client

_settings = get_settings()

# ── Prompt Templates ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert NAAC (National Assessment and Accreditation Council) 
Accreditation Advisor AI assistant for Indian higher education institutions. 
You help faculty draft SSR (Self Study Report) documents, provide criteria-wise inputs, 
identify best practices, and ensure compliance with NAAC guidelines.

Always base your responses on the retrieved context. If the context does not contain 
sufficient information, clearly state so and provide general NAAC guidance.
Be precise, professional, and use formal academic language suitable for accreditation reports."""


def _build_qa_prompt(question: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}

### Retrieved Context from NAAC Knowledge Base:
{context}

### Faculty Question:
{question}

### Answer:
"""


def _build_ssr_prompt(criterion: str, section: str, context: str, institution_data: str) -> str:
    return f"""{SYSTEM_PROMPT}

### Retrieved Context from NAAC Guidelines & Past Reports:
{context}

### Institution-Specific Data Provided:
{institution_data}

### Task:
Draft the SSR section for Criterion {criterion} - {section}.
Write in formal academic language. Use evidence-based statements, include metric references, 
and structure the response with clear headings and bullet points where appropriate.

### SSR Draft:
"""


def _build_criteria_prompt(criterion_number: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}

### Retrieved Context:
{context}

### Task:
Provide a comprehensive criteria-wise input summary for NAAC Criterion {criterion_number}.
Include: key indicators, required data points, evidence needed, common pitfalls, and tips 
for achieving higher grades.

### Criteria-Wise Input:
"""


def _build_best_practice_prompt(domain: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}

### Retrieved Context:
{context}

### Task:
Document best practices for "{domain}" that align with NAAC accreditation requirements. 
Format as: Practice Title, Objectives, Context & Perspective, Practice Description, 
Evidence of Success, Problems Encountered, and Resources Required.

### Best Practice Documentation:
"""


# ── Pipeline Functions ─────────────────────────────────────────────────────────

def _retrieve_context(query: str, top_k: Optional[int] = None) -> tuple[str, list[dict]]:
    """Retrieve relevant chunks and return (formatted_context, source_metadata)."""
    retriever = get_retriever(top_k=top_k)
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant context found in the knowledge base.", []

    context_parts = []
    sources = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Source {i}] {doc.page_content}")
        sources.append({
            "index": i,
            "source_file": doc.metadata.get("source_file", "unknown"),
            "page": doc.metadata.get("page", "N/A"),
        })

    return "\n\n".join(context_parts), sources


def answer_question(question: str) -> dict:
    """
    General NAAC Q&A using RAG.
    Returns answer text + source references.
    """
    logger.info(f"RAG Q&A query: {question[:80]}...")
    context, sources = _retrieve_context(question)
    prompt = _build_qa_prompt(question, context)
    client = get_granite_client()
    answer = client.generate(prompt, context_only=context)
    return {"answer": answer, "sources": sources, "mode": "qa"}


def draft_ssr_section(
    criterion: str,
    section: str,
    institution_data: str = "",
) -> dict:
    """
    Draft an SSR section for the given criterion using RAG + Granite.
    """
    query = f"NAAC SSR Criterion {criterion} {section} guidelines requirements"
    logger.info(f"Drafting SSR: Criterion {criterion} – {section}")
    context, sources = _retrieve_context(query)
    prompt = _build_ssr_prompt(criterion, section, context, institution_data)
    client = get_granite_client()
    draft = client.generate(prompt, context_only=context)
    return {"draft": draft, "criterion": criterion, "section": section, "sources": sources, "mode": "ssr"}


def get_criteria_inputs(criterion_number: str) -> dict:
    """
    Generate criteria-wise inputs and guidance for faculty.
    """
    query = f"NAAC Criterion {criterion_number} key metrics indicators evidence"
    logger.info(f"Criteria inputs for: Criterion {criterion_number}")
    context, sources = _retrieve_context(query)
    prompt = _build_criteria_prompt(criterion_number, context)
    client = get_granite_client()
    inputs = client.generate(prompt, context_only=context)
    return {"inputs": inputs, "criterion": criterion_number, "sources": sources, "mode": "criteria"}


def document_best_practice(domain: str) -> dict:
    """
    Generate best practice documentation for a given domain/activity.
    """
    query = f"best practices {domain} NAAC accreditation higher education"
    logger.info(f"Best practice documentation for: {domain}")
    context, sources = _retrieve_context(query)
    prompt = _build_best_practice_prompt(domain, context)
    client = get_granite_client()
    doc = client.generate(prompt, context_only=context)
    return {"documentation": doc, "domain": domain, "sources": sources, "mode": "best_practice"}
