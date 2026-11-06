"""
streamlit_app.py
NAAC Accreditation AI Assistant — Streamlit UI
Powered by IBM Granite (watsonx.ai) + RAG (ChromaDB)

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub → connect on share.streamlit.io
"""
import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="NAAC Accreditation AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (avoid slowing down first render) ────────────────────────────
@st.cache_resource(show_spinner="Loading NAAC knowledge base...")
def init_pipeline():
    """Initialise vector store and ingest documents once per session."""
    from app.services.document_ingestion import ingest_all
    results = ingest_all()
    return results


@st.cache_resource(show_spinner=False)
def get_pipeline_fns():
    from app.services.rag_pipeline import (
        answer_question,
        draft_ssr_section,
        get_criteria_inputs,
        document_best_practice,
    )
    return answer_question, draft_ssr_section, get_criteria_inputs, document_best_practice


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* IBM-themed header */
.ibm-header {
    background: linear-gradient(135deg, #0f3460 0%, #0f62fe 100%);
    color: white;
    padding: 20px 28px;
    border-radius: 12px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.ibm-header h1 { font-size: 22px; margin: 0; font-weight: 700; }
.ibm-header p  { font-size: 12px; margin: 4px 0 0; opacity: 0.8; }
.ibm-badge {
    background: #0043ce;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
    white-space: nowrap;
}

/* Chat bubbles */
.user-bubble {
    background: #0f3460;
    color: white;
    border-radius: 14px 4px 14px 14px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 14px;
    line-height: 1.6;
}
.assistant-bubble {
    background: #f7f8fa;
    border: 1px solid #e5e7eb;
    border-radius: 4px 14px 14px 14px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 90%;
    font-size: 14px;
    line-height: 1.7;
    color: #1f2328;
}
.source-tag {
    font-size: 11px;
    color: #57606a;
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 4px 8px;
    margin-top: 6px;
    display: inline-block;
}

/* Criteria cards */
.criteria-card {
    background: #f0f4ff;
    border-left: 4px solid #0f62fe;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    cursor: pointer;
    font-size: 13px;
}
.criteria-card:hover { background: #e0e7ff; }

/* Section headings */
.section-head {
    font-size: 13px;
    font-weight: 700;
    color: #0f3460;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 12px 0 6px;
}
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ibm-header">
    <div class="ibm-badge">IBM<br>watsonx</div>
    <div>
        <h1>🎓 NAAC Accreditation AI Assistant</h1>
        <p>Powered by IBM Granite · RAG (ChromaDB) · NAAC Guidelines (All 7 Criteria)</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Init ──────────────────────────────────────────────────────────────────────
try:
    ingest_results = init_pipeline()
    total_chunks = sum(ingest_results.values())
    st.success(f"Knowledge base ready — {total_chunks} NAAC document chunks indexed", icon="✅")
except Exception as e:
    st.warning(f"Knowledge base init: {e}", icon="⚠️")

answer_question, draft_ssr_section, get_criteria_inputs, document_best_practice = get_pipeline_fns()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "qa"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛠️ Assistant Mode")
    mode = st.radio(
        "Select action",
        options=["qa", "ssr", "criteria", "practice", "upload"],
        format_func=lambda x: {
            "qa":       "❓  Ask a Question",
            "ssr":      "📝  Draft SSR Section",
            "criteria": "📋  Criteria-Wise Inputs",
            "practice": "⭐  Best Practice Doc",
            "upload":   "📁  Upload Document",
        }[x],
        label_visibility="collapsed",
    )
    st.session_state.mode = mode

    st.divider()
    st.markdown("### 📚 Quick Criteria")
    criteria_map = {
        "1": "C1 – Curricular Aspects",
        "2": "C2 – Teaching-Learning",
        "3": "C3 – Research & Extension",
        "4": "C4 – Infrastructure",
        "5": "C5 – Student Support",
        "6": "C6 – Governance",
        "7": "C7 – Values & Practices",
    }
    for num, label in criteria_map.items():
        if st.button(label, key=f"c{num}", use_container_width=True):
            st.session_state.mode = "criteria"
            st.session_state["quick_criteria"] = num

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#57606a;'>
    <b>IBM Granite</b> ibm/granite-4-h-small<br>
    <b>Embeddings</b> all-MiniLM-L6-v2<br>
    <b>Vector Store</b> ChromaDB<br>
    <b>Framework</b> LangChain + FastAPI
    </div>
    """, unsafe_allow_html=True)


# ── Main Area ─────────────────────────────────────────────────────────────────
col_input, col_chat = st.columns([1, 1.6])

# ── LEFT: Input Panel ─────────────────────────────────────────────────────────
with col_input:
    st.markdown(f"<div class='section-head'>{'❓ Ask a Question' if mode=='qa' else '📝 Draft SSR' if mode=='ssr' else '📋 Criteria Inputs' if mode=='criteria' else '⭐ Best Practice' if mode=='practice' else '📁 Upload Document'}</div>", unsafe_allow_html=True)

    result = None

    # ── Q&A mode ──────────────────────────────────────────────────────────────
    if mode == "qa":
        question = st.text_area(
            "Your question",
            placeholder="e.g. What documents are required for NAAC A++ certification?",
            height=120,
            label_visibility="collapsed",
        )
        if st.button("Ask IBM Granite AI", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("Retrieving context & generating answer..."):
                    try:
                        result = answer_question(question.strip())
                        st.session_state.messages.append({"role": "user", "content": question.strip()})
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                            "mode": "Q&A",
                        })
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a question.")

    # ── SSR Draft mode ────────────────────────────────────────────────────────
    elif mode == "ssr":
        criterion = st.selectbox("Criterion", [
            "1 – Curricular Aspects",
            "2 – Teaching-Learning & Evaluation",
            "3 – Research, Innovations & Extension",
            "4 – Infrastructure & Learning Resources",
            "5 – Student Support & Progression",
            "6 – Governance, Leadership & Management",
            "7 – Institutional Values & Best Practices",
        ])
        section = st.text_input("Section Name", placeholder="e.g. Curriculum Design and Development")
        inst_data = st.text_area(
            "Institution Data (optional)",
            placeholder="e.g. We have 45 programmes, 1200 students, 68 faculty, 3 MoUs...",
            height=100,
        )
        if st.button("Generate SSR Draft", type="primary", use_container_width=True):
            cnum = criterion.split(" ")[0]
            sec = section.strip() or "General Overview"
            with st.spinner("Drafting SSR section..."):
                try:
                    result = draft_ssr_section(cnum, sec, inst_data)
                    st.session_state.messages.append({"role": "user", "content": f"Draft SSR – Criterion {cnum}: {sec}"})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["draft"],
                        "sources": result["sources"],
                        "mode": "SSR Draft",
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Criteria mode ─────────────────────────────────────────────────────────
    elif mode == "criteria":
        quick = st.session_state.pop("quick_criteria", None)
        default_idx = int(quick) - 1 if quick else 0
        cnum = st.selectbox("Select Criterion", [
            "1 – Curricular Aspects",
            "2 – Teaching-Learning & Evaluation",
            "3 – Research, Innovations & Extension",
            "4 – Infrastructure & Learning Resources",
            "5 – Student Support & Progression",
            "6 – Governance, Leadership & Management",
            "7 – Institutional Values & Best Practices",
        ], index=default_idx)
        if st.button("Get Criteria Inputs", type="primary", use_container_width=True):
            num = cnum.split(" ")[0]
            with st.spinner("Generating criteria inputs..."):
                try:
                    result = get_criteria_inputs(num)
                    st.session_state.messages.append({"role": "user", "content": f"Criteria inputs for Criterion {num}"})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["inputs"],
                        "sources": result["sources"],
                        "mode": "Criteria Inputs",
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Best Practice mode ────────────────────────────────────────────────────
    elif mode == "practice":
        domain = st.text_input(
            "Domain / Activity",
            placeholder="e.g. Student Mentoring Program, Green Campus Initiative...",
        )
        if st.button("Generate Best Practice Doc", type="primary", use_container_width=True):
            if domain.strip():
                with st.spinner("Documenting best practice..."):
                    try:
                        result = document_best_practice(domain.strip())
                        st.session_state.messages.append({"role": "user", "content": f"Best practice: {domain.strip()}"})
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["documentation"],
                            "sources": result["sources"],
                            "mode": "Best Practice",
                        })
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a domain.")

    # ── Upload mode ───────────────────────────────────────────────────────────
    elif mode == "upload":
        category = st.selectbox("Document Category", [
            "naac_guidelines",
            "ssr_templates",
            "reports",
        ])
        uploaded = st.file_uploader(
            "Choose file (PDF, DOCX, TXT, MD)",
            type=["pdf", "docx", "txt", "md"],
        )
        if st.button("Upload & Index", type="primary", use_container_width=True):
            if uploaded:
                import shutil
                from pathlib import Path
                from app.core import get_settings
                from app.services.document_ingestion import ingest_directory
                s = get_settings()
                dir_map = {
                    "naac_guidelines": s.naac_docs_dir,
                    "ssr_templates": s.ssr_templates_dir,
                    "reports": s.reports_dir,
                }
                dest_dir = Path(dir_map[category])
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / uploaded.name
                with open(dest, "wb") as f:
                    f.write(uploaded.getbuffer())
                with st.spinner("Indexing document..."):
                    chunks = ingest_directory(str(dest_dir))
                st.success(f"Indexed '{uploaded.name}' — {chunks} chunks added to knowledge base.")
            else:
                st.warning("Please select a file first.")

    # ── Info box ──────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("ℹ️ About this Assistant"):
        st.markdown("""
**NAAC Accreditation AI Assistant** uses:
- **IBM Granite** (`granite-4-h-small`) on watsonx.ai Lite for text generation
- **ChromaDB** vector store with semantic search (top-5 retrieval)
- **LangChain RAG** pipeline to ground every answer in NAAC guidelines
- **Automatic fallback** to context synthesis if IBM API is unavailable

Configure credentials in `.env` → `IBM_CLOUD_API_KEY` & `IBM_PROJECT_ID`
        """)


# ── RIGHT: Chat History ───────────────────────────────────────────────────────
with col_chat:
    st.markdown("<div class='section-head'>💬 Conversation</div>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.info("👋 Select an action from the left panel and submit to start chatting with the NAAC AI Assistant.")
    else:
        chat_container = st.container(height=600)
        with chat_container:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"<div class='user-bubble'>👤 <b>You</b><br>{msg['content']}</div>", unsafe_allow_html=True)
                else:
                    badge = msg.get("mode", "Answer")
                    sources = msg.get("sources", [])
                    src_html = ""
                    if sources:
                        src_list = " · ".join([f"[{s['index']}] {s['source_file']}" for s in sources])
                        src_html = f"<div class='source-tag'>📚 Sources: {src_list}</div>"
                    content = msg["content"].replace("\n", "<br>")
                    st.markdown(
                        f"<div class='assistant-bubble'>"
                        f"🤖 <b>NAAC Assistant</b> <small>({badge})</small><br><br>"
                        f"{content}"
                        f"{src_html}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
