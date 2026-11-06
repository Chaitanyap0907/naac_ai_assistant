# NAAC Accreditation AI Assistant 🎓

> AI-powered faculty assistant for NAAC accreditation, built with **IBM Granite (watsonx.ai)** and **RAG (ChromaDB)**.

---

## 🏗️ Architecture

```
Faculty Browser
      │
      ▼
FastAPI Backend  ←──── IBM Granite (watsonx.ai Lite)
      │                    ibm/granite-13b-chat-v2
      ▼
RAG Pipeline
      │
      ▼
ChromaDB (Vector Store)
      │
      ▼
NAAC Documents (PDF/DOCX/TXT)
 ├── NAAC Guidelines
 ├── SSR Templates
 └── Past Reports
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💬 NAAC Q&A | Ask any accreditation question, grounded in retrieved documents |
| 📝 SSR Drafting | Auto-generate Self Study Report sections by criterion |
| 📋 Criteria Inputs | Metrics, evidence requirements, grading guidance for all 7 criteria |
| ⭐ Best Practices | NAAC-formatted best practice documentation |
| 📁 Document Upload | Upload & index PDF/DOCX/TXT into the knowledge base |

---

## 🚀 Quick Start

### 1. Clone / Extract the project
```bash
cd ibm_agent
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure IBM Cloud credentials

Copy the example file and fill in your credentials:
```bash
copy .env.example .env      # Windows
cp .env.example .env        # Linux/macOS
```

Edit `.env`:
```env
IBM_CLOUD_API_KEY=your_actual_api_key
IBM_PROJECT_ID=your_watsonx_project_id
IBM_CLOUD_URL=https://us-south.ml.cloud.ibm.com
```

> **Get free IBM Cloud Lite credentials:**
> 1. Register at https://cloud.ibm.com (free, no credit card needed)
> 2. Create a **watsonx.ai** project at https://dataplatform.cloud.ibm.com
> 3. Go to **Manage → Access** → copy the Project ID
> 4. Go to **Manage → API Keys** → create an IBM Cloud API key

### 5. Run the application
```bash
python main.py
```

Open http://localhost:8000 in your browser.

API documentation: http://localhost:8000/docs

---

## 📁 Project Structure

```
ibm_agent/
├── main.py                    # Entry point
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
├── app/
│   ├── __init__.py            # FastAPI app factory
│   ├── core/
│   │   ├── config.py          # pydantic-settings config
│   │   └── logging.py         # Loguru logger
│   ├── services/
│   │   ├── ibm_granite.py     # IBM Granite LLM wrapper
│   │   ├── document_ingestion.py  # ChromaDB ingestion
│   │   └── rag_pipeline.py    # RAG pipeline + prompts
│   └── routers/
│       ├── chat.py            # Chat API endpoints
│       └── ingest.py          # Document upload endpoints
├── data/
│   ├── naac_documents/        # NAAC guidelines (auto-indexed)
│   ├── ssr_templates/         # SSR templates
│   ├── reports/               # Past reports
│   └── chroma_db/             # ChromaDB vector store (auto-created)
├── static/
│   ├── css/style.css          # Frontend styles
│   └── js/app.js              # Frontend logic
└── templates/
    └── index.html             # Faculty web UI
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat/ask` | General NAAC Q&A |
| POST | `/api/chat/draft-ssr` | Draft SSR section |
| POST | `/api/chat/criteria-inputs` | Criteria-wise guidance |
| POST | `/api/chat/best-practice` | Best practice docs |
| POST | `/api/ingest/upload` | Upload & index document |
| POST | `/api/ingest/ingest-all` | Re-index all documents |
| GET | `/health` | Health check |

---

## 🛠️ IBM Technologies Used

| Technology | Role |
|-----------|------|
| **IBM watsonx.ai** | Hosted LLM platform (Lite tier – free) |
| **IBM Granite 13B Chat v2** | Core language model for generation |
| **IBM Slate Embeddings** | (Optional) IBM embedding model |
| **IBM Cloud IAM** | API key authentication |

---

## 📚 NAAC Coverage

All 7 NAAC criteria are covered:
- **C1** – Curricular Aspects
- **C2** – Teaching-Learning & Evaluation  
- **C3** – Research, Innovations & Extension
- **C4** – Infrastructure & Learning Resources
- **C5** – Student Support & Progression
- **C6** – Governance, Leadership & Management
- **C7** – Institutional Values & Best Practices

---

## 🔧 Customization

**Add your institution's documents:**
- Drop PDF/DOCX/TXT files into `data/naac_documents/`
- Re-run: `POST /api/ingest/ingest-all`

**Change the Granite model:**
```env
GRANITE_MODEL_ID=ibm/granite-20b-code-instruct-v2
```

**Adjust RAG parameters:**
```env
RAG_TOP_K=7
RAG_MAX_TOKENS=2048
RAG_TEMPERATURE=0.2
```
