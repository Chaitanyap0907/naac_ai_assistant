"""
main.py
Application entry point -- run with:  python main.py
or:  uvicorn main:app --reload
"""
import uvicorn
from app import create_app
from app.core import get_settings
from app.services.document_ingestion import ingest_all

settings = get_settings()

# Create the FastAPI application
app = create_app()


if __name__ == "__main__":
    print("=" * 65)
    print("  NAAC Accreditation AI Assistant")
    print("  Powered by IBM Granite (watsonx.ai) + RAG (ChromaDB)")
    print("=" * 65)
    print("\nIngesting NAAC knowledge base documents...")
    results = ingest_all()
    total = sum(results.values())
    print(f"  Ingested {total} chunks total: {results}")
    print(f"\nStarting server at http://{settings.app_host}:{settings.app_port}")
    print(f"  API Docs -> http://localhost:{settings.app_port}/docs\n")

    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level="info",
    )
