from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from config.config import settings
from api.routes.embeddings import router as embeddings_router
from api.routes.search import router as search_router
from core.embedding_manager import EmbeddingManager

logger.remove() 
logger.add(sys.stderr, level="INFO") 
logger.add("logs/app.log", rotation="10 MB", level="DEBUG")

def create_embedding_manager() -> EmbeddingManager:
    logger.info(f"Initializing EmbeddingManager with model: {settings.default_model_name} and cache: {settings.default_embedding_cache_path}")
    return EmbeddingManager(
        model_name=settings.default_model_name, 
        cache_dir=settings.default_embedding_cache_path,
        use_faiss_index=True,
        faiss_metric='cosine'
    )

app = FastAPI(
    title=settings.app_name,
    description="API for LLM Embedding Space Exploration",
    version="0.1.0"
)

app.state.embedding_manager = None

async def get_embedding_manager_dependency() -> EmbeddingManager:
    if app.state.embedding_manager is None:
        logger.warning("EmbeddingManager not initialized in app.state, creating now.")
        app.state.embedding_manager = create_embedding_manager()
    return app.state.embedding_manager

origins = [
        "http://localhost:3000", 
        "http://localhost:3001", 
    ]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"], 
        allow_headers=["*"], 
    )

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting up {settings.app_name}...")
    app.state.embedding_manager = create_embedding_manager()
    logger.info("EmbeddingManager initialized and attached to app.state.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.app_name}...")
    if hasattr(app.state, "embedding_manager") and app.state.embedding_manager is not None:
        logger.info("EmbeddingManager shutdown (placeholder - no specific cleanup action implemented).")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

from api.routes.search import get_embedding_manager as search_get_em_placeholder
app.dependency_overrides[search_get_em_placeholder] = get_embedding_manager_dependency

# Dependency override for routes in embeddings.py
from api.routes.embeddings import get_embedding_manager as embeddings_get_em_placeholder
app.dependency_overrides[embeddings_get_em_placeholder] = get_embedding_manager_dependency

app.include_router(embeddings_router, prefix="/api/v1/embeddings", tags=["Embeddings"])
app.include_router(search_router, prefix="/api/v1/search", tags=["Search"])

logger.info("FastAPI app configured with EmbeddingManager and all routers.") 