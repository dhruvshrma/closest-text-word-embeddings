from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from config import settings
from api.routes.embeddings import router as embeddings_router

logger.remove() 
logger.add(sys.stderr, level="INFO") 
logger.add("logs/app.log", rotation="10 MB", level="DEBUG")
app = FastAPI(
    title=settings.app_name,
    description="API for LLM Embedding Space Exploration",
    version="0.1.0"
)

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

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.app_name}...")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

app.include_router(embeddings_router, prefix="/api/v1/embeddings", tags=["embeddings"])

logger.info("FastAPI app configured.") 