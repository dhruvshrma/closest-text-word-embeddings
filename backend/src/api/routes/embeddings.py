from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os # Keep os for path joining for cache_dir
from loguru import logger


from core.embedding_manager import EmbeddingManager
from config import settings

router = APIRouter()

EMBEDDING_MANAGER_INSTANCE = None

class TextPayload(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    text: str
    embedding: List[float]

class ModelInfo(BaseModel):
    model_name: str
    description: str


def get_embedding_manager():
    global EMBEDDING_MANAGER_INSTANCE
    if EMBEDDING_MANAGER_INSTANCE is None:
        try:
            cache_dir = os.path.join(os.path.dirname(__file__), '../../../../data/cache')
            os.makedirs(cache_dir, exist_ok=True)

            EMBEDDING_MANAGER_INSTANCE = EmbeddingManager(
                model_name=settings.default_model_name,
                cache_dir=cache_dir
            )
            logger.info(f"EmbeddingManager initialized for router with model: {settings.default_model_name}")
            logger.info(f"Router EmbeddingManager cache directory: {EMBEDDING_MANAGER_INSTANCE.cache_dir.resolve()}")
        except Exception as e:
            logger.exception(f"Failed to initialize EmbeddingManager for router: {e}")
            raise RuntimeError(f"Could not initialize EmbeddingManager for router: {e}")
    return EMBEDDING_MANAGER_INSTANCE

@router.on_event("startup")
async def startup_router_event():
    logger.info("Initializing embedding manager for embeddings router...")
    get_embedding_manager()
    logger.info("Embedding manager for router initialized and ready.")


@router.post("/embed", response_model=EmbeddingResponse, summary="Get embedding for text")
async def get_embedding_api(payload: TextPayload):
    manager = get_embedding_manager()
    try:
        logger.debug(f"Router: Received request to embed text: \"{payload.text[:50]}...\"")
        embedding_array = manager.get_embedding(payload.text)
        logger.debug(f"Router: Successfully generated embedding for: \"{payload.text[:50]}...\"")
        return EmbeddingResponse(text=payload.text, embedding=embedding_array.tolist())
    except Exception as e:
        logger.exception(f"Router: Error processing /embed request for text \"{payload.text[:50]}...\": {e}")
        raise HTTPException(status_code=500, detail="Error generating embedding")

@router.get("/models", response_model=List[ModelInfo], summary="List available models")
async def list_models_api():
    manager = get_embedding_manager()
    try:
        logger.debug("Router: Received request to list models.")
        return [
            ModelInfo(model_name=manager.model_name, description=settings.default_model_description)
        ]
    except Exception as e:
        logger.exception(f"Router: Error processing /models request: {e}")
        raise HTTPException(status_code=500, detail="Error listing models")

@router.get("/", summary="Get embedding router information", description="Basic info for the embeddings router.")
async def get_embeddings_router_info():
    logger.info("GET request to /embeddings/ router base")
    return {"message": "This is the embeddings API router. Use /embed or /models."}

logger.info("Embeddings API router configured in backend/src/api/routes/embeddings.py")

