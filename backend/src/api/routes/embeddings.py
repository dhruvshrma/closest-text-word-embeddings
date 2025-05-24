from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from loguru import logger


from core.embedding_manager import EmbeddingManager
from config import settings

router = APIRouter()


class TextPayload(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    text: str
    embedding: List[float]


class ModelInfo(BaseModel):
    model_name: str
    embedding_dim: int
    description: str


async def get_embedding_manager() -> EmbeddingManager:
    raise NotImplementedError(
        "This get_embedding_manager is a placeholder and should be overridden in main.py using app.dependency_overrides"
    )


@router.post(
    "/embed", response_model=EmbeddingResponse, summary="Get embedding for text"
)
async def get_embedding_api(
    payload: TextPayload, manager: EmbeddingManager = Depends(get_embedding_manager)
):
    try:
        logger.debug(
            f'Router: Received request to embed text: "{payload.text[:50]}..."'
        )
        embedding_array = manager.get_embedding(payload.text)
        logger.debug(
            f'Router: Successfully generated embedding for: "{payload.text[:50]}..."'
        )
        return EmbeddingResponse(text=payload.text, embedding=embedding_array.tolist())
    except Exception as e:
        logger.exception(
            f'Router: Error processing /embed request for text "{payload.text[:50]}...": {e}'
        )
        if isinstance(e, NotImplementedError) and "placeholder" in str(e):
            raise HTTPException(
                status_code=500,
                detail="Embedding service misconfiguration: Dependency not overridden.",
            )
        raise HTTPException(
            status_code=500, detail=f"Error generating embedding: {str(e)}"
        )


@router.get("/models", response_model=List[ModelInfo], summary="List available models")
async def list_models_api(manager: EmbeddingManager = Depends(get_embedding_manager)):
    try:
        logger.debug("Router: Received request to list models.")
        available_models = manager.get_available_models()

        response_models = []
        for model_data in available_models:
            embedding_dim = manager.embedding_dim
            assert (
                embedding_dim is not None
            ), "EmbeddingManager.embedding_dim should not be None after initialization."
            response_models.append(
                ModelInfo(
                    model_name=model_data.get("model_id", "Unknown Model"),
                    embedding_dim=embedding_dim,
                    description=model_data.get("description", "N/A"),
                )
            )
        return response_models
    except Exception as e:
        logger.exception(f"Router: Error processing /models request: {e}")
        if isinstance(e, NotImplementedError) and "placeholder" in str(e):
            raise HTTPException(
                status_code=500,
                detail="Embedding service misconfiguration: Dependency not overridden.",
            )
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@router.get(
    "/",
    summary="Get embedding router information",
    description="Basic info for the embeddings router.",
)
async def get_embeddings_router_info():
    logger.info("GET request to /embeddings/ router base")
    return {"message": "This is the embeddings API router. Use /embed or /models."}


logger.info("Embeddings API router configured in backend/src/api/routes/embeddings.py")
