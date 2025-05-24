from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys
from contextlib import asynccontextmanager
import os
from pathlib import Path

from config.config import settings
from api.routes import embeddings as embeddings_api_module
from api.routes import search as search_api_module
from api.routes import interpolation as interpolation_api_module
from core.embedding_manager import EmbeddingManager

logger.remove() 
logger.add(sys.stderr, level="INFO") 
logger.add("logs/app.log", rotation="10 MB", level="DEBUG")

# Define paths for prebuilt data - these should ideally match your script's output
# and could be moved to config.py later.
PREBUILT_MAX_WORDS = 600000 # As per your build_conceptnet_index.py script
PREBUILT_MODEL_NAME_PATH_PART = settings.default_model_name.replace("/", "_") # Use model from settings for consistency
PREBUILT_FAISS_METRIC = "cosine" # As per your build_conceptnet_index.py and common default

# Get the directory where main.py is located (should be backend/)
MAIN_PY_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory is backend/data/
PREBUILT_DATA_ROOT_DIR = os.path.join(MAIN_PY_DIR, "data")

PREBUILT_WORDS_FILENAME = f"conceptnet_{PREBUILT_MAX_WORDS}_words.pkl"
PREBUILT_EMBEDDINGS_FILENAME = f"conceptnet_{PREBUILT_MAX_WORDS}_embeddings_{PREBUILT_MODEL_NAME_PATH_PART}.npy"
PREBUILT_FAISS_INDEX_FILENAME = f"conceptnet_{PREBUILT_MAX_WORDS}_faiss_{PREBUILT_MODEL_NAME_PATH_PART}_{PREBUILT_FAISS_METRIC}.index"

# String paths from os.path
str_prebuilt_words_path = os.path.join(PREBUILT_DATA_ROOT_DIR, PREBUILT_WORDS_FILENAME)
str_prebuilt_embeddings_path = os.path.join(PREBUILT_DATA_ROOT_DIR, PREBUILT_EMBEDDINGS_FILENAME)
str_prebuilt_faiss_index_path = os.path.join(PREBUILT_DATA_ROOT_DIR, PREBUILT_FAISS_INDEX_FILENAME)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    # Initialize EmbeddingManager
    # It will use defaults from settings if not overridden here
    # Ensure model_name here matches the one used for pre-building if you want to use the prebuilt index directly.
    app.state.embedding_manager = EmbeddingManager(
        model_name=settings.default_model_name, 
        cache_dir=settings.default_embedding_cache_path,
        use_faiss_index=True, # Attempt to use FAISS
        faiss_metric=PREBUILT_FAISS_METRIC # Match the prebuilt index metric
    )
    logger.info(f"EmbeddingManager initialized with model: {app.state.embedding_manager.model_name_or_path}")

    # Attempt to load pre-built ConceptNet data
    logger.info(f"Attempting to load pre-built ConceptNet data from: {PREBUILT_DATA_ROOT_DIR}")
    logger.info(f" - Words file: {PREBUILT_WORDS_FILENAME} (exists: {os.path.exists(str_prebuilt_words_path)})")
    logger.info(f" - Embeddings file: {PREBUILT_EMBEDDINGS_FILENAME} (exists: {os.path.exists(str_prebuilt_embeddings_path)})")
    logger.info(f" - FAISS index file: {PREBUILT_FAISS_INDEX_FILENAME} (exists: {os.path.exists(str_prebuilt_faiss_index_path)})")

    try:
        if os.path.exists(str_prebuilt_words_path):
            # Convert string paths to Path objects for the call
            words_p = Path(str_prebuilt_words_path)
            embeddings_p = Path(str_prebuilt_embeddings_path) if os.path.exists(str_prebuilt_embeddings_path) else None
            index_p = Path(str_prebuilt_faiss_index_path) if os.path.exists(str_prebuilt_faiss_index_path) else None 
            
            app.state.embedding_manager.load_prebuilt_data(
                words_path=words_p,
                embeddings_path=embeddings_p,
                index_path=index_p
            )
            logger.info("Successfully attempted to load pre-built data.")
            status = app.state.embedding_manager.get_corpus_status()
            logger.info(f"Corpus status after loading: {status}")
        else:
            logger.warning(f"Pre-built words file not found at {str_prebuilt_words_path}. Skipping load of pre-built ConceptNet data. The application will run without a preloaded corpus or you can load one via API.")
    except Exception as e:
        logger.exception(f"Error loading pre-built ConceptNet data: {e}")
        logger.error("Application will continue without pre-loaded ConceptNet data. You can load a corpus via API.")

    yield
    logger.info("Application shutdown...")
    # Clean up resources if any (e.g., app.state.embedding_manager.cleanup() if you add such a method)

app = FastAPI(lifespan=lifespan)

# Global EmbeddingManager dependency
async def get_global_embedding_manager() -> EmbeddingManager:
    if not hasattr(app.state, "embedding_manager") or app.state.embedding_manager is None:
        # This case should ideally not be reached if lifespan initializes it correctly.
        logger.error("Global Embedding manager not initialized.")
        raise HTTPException(status_code=500, detail="Global Embedding manager not initialized.")
    return app.state.embedding_manager

# Corrected Dependency Overrides:
# Override the get_embedding_manager function *within each router module*
app.dependency_overrides[embeddings_api_module.get_embedding_manager] = get_global_embedding_manager
app.dependency_overrides[search_api_module.get_embedding_manager] = get_global_embedding_manager
app.dependency_overrides[interpolation_api_module.get_embedding_manager] = get_global_embedding_manager

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Corrected Router Inclusion:
# Include the .router attribute from each module
app.include_router(embeddings_api_module.router, prefix="/api/v1/embeddings", tags=["Embeddings"])
app.include_router(search_api_module.router, prefix="/api/v1/search", tags=["Search"])
app.include_router(interpolation_api_module.router, prefix="/api/v1/interpolation", tags=["Interpolation"])

@app.get("/")
async def root():
    return {"message": "Welcome to the LLM Embedding Explorer API"}

# (If you have other routers like analysis_router, they would be included here too)

logger.info("FastAPI app configured with EmbeddingManager and all routers.") 