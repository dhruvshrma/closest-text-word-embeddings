from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
import numpy as np
from loguru import logger
from api.models import InterpolationRequest, InterpolationResponse, PathPoint
from core.embedding_manager import EmbeddingManager
from core.interpolation import InterpolationEngine

router = APIRouter()


# Dependency function for this router (will be overridden by main app)
async def get_embedding_manager() -> EmbeddingManager:
    raise NotImplementedError(
        "Interpolation router's get_embedding_manager was not overridden. Check FastAPI startup."
    )


@router.post(
    "/path",
    response_model=InterpolationResponse,
    summary="Generate an interpolation path between two words",
)
async def create_interpolation_path(
    request: InterpolationRequest, em: EmbeddingManager = Depends(get_embedding_manager)
):
    logger.info(f"Received interpolation request: {request.model_dump_json(indent=2)}")

    if (
        em.corpus_texts is None
        or not em._faiss_actually_enabled
        or not em.faiss_engine
        or not em.faiss_engine.index
    ):
        detailed_error = "Corpus not loaded, FAISS not enabled, FAISS engine not initialized, FAISS index not built, or FAISS index is empty."
        if em.corpus_texts is None: detailed_error = "Corpus texts are None. " + detailed_error
        if not em._faiss_actually_enabled: detailed_error = "FAISS not actually enabled. " + detailed_error
        if not em.faiss_engine: detailed_error = "FAISS engine is None. " + detailed_error
        if hasattr(em, 'faiss_engine') and em.faiss_engine and not em.faiss_engine.index: detailed_error = "FAISS index is None. " + detailed_error
        if hasattr(em, 'faiss_engine') and em.faiss_engine and em.faiss_engine.index and em.faiss_engine.index.ntotal == 0: detailed_error = "FAISS index is empty. " + detailed_error 
        logger.error(f"Cannot create interpolation path: {detailed_error}")
        raise HTTPException(status_code=503,detail=f"Interpolation prerequisite failed: {detailed_error}")

    try:
        logger.debug("Step 1: Getting embeddings for start and end words...")
        start_emb_list = em.get_embeddings([request.start_word])
        end_emb_list = em.get_embeddings([request.end_word])

        if start_emb_list.size == 0 or end_emb_list.size == 0:
            logger.error(
                f"Could not get embeddings for start/end words: '{request.start_word}', '{request.end_word}'"
            )
            raise HTTPException(
                status_code=404,
                detail="Could not retrieve embeddings for start or end word.",
            )
        
        start_emb = start_emb_list[0]
        end_emb = end_emb_list[0]
        logger.debug(f"Start_emb shape: {start_emb.shape}, dtype: {start_emb.dtype}. Has NaN/Inf: {np.isnan(start_emb).any() or np.isinf(start_emb).any()}")
        logger.debug(f"End_emb shape: {end_emb.shape}, dtype: {end_emb.dtype}. Has NaN/Inf: {np.isnan(end_emb).any() or np.isinf(end_emb).any()}")
        # logger.debug(f"Start_emb values (first 5): {start_emb[:5]}") # Optional: log some values
        # logger.debug(f"End_emb values (first 5): {end_emb[:5]}")

        control_embeddings_list: Optional[List[np.ndarray]] = None
        actual_control_word_embeddings_for_response: Optional[List[List[float]]] = None
        if request.method == "bezier" and request.control_words:
            logger.debug("Getting control word embeddings...")
            if not request.control_words:
                logger.warning("Bezier method chosen but no control words provided.")
            else:
                raw_control_embs_array = em.get_embeddings(request.control_words)
                if raw_control_embs_array.shape[0] != len(request.control_words):
                    logger.error("Could not get embeddings for all control words.")
                    raise HTTPException(
                        status_code=404,
                        detail="Could not retrieve embeddings for all control words.",
                    )
                control_embeddings_list = [
                    raw_control_embs_array[i]
                    for i in range(raw_control_embs_array.shape[0])
                ]
                actual_control_word_embeddings_for_response = (
                    raw_control_embs_array.tolist()
                )
                logger.debug(f"Control embeddings ({len(control_embeddings_list)}) obtained.")

        logger.debug(f"Step 2: Generating interpolation path using method: {request.method}...")
        interpolation_engine = InterpolationEngine()
        path_embeddings: np.ndarray

        if request.method == "linear":
            path_embeddings = interpolation_engine.linear_interpolation(
                start_emb, end_emb, request.steps
            )
        elif request.method == "slerp":
            path_embeddings = interpolation_engine.slerp(
                start_emb, end_emb, request.steps
            )
        elif request.method == "bezier":
            logger.error("Bezier interpolation method is not yet fully implemented.")
            raise HTTPException(
                status_code=501, detail="Bezier interpolation is not yet implemented."
            )
        else:
            logger.error(f"Unknown interpolation method: {request.method}")
            raise HTTPException(
                status_code=400,
                detail=f"Unknown interpolation method: {request.method}",
            )

        if path_embeddings.size == 0:
            logger.error("Interpolation resulted in an empty path.")
            raise HTTPException(
                status_code=500, detail="Interpolation resulted in an empty path."
            )
        
        logger.debug(f"Path_embeddings shape: {path_embeddings.shape}, dtype: {path_embeddings.dtype}. Has NaN/Inf: {np.isnan(path_embeddings).any() or np.isinf(path_embeddings).any()}")
        # logger.debug(f"Path_embeddings (first point, first 5 vals): {path_embeddings[0, :5] if path_embeddings.ndim == 2 and path_embeddings.shape[0] > 0 else 'N/A'}")

        path_points_response: List[PathPoint] = []
        all_neighbors_results: List[List[dict]] = [[] for _ in range(path_embeddings.shape[0])] # Initialize

        if request.k_neighbors > 0 and path_embeddings.size > 0:
            logger.debug(f"Step 3: Searching for {request.k_neighbors} neighbors for {path_embeddings.shape[0]} path points...")
            try:
                # Check for non-finite values before sending to FAISS
                if np.isnan(path_embeddings).any() or np.isinf(path_embeddings).any():
                    logger.error("Path embeddings contain NaN or Inf values *before* FAISS search. Aborting search.")
                    # Depending on desired behavior, you might raise or just have empty neighbors
                    # For now, we proceed but neighbors will be empty due to pre-initialization of all_neighbors_results
                    # Or, more strictly:
                    raise ValueError("Path embeddings contain non-finite values, cannot perform FAISS search.") 
                
                all_neighbors_results = em.search_by_embeddings(
                    path_embeddings, k=request.k_neighbors
                )
                logger.debug(f"Nearest neighbor search completed. Found results for {len(all_neighbors_results)} points.")
            except ValueError as ve_search: # Catch the ValueError from pre-check
                logger.error(f"ValueError before FAISS search: {ve_search}")
                # all_neighbors_results is already initialized to empty lists
            except Exception as search_exc:
                logger.exception(
                    f"Error during batch nearest neighbor search for path points: {search_exc}"
                )
                # all_neighbors_results is already initialized to empty lists, so path points will have no neighbors

        logger.debug("Step 4: Assembling path points response...")
        for i, p_emb in enumerate(path_embeddings):
            t_value = i / (request.steps - 1) if request.steps > 1 else 0

            nearest_words_for_point: List[str] = []
            nearest_distances_for_point: List[float] = []

            if request.k_neighbors > 0 and i < len(all_neighbors_results):
                # all_neighbors_results is List[List[Dict[str, Any]]]
                # Each inner list corresponds to a path point's neighbors
                for neighbor_info in all_neighbors_results[i]:
                    nearest_words_for_point.append(neighbor_info.get("text", "Unknown"))
                    nearest_distances_for_point.append(
                        float(neighbor_info.get("score", 0.0))
                    )

            path_points_response.append(
                PathPoint(
                    step=i,
                    embedding=p_emb.tolist(),
                    nearest_words=nearest_words_for_point,
                    nearest_distances=nearest_distances_for_point,
                    t_value=t_value,
                )
            )

        logger.info("Successfully created interpolation path response.")
        return InterpolationResponse(
            path=path_points_response,
            start_word_embedding=start_emb.tolist(),
            end_word_embedding=end_emb.tolist(),
            control_word_embeddings=actual_control_word_embeddings_for_response,
            method_used=request.method,
        )

    except HTTPException:  # Re-raise HTTPExceptions directly
        raise
    except ValueError as ve:
        logger.error(f"ValueError during interpolation processing: {ve}")
        # This will catch the ValueError if raised from the NaN/Inf check before FAISS
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Unexpected error creating interpolation path: {e}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


logger.info(
    "Interpolation API router configured in backend/src/api/routes/interpolation.py"
)
