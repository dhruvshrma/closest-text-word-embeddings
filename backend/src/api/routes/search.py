from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional

from api.models import (
    SearchQuery,
    SearchResponse,
    SearchResultItem,
    BatchSearchQuery,
    BatchSearchResponse,
    BatchSearchResultItem,
    LoadCorpusRequest,
    LoadCorpusResponse,
)

from core.embedding_manager import EmbeddingManager
from data.dataset import (
    NltkCommonWordsDataset,
    ConceptNetWordsDataset,
    Dataset,
)  # Import your dataset loaders

router = APIRouter()

# TODO: Refactor to use FastAPI's Depends for proper state management and testing.
embedding_manager_instance = EmbeddingManager(
    model_name="all-MiniLM-L6-v2", use_faiss_index=True
)


async def get_embedding_manager() -> EmbeddingManager:
    if embedding_manager_instance is None:
        raise HTTPException(status_code=503, detail="EmbeddingManager not initialized")
    return embedding_manager_instance


@router.post("/load-corpus", response_model=LoadCorpusResponse)
async def load_corpus_route(
    request: LoadCorpusRequest, em: EmbeddingManager = Depends(get_embedding_manager)
):
    """Loads a specified dataset into the EmbeddingManager and builds the FAISS index."""
    dataset: Optional[Dataset] = None
    dataset_words: List[str] = []

    try:
        if request.dataset_name == "nltk_common_words":
            # Parameters from request or defaults
            top_k = request.max_words if request.max_words is not None else 10000
            min_len = (
                request.min_word_length if request.min_word_length is not None else 3
            )
            dataset = NltkCommonWordsDataset(top_k=top_k, min_length=min_len)
        elif request.dataset_name == "conceptnet_words":
            max_w = request.max_words if request.max_words is not None else 10000
            min_len = (
                request.min_word_length if request.min_word_length is not None else 3
            )
            # stream_limit for ConceptNet can be added to LoadCorpusRequest if needed
            dataset = ConceptNetWordsDataset(
                max_words=max_w, min_word_length=min_len, stream_limit=200000
            )  # Default stream_limit
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown dataset_name: {request.dataset_name}"
            )

        if dataset:
            print(f"Loading dataset: {request.dataset_name}")
            dataset_words = dataset.load()  # This can take time
            print(
                f"Dataset {request.dataset_name} loaded with {len(dataset_words)} words."
            )
        else:
            # Should have been caught by unknown dataset_name, but as a safeguard
            raise HTTPException(
                status_code=500, detail="Dataset could not be initialized."
            )

        if not dataset_words:
            # If dataset loaded but returned no words (e.g. NLTK fallback or ConceptNet empty)
            msg = f"Dataset '{request.dataset_name}' loaded but resulted in an empty word list."
            print(msg)
            # Decide if this is an error or just a state to report
            # For now, let load_corpus handle it, it might build an empty index or warn.
            raise HTTPException(status_code=404, detail=msg)

        em.load_corpus(texts=dataset_words, rebuild_index=request.rebuild_index)

        num_loaded = len(dataset_words)
        idx_status = "Not applicable (FAISS disabled or not used)"
        if em._faiss_actually_enabled and em.faiss_engine and em.faiss_engine.index:
            idx_status = f"Built/Updated with {em.faiss_engine.index.ntotal} vectors."
        elif em.initial_use_faiss_request and not em._faiss_actually_enabled:
            idx_status = "Requested but not built (FAISS unavailable/error)."

        return LoadCorpusResponse(
            message=f"Corpus '{request.dataset_name}' loaded successfully.",
            num_documents_loaded=num_loaded,
            index_status=idx_status,
        )
    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail=f"Dataset dependency missing for {request.dataset_name}: {e}",
        )
    except Exception as e:
        # Log the exception e for debugging
        print(f"Error loading corpus {request.dataset_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading corpus {request.dataset_name}: {str(e)}",
        )


@router.post("/search", response_model=SearchResponse)
async def search_route(
    request: SearchQuery, em: EmbeddingManager = Depends(get_embedding_manager)
):
    """Performs a nearest neighbor search for a single query text."""
    if not em._faiss_actually_enabled or not em.faiss_engine:
        raise HTTPException(
            status_code=503, detail="FAISS search is not available or not initialized."
        )
    if em.corpus_texts is None or (
        em.faiss_engine.index is None or em.faiss_engine.index.ntotal == 0
    ):
        raise HTTPException(
            status_code=404,
            detail="Corpus not loaded or FAISS index empty. Please load a corpus first.",
        )

    results = em.search_corpus(query_texts=[request.text], k=request.k)
    if results is None or not results[0]:
        found_results = []  # No results found
    else:
        found_results = [SearchResultItem(**item) for item in results[0]]

    return SearchResponse(query_text=request.text, results=found_results)


@router.post("/batch-search", response_model=BatchSearchResponse)
async def batch_search_route(
    request: BatchSearchQuery, em: EmbeddingManager = Depends(get_embedding_manager)
):
    """Performs a nearest neighbor search for a batch of query texts."""
    if not em._faiss_actually_enabled or not em.faiss_engine:
        raise HTTPException(
            status_code=503, detail="FAISS search is not available or not initialized."
        )
    if em.corpus_texts is None or (
        em.faiss_engine.index is None or em.faiss_engine.index.ntotal == 0
    ):
        raise HTTPException(
            status_code=404,
            detail="Corpus not loaded or FAISS index empty. Please load a corpus first.",
        )

    search_results_raw = em.search_corpus(query_texts=request.texts, k=request.k)

    if search_results_raw is None:
        # This case implies an issue with search_corpus itself (e.g., FAISS disabled mid-way, though unlikely with checks)
        raise HTTPException(
            status_code=500, detail="Batch search failed to produce results."
        )

    response_items = []
    for i, query_text in enumerate(request.texts):
        matches = [SearchResultItem(**item) for item in search_results_raw[i]]
        response_items.append(
            BatchSearchResultItem(query_text=query_text, matches=matches)
        )

    return BatchSearchResponse(all_results=response_items)


# TODO: Add this router to the main FastAPI application in main.py
# app.include_router(search_router, prefix="/api/v1/search", tags=["Search"])
