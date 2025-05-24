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

# Removed: embedding_manager_instance = EmbeddingManager(...)


async def get_embedding_manager() -> EmbeddingManager:
    """Placeholder for dependency injection. This should be overridden by the main app."""
    # This function body will effectively be replaced by the one in main.py via dependency_overrides
    # If not overridden, calls to this will fail, indicating a setup problem.
    raise NotImplementedError(
        "Search router's get_embedding_manager was not overridden by the main application. Check FastAPI startup and dependency overrides."
    )


@router.post(
    "/load-corpus",
    response_model=LoadCorpusResponse,
    summary="Load a dataset into the search engine",
)
async def load_corpus_route(
    request: LoadCorpusRequest, em: EmbeddingManager = Depends(get_embedding_manager)
):
    """Loads a specified dataset into the EmbeddingManager and builds the FAISS index."""
    dataset: Optional[Dataset] = None
    dataset_words: List[str] = []

    try:
        if request.dataset_name == "nltk_common_words":
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
            dataset = ConceptNetWordsDataset(
                max_words=max_w, min_word_length=min_len, stream_limit=200000
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown dataset_name: {request.dataset_name}"
            )

        if dataset:
            print(f"Loading dataset: {request.dataset_name}")
            dataset_words = dataset.load()
            print(
                f"Dataset {request.dataset_name} loaded with {len(dataset_words)} words."
            )
        else:
            raise HTTPException(
                status_code=500, detail="Dataset could not be initialized."
            )

        if not dataset_words:
            msg = f"Dataset '{request.dataset_name}' loaded but resulted in an empty word list."
            print(msg)
            raise HTTPException(status_code=404, detail=msg)

        # Use the injected EmbeddingManager (em)
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
        print(f"Error loading corpus {request.dataset_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading corpus {request.dataset_name}: {str(e)}",
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search for similar texts using the loaded corpus",
)
async def search_route(
    request: SearchQuery,
    em: EmbeddingManager = Depends(
        get_embedding_manager
    ),  # Uses the overridden manager
):
    """Performs a nearest neighbor search for a single query text."""
    # The injected 'em' is now the global manager from app.state
    if not em.corpus_texts:
        raise HTTPException(
            status_code=404,
            detail="Corpus not loaded. Please load a corpus first via /load-corpus or ensure pre-built data loaded on startup.",
        )

    if (
        not em._faiss_actually_enabled
        or not em.faiss_engine
        or not em.faiss_engine.index
        or em.faiss_engine.index.ntotal == 0
    ):
        # Even if corpus_texts is loaded, FAISS might not be ready for search
        raise HTTPException(
            status_code=503,
            detail="FAISS search is not available, not initialized, or index is empty. Ensure corpus is loaded and FAISS is enabled/working.",
        )

    results = em.search_corpus(query_texts=[request.text], k=request.k)
    # search_corpus now returns List[List[Dict]] so results[0] is the list of dicts for the single query
    found_results = (
        [SearchResultItem(**item) for item in results[0]]
        if results and results[0]
        else []
    )

    return SearchResponse(query_text=request.text, results=found_results)


@router.post(
    "/batch-search",
    response_model=BatchSearchResponse,
    summary="Batch search for similar texts",
)
async def batch_search_route(
    request: BatchSearchQuery,
    em: EmbeddingManager = Depends(
        get_embedding_manager
    ),  # Uses the overridden manager
):
    """Performs a nearest neighbor search for a batch of query texts."""
    if not em.corpus_texts:
        raise HTTPException(
            status_code=404,
            detail="Corpus not loaded. Please load a corpus first via /load-corpus or ensure pre-built data loaded on startup.",
        )

    if (
        not em._faiss_actually_enabled
        or not em.faiss_engine
        or not em.faiss_engine.index
        or em.faiss_engine.index.ntotal == 0
    ):
        raise HTTPException(
            status_code=503,
            detail="FAISS search is not available, not initialized, or index is empty. Ensure corpus is loaded and FAISS is enabled/working.",
        )

    search_results_raw = em.search_corpus(query_texts=request.texts, k=request.k)

    response_items = []
    for i, query_text in enumerate(request.texts):
        # search_results_raw[i] contains the list of dicts for the i-th query text
        matches = (
            [SearchResultItem(**item) for item in search_results_raw[i]]
            if search_results_raw
            and len(search_results_raw) > i
            and search_results_raw[i]
            else []
        )
        response_items.append(
            BatchSearchResultItem(query_text=query_text, matches=matches)
        )

    return BatchSearchResponse(all_results=response_items)


# TODO: Add this router to the main FastAPI application in main.py
# app.include_router(search_router, prefix="/api/v1/search", tags=["Search"])
