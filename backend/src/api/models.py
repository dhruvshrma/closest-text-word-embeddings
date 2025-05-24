from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class TextListInput(BaseModel):
    texts: List[str]
    use_cache: bool = True
    batch_size: int = 32


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model_info: Dict[str, str]


class ModelInfo(BaseModel):
    model_id: str
    description: str


class AvailableModelsResponse(BaseModel):
    models: List[ModelInfo]


# --- Search Related Models ---


class SearchQuery(BaseModel):
    text: str  # A single query text for simplicity in this example, can be extended to List[str]
    k: int = 5


class SearchResultItem(BaseModel):
    text: str
    index: int
    distance: float


class SearchResponse(BaseModel):
    query_text: str
    results: List[SearchResultItem]
    # Could add query_embedding if needed for debugging or further use


class BatchSearchQuery(BaseModel):
    texts: List[str]
    k: int = 5


class BatchSearchResultItem(BaseModel):
    query_text: str
    matches: List[SearchResultItem]


class BatchSearchResponse(BaseModel):
    all_results: List[BatchSearchResultItem]


# --- Corpus Loading ---
class LoadCorpusRequest(BaseModel):
    dataset_name: str  # e.g., "nltk_common_words", "conceptnet_words"
    max_words: Optional[int] = None  # For datasets that support it
    min_word_length: Optional[int] = None  # For datasets that support it
    rebuild_index: bool = False
    # Add other dataset-specific params as needed


class LoadCorpusResponse(BaseModel):
    message: str
    num_documents_loaded: int
    index_status: str  # e.g., "Built", "Already Existed", "Not Built (FAISS disabled)"
