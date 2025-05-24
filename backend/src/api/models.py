from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal


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
    text: str = Field(..., description="The query text to search for.")
    k: int = Field(5, gt=0, description="Number of nearest neighbors to return.")


class SearchResultItem(BaseModel):
    text: str
    score: float
    id: Optional[int] = None  # Corpus index ID


class SearchResponse(BaseModel):
    query_text: str
    results: List[SearchResultItem]
    # Could add query_embedding if needed for debugging or further use


class BatchSearchQuery(BaseModel):
    texts: List[str] = Field(..., description="A list of query texts to search for.")
    k: int = Field(
        5, gt=0, description="Number of nearest neighbors to return for each query."
    )


class BatchSearchResultItem(BaseModel):
    query_text: str
    matches: List[SearchResultItem]


class BatchSearchResponse(BaseModel):
    all_results: List[BatchSearchResultItem]


# --- Corpus Loading ---
class LoadCorpusRequest(BaseModel):
    dataset_name: str = Field(
        ...,
        description="Name of the dataset to load (e.g., 'nltk_common_words', 'conceptnet_words')",
    )
    max_words: Optional[int] = Field(
        None, description="Maximum number of words to load from the dataset"
    )
    min_word_length: Optional[int] = Field(
        None, description="Minimum length for words to be included"
    )
    rebuild_index: bool = Field(
        False, description="Force rebuilding the FAISS index even if one exists"
    )
    # Add other dataset-specific params as needed


class LoadCorpusResponse(BaseModel):
    message: str
    num_documents_loaded: int
    index_status: str  # e.g., "Built", "Already Existed", "Not Built (FAISS disabled)"


# --- New Models for Interpolation API ---


class InterpolationRequest(BaseModel):
    start_word: str = Field(..., description="The starting word for interpolation.")
    end_word: str = Field(..., description="The ending word for interpolation.")
    steps: int = Field(
        50, gt=1, description="Number of interpolation steps (must be > 1)."
    )
    method: Literal["linear", "slerp", "bezier"] = Field(
        "slerp", description="Interpolation method."
    )
    control_words: Optional[List[str]] = Field(
        None, description="List of control words for Bezier curve interpolation."
    )
    k_neighbors: int = Field(
        5, gt=0, description="Number of nearest neighbors to find for each path point."
    )


class PathPoint(BaseModel):
    step: int = Field(..., description="Index of this point in the path sequence.")
    embedding: List[float] = Field(
        ..., description="The embedding vector at this point."
    )
    nearest_words: List[str] = Field(
        ..., description="List of nearest words to this point."
    )
    nearest_distances: List[float] = Field(
        ..., description="List of distances to the nearest words."
    )
    t_value: float = Field(
        ..., description="Interpolation parameter t (typically 0 to 1) for this point."
    )


class InterpolationResponse(BaseModel):
    path: List[PathPoint] = Field(
        ..., description="List of points forming the interpolation path."
    )
    start_word_embedding: Optional[List[float]] = Field(
        None, description="Embedding of the start word."
    )
    end_word_embedding: Optional[List[float]] = Field(
        None, description="Embedding of the end word."
    )
    control_word_embeddings: Optional[List[List[float]]] = Field(
        None, description="Embeddings of the control words, if used."
    )
    method_used: str = Field(..., description="The interpolation method that was used.")
    # total_distance: float # This might be ambiguous (Euclidean in original space, or along path?)
    # For now, let frontend calculate if needed from start/end embeddings.
