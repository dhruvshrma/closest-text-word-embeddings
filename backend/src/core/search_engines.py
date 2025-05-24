from typing import Protocol, Tuple, runtime_checkable
import numpy as np
import faiss

# import hnswlib # HNSW is an alternative, not implementing yet


@runtime_checkable
class SearchEngine(Protocol):
    """Protocol for nearest neighbor search engines"""

    def build_index(self, embeddings: np.ndarray) -> None:
        """Build search index from embeddings"""
        ...

    def search(
        self, query_embeddings: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors for a batch of query embeddings.

        Args:
            query_embeddings: A 2D numpy array of query embeddings.
            k: The number of nearest neighbors to find.

        Returns:
            A tuple containing:
                - distances: A 2D numpy array of distances to the nearest neighbors.
                - indices: A 2D numpy array of indices of the nearest neighbors.
        """
        ...


class FAISSEngine:
    """FAISS implementation of SearchEngine protocol"""

    def __init__(self, dimension: int, metric: str = "cosine"):
        self.metric = metric.lower()
        self.dimension = dimension
        self.index = None

        if self.metric not in ["cosine", "l2"]:
            raise ValueError(f"Unsupported metric: {metric}. Choose 'cosine' or 'l2'.")

        if self.metric == "cosine":
            # For cosine similarity, FAISS uses IndexFlatIP on normalized vectors
            self.index = faiss.IndexFlatIP(self.dimension)
        else:  # L2 (Euclidean)
            self.index = faiss.IndexFlatL2(self.dimension)

    def build_index(self, embeddings: np.ndarray) -> None:
        """
        Builds the FAISS index from the provided embeddings.

        Args:
            embeddings: A 2D numpy array of embeddings to be indexed.
                        Embeddings should be C-contiguous and of type np.float32.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embeddings must be a 2D array with dimension {self.dimension}"
            )

        embeddings_to_add = embeddings.astype(np.float32)
        if not embeddings_to_add.flags["C_CONTIGUOUS"]:
            embeddings_to_add = np.ascontiguousarray(embeddings_to_add)

        if self.metric == "cosine":
            # Normalize embeddings for cosine similarity if using IndexFlatIP
            faiss.normalize_L2(embeddings_to_add)

        if self.index is None:  # Should have been initialized in __init__
            raise RuntimeError("FAISS index was not initialized.")

        self.index.add(embeddings_to_add)
        print(f"FAISS index built successfully with {self.index.ntotal} vectors.")

    def search(
        self, query_embeddings: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the index for the k nearest neighbors of the query embeddings.

        Args:
            query_embeddings: A 2D numpy array of query embeddings.
                              Should be C-contiguous and of type np.float32.
            k: The number of nearest neighbors to retrieve.

        Returns:
            A tuple (distances, indices) where:
                - distances: A 2D numpy array of shape (num_queries, k) containing
                             the distances to the k nearest neighbors.
                - indices: A 2D numpy array of shape (num_queries, k) containing
                           the indices of the k nearest neighbors.
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError(
                "Index is not built or is empty. Call build_index first."
            )

        if query_embeddings.ndim == 1:  # Single query vector
            query_embeddings = np.expand_dims(query_embeddings, axis=0)

        if query_embeddings.ndim != 2 or query_embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Query embeddings must be a 2D array with dimension {self.dimension}"
            )

        queries_to_search = query_embeddings.astype(np.float32)
        if not queries_to_search.flags["C_CONTIGUOUS"]:
            queries_to_search = np.ascontiguousarray(queries_to_search)

        if self.metric == "cosine":
            # Normalize query vectors for cosine similarity if using IndexFlatIP
            faiss.normalize_L2(queries_to_search)

        distances, indices = self.index.search(queries_to_search, k)

        # For IndexFlatIP (cosine similarity), distances are inner products.
        # To convert to cosine distance: dist = 1 - inner_product.
        # However, FAISS's IndexFlatIP returns dot products, higher is better.
        # If you need actual cosine distances (smaller is better), you might need to adjust.
        # For now, returning raw IndexFlatIP scores or L2 distances.
        return distances, indices
