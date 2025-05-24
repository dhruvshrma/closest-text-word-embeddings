import pytest
import numpy as np
from core.search_engines import FAISSEngine # Adjust import path as needed

@pytest.fixture
def sample_embeddings() -> np.ndarray:
    # Simple, distinct embeddings for predictable results
    return np.array([
        [0.1, 0.2, 0.7],  # Corresponds to "apple"
        [0.8, 0.1, 0.1],  # Corresponds to "banana"
        [0.3, 0.6, 0.1],  # Corresponds to "carrot"
        [0.2, 0.2, 0.6],  # A bit similar to apple
    ]).astype(np.float32)

@pytest.fixture
def sample_word_list() -> list:
    return ["apple", "banana", "carrot", "similar_to_apple"]

class TestFAISSEngine:
    def test_initialization(self):
        engine_l2 = FAISSEngine(dimension=3, metric='l2')
        assert engine_l2.index is not None
        assert engine_l2.index.d == 3

        engine_cosine = FAISSEngine(dimension=3, metric='cosine')
        assert engine_cosine.index is not None
        assert engine_cosine.index.d == 3

        with pytest.raises(ValueError):
            FAISSEngine(dimension=3, metric='unsupported_metric')

    def test_build_and_search_l2(self, sample_embeddings: np.ndarray):
        engine = FAISSEngine(dimension=3, metric='l2')
        engine.build_index(sample_embeddings)
        assert engine.index.ntotal == len(sample_embeddings)

        query_vector = np.array([[0.15, 0.25, 0.65]], dtype=np.float32) # Closest to apple and similar_to_apple
        distances, indices = engine.search(query_vector, k=2)
        
        assert indices.shape == (1, 2)
        assert distances.shape == (1, 2)
        
        # Expected indices (0: apple, 3: similar_to_apple)
        # Order might depend on exact L2 distances, so check presence
        assert 0 in indices[0]
        assert 3 in indices[0]
        # Distances should be positive for L2
        assert np.all(distances >= 0)

    def test_build_and_search_cosine(self, sample_embeddings: np.ndarray):
        engine = FAISSEngine(dimension=3, metric='cosine')
        engine.build_index(sample_embeddings)
        assert engine.index.ntotal == len(sample_embeddings)

        # Query vector similar to "apple"
        query_vector = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
        distances, indices = engine.search(query_vector, k=1)
        
        assert indices.shape == (1, 1)
        assert distances.shape == (1, 1)
        assert indices[0][0] == 0 # Index of "apple"
        # For cosine similarity (inner product with normalized vectors), distance should be close to 1.0 for identical
        assert distances[0][0] > 0.9 # Expect high similarity for self-query essentially

        # Query vector for "banana-like"
        query_vector_banana = np.array([[0.7, 0.2, 0.2]], dtype=np.float32)
        distances_b, indices_b = engine.search(query_vector_banana, k=1)
        assert indices_b[0][0] == 1 # Index of "banana"
        assert distances_b[0][0] > 0.8 # Reasonably high similarity

    def test_search_empty_index(self):
        engine = FAISSEngine(dimension=3, metric='l2')
        query_vector = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        with pytest.raises(RuntimeError, match="Index is not built or is empty"): # Check error message
            engine.search(query_vector, k=1)

    def test_build_with_wrong_dimension(self, sample_embeddings: np.ndarray):
        engine = FAISSEngine(dimension=2, metric='l2') # Expects 2D, gets 3D
        with pytest.raises(ValueError, match="Embeddings must be a 2D array with dimension 2"):
            engine.build_index(sample_embeddings)

    def test_search_with_wrong_dimension(self, sample_embeddings: np.ndarray):
        engine = FAISSEngine(dimension=3, metric='l2')
        engine.build_index(sample_embeddings)
        query_wrong_dim = np.array([[0.1, 0.2]], dtype=np.float32) # Query is 2D
        with pytest.raises(ValueError, match="Query embeddings must be a 2D array with dimension 3"):
            engine.search(query_wrong_dim, k=1)

    def test_search_single_vs_batch_query(self, sample_embeddings: np.ndarray):
        engine = FAISSEngine(dimension=3, metric='cosine')
        engine.build_index(sample_embeddings)

        query_single = np.array([0.1, 0.2, 0.7], dtype=np.float32) # "apple"
        query_batch = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]], dtype=np.float32) # "apple", "banana"

        distances_s, indices_s = engine.search(query_single, k=1)
        distances_b, indices_b = engine.search(query_batch, k=1)

        assert indices_s.shape == (1,1)
        assert indices_b.shape == (2,1)
        assert indices_s[0][0] == indices_b[0][0] # First result should be the same
        assert indices_b[0][0] == 0 # apple
        assert indices_b[1][0] == 1 # banana 