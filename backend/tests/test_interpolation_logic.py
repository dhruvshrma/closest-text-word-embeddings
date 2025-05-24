import pytest
import numpy as np
import os
from pathlib import Path

# Adjust import paths based on your project structure and how tests are run.
# Assuming 'backend.src' is in PYTHONPATH or the project is installed.
from core.embedding_manager import EmbeddingManager
from core.interpolation import InterpolationEngine
from config.config import settings # For default model name, cache path

# --- Test Configuration ---
PREBUILT_MAX_WORDS = 600000 
PREBUILT_MODEL_NAME = settings.default_model_name 
PREBUILT_MODEL_NAME_PATH_PART = PREBUILT_MODEL_NAME.replace("/", "_")
PREBUILT_FAISS_METRIC = "cosine"

TEST_FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_FILE_DIR.parent 
BACKEND_DIR = PROJECT_ROOT
PREBUILT_DATA_ROOT_DIR = BACKEND_DIR / "data"

WORDS_PATH_STR = str(PREBUILT_DATA_ROOT_DIR / f"conceptnet_{PREBUILT_MAX_WORDS}_words.pkl")
EMBEDDINGS_PATH_STR = str(PREBUILT_DATA_ROOT_DIR / f"conceptnet_{PREBUILT_MAX_WORDS}_embeddings_{PREBUILT_MODEL_NAME_PATH_PART}.npy")
FAISS_INDEX_PATH_STR = str(PREBUILT_DATA_ROOT_DIR / f"conceptnet_{PREBUILT_MAX_WORDS}_faiss_{PREBUILT_MODEL_NAME_PATH_PART}_{PREBUILT_FAISS_METRIC}.index")

prebuilt_files_exist = (
    os.path.exists(WORDS_PATH_STR) and 
    os.path.exists(EMBEDDINGS_PATH_STR) and 
    os.path.exists(FAISS_INDEX_PATH_STR)
)

@pytest.mark.skipif(not prebuilt_files_exist, reason="Prebuilt ConceptNet data files not found in backend/data/. Run build_conceptnet_index.py first.")
def test_interpolation_path_logic():
    """Tests the core logic of generating an interpolation path and finding neighbors."""
    print(f"\n--- Running test_interpolation_path_logic ---", flush=True)
    # ... (existing path printouts) ...
    print(f"PROJECT_ROOT determined as: {PROJECT_ROOT}")
    print(f"PREBUILT_DATA_ROOT_DIR: {PREBUILT_DATA_ROOT_DIR}")
    print(f"Words file: {WORDS_PATH_STR} (Exists: {os.path.exists(WORDS_PATH_STR)})")
    print(f"Embeddings file: {EMBEDDINGS_PATH_STR} (Exists: {os.path.exists(EMBEDDINGS_PATH_STR)})")
    print(f"Index file: {FAISS_INDEX_PATH_STR} (Exists: {os.path.exists(FAISS_INDEX_PATH_STR)})")

    print("Initializing EmbeddingManager...")
    em = EmbeddingManager(
        model_name=PREBUILT_MODEL_NAME,
        cache_dir=settings.default_embedding_cache_path, 
        use_faiss_index=True,
        faiss_metric=PREBUILT_FAISS_METRIC
    )
    print(f"EmbeddingManager initialized. Model: {em.model_name_or_path}, FAISS enabled: {em._faiss_actually_enabled}")

    print("Loading pre-built data into EmbeddingManager...")
    em.load_prebuilt_data(
        words_path=Path(WORDS_PATH_STR),
        embeddings_path=Path(EMBEDDINGS_PATH_STR) if os.path.exists(EMBEDDINGS_PATH_STR) else None,
        index_path=Path(FAISS_INDEX_PATH_STR) if os.path.exists(FAISS_INDEX_PATH_STR) else None
    )
    corpus_status = em.get_corpus_status()
    print(f"Corpus status after loading: {corpus_status}")
    assert corpus_status["num_texts"] > 0, "Corpus texts were not loaded."
    assert corpus_status["embeddings_loaded"], "Corpus embeddings were not loaded."
    assert em.corpus_embeddings is not None, "Corpus embeddings array is None after load."
    assert corpus_status["faiss_index_active"], "FAISS index is not active."
    assert corpus_status["faiss_index_size"] > 0, "FAISS index is empty after loading prebuilt data."

    # *** Pre-check: Test FAISS search with an original corpus embedding ***
    print("\n--- FAISS Pre-check ---", flush=True)
    if em.corpus_embeddings is not None and em.corpus_embeddings.shape[0] > 0:
        sample_corpus_embedding = np.expand_dims(em.corpus_embeddings[0], axis=0) # Take the first embedding
        print(f"Performing test search with first corpus embedding. Shape: {sample_corpus_embedding.shape}, Dtype: {sample_corpus_embedding.dtype}", flush=True)
        try:
            pre_check_distances, pre_check_indices = em.faiss_engine.search(sample_corpus_embedding, k=2) # type: ignore
            print(f"FAISS Pre-check search successful. Indices: {pre_check_indices}, Distances: {pre_check_distances}", flush=True)
            assert pre_check_indices[0][0] == 0, "FAISS Pre-check: First result should be the item itself (index 0)"
        except Exception as e_precheck:
            print(f"FAISS Pre-check search FAILED: {e_precheck}", flush=True)
            pytest.fail(f"FAISS Pre-check search failed: {e_precheck}")
    else:
        print("Skipping FAISS pre-check as corpus_embeddings are not available after load.", flush=True)
        pytest.skip("Skipping FAISS pre-check as corpus_embeddings are not available after load.")
    print("--- End FAISS Pre-check ---\n", flush=True)

    start_word = "king"
    end_word = "queen"
    steps = 5
    method = "linear"
    k_neighbors = 2

    print(f"Test parameters: start='{start_word}', end='{end_word}', steps={steps}, method='{method}', k={k_neighbors}")

    print("Getting start/end word embeddings...")
    start_emb_list = em.get_embeddings([start_word])
    end_emb_list = em.get_embeddings([end_word])
    assert start_emb_list.size > 0, f"Could not get embedding for start word: {start_word}"
    assert end_emb_list.size > 0, f"Could not get embedding for end word: {end_word}"
    start_emb = start_emb_list[0]
    end_emb = end_emb_list[0]
    print(f"Start_emb shape: {start_emb.shape}, End_emb shape: {end_emb.shape}")
    print(f"Start_emb dtype: {start_emb.dtype}, flags: {start_emb.flags}")
    print(f"End_emb dtype: {end_emb.dtype}, flags: {end_emb.flags}")
    assert not (np.isnan(start_emb).any() or np.isinf(start_emb).any()), "Start embedding contains NaN/Inf"
    assert not (np.isnan(end_emb).any() or np.isinf(end_emb).any()), "End embedding contains NaN/Inf"

    print("Generating interpolation path...")
    interpolation_engine = InterpolationEngine()
    path_embeddings = np.array([]) 
    if method == "linear":
        path_embeddings = interpolation_engine.linear_interpolation(start_emb, end_emb, steps)
    elif method == "slerp":
        path_embeddings = interpolation_engine.slerp(start_emb, end_emb, steps)
    else:
        pytest.fail(f"Test does not support method: {method}")
    
    assert path_embeddings.size > 0, "Interpolation resulted in an empty path."
    assert path_embeddings.shape[0] == steps, f"Path has {path_embeddings.shape[0]} points, expected {steps}."
    print(f"Generated path_embeddings: Shape: {path_embeddings.shape}, Dtype: {path_embeddings.dtype}, Flags: {path_embeddings.flags}", flush=True)
    # print(f"Sample path_embeddings[0, :5]: {path_embeddings[0, :5]}", flush=True) # Uncomment for value inspection
    assert not (np.isnan(path_embeddings).any() or np.isinf(path_embeddings).any()), "Path embeddings contain NaN/Inf"
    
    print(f"Searching for {k_neighbors} nearest neighbors for {path_embeddings.shape[0]} path points...", flush=True)
    all_neighbors_results = [[]] 
    if k_neighbors > 0:
        print(f"Preparing path_embeddings for search. Current dtype: {path_embeddings.dtype}, flags: {path_embeddings.flags}", flush=True)
        # Ensure path_embeddings are float32 and C-contiguous before passing to search_by_embeddings
        # Although search_by_embeddings -> FAISSEngine.search should handle this, being explicit here for testing
        print("Attempting: path_embeddings_clean = np.ascontiguousarray(path_embeddings, dtype=np.float32)", flush=True)
        path_embeddings_clean = np.ascontiguousarray(path_embeddings, dtype=np.float32)
        print(f"Cleaned path_embeddings for search - Shape: {path_embeddings_clean.shape}, Dtype: {path_embeddings_clean.dtype}, Flags: {path_embeddings_clean.flags}", flush=True)

        all_neighbors_results = em.search_by_embeddings(path_embeddings_clean, k=k_neighbors)
        assert len(all_neighbors_results) == steps, "Neighbor search did not return results for all path points."
        print(f"Neighbor search completed. Results for {len(all_neighbors_results)} points.", flush=True)

        # Check and print neighbors for the first path point
        if steps > 0 and len(all_neighbors_results) > 0 and len(all_neighbors_results[0]) > 0:
            print(f"Neighbors for first path point ({all_neighbors_results[0][0]['text']}, score: {all_neighbors_results[0][0]['score']:.4f}...)", flush=True)
            assert len(all_neighbors_results[0]) <= k_neighbors, "More neighbors returned than requested for first point."
            
            # Check and print neighbors for the second path point
            print(f"Checking for neighbors for the second path point (index 1)...", flush=True)
            if len(all_neighbors_results) > 1 and len(all_neighbors_results[1]) > 0:
                print(f"Neighbors for second path point ({all_neighbors_results[1][0]['text']}, score: {all_neighbors_results[1][0]['score']:.4f}...)", flush=True)
                assert len(all_neighbors_results[1]) <= k_neighbors, "More neighbors returned than requested for second point."
            elif len(all_neighbors_results) <= 1:
                print(f"Cannot check second path point: all_neighbors_results has {len(all_neighbors_results)} item(s) (expected >1 for steps={steps}).", flush=True)
            else: # len(all_neighbors_results) > 1 but len(all_neighbors_results[1]) == 0
                print(f"No neighbors found for the second path point (all_neighbors_results[1] is empty). len(all_neighbors_results[1]): {len(all_neighbors_results[1])}", flush=True)

            # Original assertions for the first point's first neighbor's structure
            if len(all_neighbors_results[0]) > 0: 
                 assert "text" in all_neighbors_results[0][0], "Neighbor result item missing 'text' key for first point."
                 assert "score" in all_neighbors_results[0][0], "Neighbor result item missing 'score' key for first point."
        elif steps > 0: # This means len(all_neighbors_results[0]) == 0 or len(all_neighbors_results) == 0
            print(f"No neighbors found for the first path point, or main results list is empty. len(all_neighbors_results): {len(all_neighbors_results)}", flush=True)
            if len(all_neighbors_results) > 0:
                print(f"len(all_neighbors_results[0]): {len(all_neighbors_results[0])}", flush=True)
    else: # k_neighbors == 0
        print("k_neighbors is 0, skipping neighbor search.", flush=True)

    print("--- test_interpolation_path_logic completed successfully ---", flush=True) 