# backend/scripts/build_conceptnet_index.py
import sys
import pathlib # Will be replaced by os where path joining is done
import os # Import os module
import pickle
import numpy as np
import faiss # Ensure faiss is installed
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

try:
    from data.dataset import ConceptNetWordsDataset
    from core.embedding_manager import EmbeddingManager
    from core.search_engines import FAISSEngine 
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(f"PROJECT_ROOT ({PROJECT_ROOT}) should be the parent of 'backend'.")
    print("Ensure the script is run from a context where 'backend.src' is importable,")
    print("or that your PYTHONPATH is set up correctly. Current sys.path includes this script's attempt to add PROJECT_ROOT.")
    sys.exit(1)

# --- Configuration --- 
MAX_WORDS = 600000  # Target number of words from ConceptNet.
MIN_WORD_LENGTH = 3

CONCEPTNET_STREAM_LIMIT = 0 

MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_METRIC = 'cosine' # 'cosine' or 'l2'

OUTPUT_DATA_ROOT_DIR = os.path.join(BACKEND_DIR, "data")
CONCEPTNET_WORDS_FILENAME = f"conceptnet_{MAX_WORDS}_words.pkl"
CONCEPTNET_EMBEDDINGS_FILENAME = f"conceptnet_{MAX_WORDS}_embeddings_{MODEL_NAME.replace('/','_')}.npy"
CONCEPTNET_FAISS_INDEX_FILENAME = f"conceptnet_{MAX_WORDS}_faiss_{MODEL_NAME.replace('/','_')}_{FAISS_METRIC}.index"

CONCEPTNET_WORDS_PATH = os.path.join(OUTPUT_DATA_ROOT_DIR, CONCEPTNET_WORDS_FILENAME)
CONCEPTNET_EMBEDDINGS_PATH = os.path.join(OUTPUT_DATA_ROOT_DIR, CONCEPTNET_EMBEDDINGS_FILENAME)
CONCEPTNET_FAISS_INDEX_PATH = os.path.join(OUTPUT_DATA_ROOT_DIR, CONCEPTNET_FAISS_INDEX_FILENAME)

HF_DATASET_CACHE_DIR = os.path.join(BACKEND_DIR, "src", "data", "cache", "huggingface_datasets_loader_cache")

SCRIPT_EMBEDDING_CACHE_DIR = os.path.join(OUTPUT_DATA_ROOT_DIR, "cache", "script_embeddings_cache")

def main():
    print("--- Starting ConceptNet Full Index Build Script ---")
    start_time_total = time.time()

    # Create directories if they don't exist using os.makedirs
    os.makedirs(OUTPUT_DATA_ROOT_DIR, exist_ok=True)
    os.makedirs(HF_DATASET_CACHE_DIR, exist_ok=True)
    os.makedirs(SCRIPT_EMBEDDING_CACHE_DIR, exist_ok=True)

    # 1. Load ConceptNet Words
    print(f"\n[PHASE 1/5] Loading ConceptNet words...")
    print(f" - Target max words: {MAX_WORDS}, Min length: {MIN_WORD_LENGTH}")
    print(f" - ConceptNetWordsDataset stream_limit: {'Unlimited' if CONCEPTNET_STREAM_LIMIT == 0 else CONCEPTNET_STREAM_LIMIT}")
    print(f" - HuggingFace dataset cache: {HF_DATASET_CACHE_DIR}")
    phase_start_time = time.time()
    try:
        dataset_loader = ConceptNetWordsDataset(
            max_words=MAX_WORDS,
            min_word_length=MIN_WORD_LENGTH,
            stream_limit=CONCEPTNET_STREAM_LIMIT, 
            cache_dir=str(HF_DATASET_CACHE_DIR) # ConceptNetWordsDataset might expect string path
        )
        conceptnet_words = dataset_loader.load() # THIS IS THE VERY LONG PART
    except Exception as e:
        print(f"FATAL: Error loading ConceptNetWordsDataset: {e}")
        sys.exit(1)
    print(f"Phase 1 completed in {time.time() - phase_start_time:.2f} seconds.")

    if not conceptnet_words:
        print("FATAL: No words loaded from ConceptNet. Exiting.")
        sys.exit(1)
    print(f"Successfully loaded {len(conceptnet_words)} words from ConceptNet.")

    # 2. Save the words list
    print(f"\n[PHASE 2/5] Saving ConceptNet words list...")
    phase_start_time = time.time()
    try:
        with open(CONCEPTNET_WORDS_PATH, 'wb') as f:
            pickle.dump(conceptnet_words, f)
        print(f"ConceptNet words saved to: {CONCEPTNET_WORDS_PATH}")
    except Exception as e:
        print(f"Error saving ConceptNet words: {e}")
    print(f"Phase 2 completed in {time.time() - phase_start_time:.2f} seconds.")

    # 3. Initialize EmbeddingManager and Generate Embeddings
    print(f"\n[PHASE 3/5] Generating embeddings for {len(conceptnet_words)} words...")
    print(f" - Model: {MODEL_NAME}")
    print(f" - Embedding cache for script: {SCRIPT_EMBEDDING_CACHE_DIR}")
    phase_start_time = time.time()
    try:
        embedding_manager = EmbeddingManager(
            model_name=MODEL_NAME,
            cache_dir=str(SCRIPT_EMBEDDING_CACHE_DIR), # EmbeddingManager expects string or Path
            use_faiss_index=False 
        )
        word_embeddings = embedding_manager.get_embeddings(conceptnet_words, use_cache=True)
    except Exception as e:
        print(f"FATAL: Error generating embeddings: {e}")
        sys.exit(1)
    print(f"Phase 3 completed in {time.time() - phase_start_time:.2f} seconds.")
    print(f"Successfully generated {word_embeddings.shape[0]} embeddings of dimension {word_embeddings.shape[1]}.")

    # 4. Save embeddings
    print(f"\n[PHASE 4/5] Saving word embeddings...")
    phase_start_time = time.time()
    try:
        np.save(str(CONCEPTNET_EMBEDDINGS_PATH), word_embeddings)
        print(f"Word embeddings saved to: {CONCEPTNET_EMBEDDINGS_PATH}")
    except Exception as e:
        print(f"Error saving word embeddings: {e}")
    print(f"Phase 4 completed in {time.time() - phase_start_time:.2f} seconds.")

    # 5. Build and Save FAISS Index
    print(f"\n[PHASE 5/5] Building and saving FAISS index...")
    print(f" - FAISS metric: {FAISS_METRIC}")
    phase_start_time = time.time()
    embedding_dim = word_embeddings.shape[1]
    try:
        faiss_engine_builder = FAISSEngine(dimension=embedding_dim, metric=FAISS_METRIC)
        faiss_engine_builder.build_index(word_embeddings) 
        faiss_index_to_save = faiss_engine_builder.index
        print(f"FAISS index built. Index type: {type(faiss_index_to_save)}, Total vectors: {faiss_index_to_save.ntotal if faiss_index_to_save else 'N/A'}")
        
        if faiss_index_to_save:
            faiss.write_index(faiss_index_to_save, str(CONCEPTNET_FAISS_INDEX_PATH))
            print(f"FAISS index saved to: {CONCEPTNET_FAISS_INDEX_PATH}")
        else:
            print("No FAISS index was built, skipping save.")
    except Exception as e:
        print(f"FATAL: Error building or saving FAISS index: {e}")
        sys.exit(1)
    print(f"Phase 5 completed in {time.time() - phase_start_time:.2f} seconds.")

    total_duration = time.time() - start_time_total
    print(f"\n--- ConceptNet Full Index Build Script Finished in {total_duration // 60:.0f}m {total_duration % 60:.0f}s ---")

if __name__ == "__main__":

    print(f"Executing script: {__file__}")
    print(f"Current working directory: {os.getcwd()}") 
    print(f"Project root (derived): {PROJECT_ROOT}")
    print(f"System path includes: {PROJECT_ROOT}")
    main() 