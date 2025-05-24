from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Type, Union
import hashlib
import pickle
from pathlib import Path
from core.search_engines import FAISSEngine
from config.config import settings

# Conditional FAISS import
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


class EmbeddingManager:
    def __init__(
        self,
        model_name: str = settings.default_model_name,
        cache_dir: Optional[Union[str, Path]] = None,
        use_faiss_index: bool = True,
        faiss_metric: str = "cosine",
    ):
        self.model_name_or_path = model_name
        self.model = SentenceTransformer(self.model_name_or_path)
        embed_dim = self.model.get_sentence_embedding_dimension()
        if embed_dim is None:
            raise ValueError(
                f"Could not determine embedding dimension from model: {self.model_name_or_path}"
            )
        self.embedding_dim: int = embed_dim

        effective_cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path(settings.default_embedding_cache_path)
        )
        self.cache_dir = effective_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.corpus_texts: Optional[List[str]] = None
        self.corpus_embeddings: Optional[np.ndarray] = None

        self.initial_use_faiss_request = use_faiss_index
        self._faiss_actually_enabled = FAISS_AVAILABLE and use_faiss_index

        self.faiss_engine: Optional[FAISSEngine] = None

        if self._faiss_actually_enabled and faiss is not None:
            try:
                self.faiss_engine = FAISSEngine(
                    dimension=self.embedding_dim, metric=faiss_metric
                )
                print(
                    f"EmbeddingManager initialized with FAISSEngine (Metric: {faiss_metric}, Dim: {self.embedding_dim})."
                )
            except Exception as e:
                print(f"Error initializing FAISSEngine: {e}. FAISS will be disabled.")
                self._faiss_actually_enabled = False
                self.faiss_engine = None
        elif use_faiss_index and not FAISS_AVAILABLE:
            print(
                "Warning: FAISS use was requested, but FAISS library is not installed. FAISS features will be disabled."
            )
        else:
            print(
                f"EmbeddingManager initialized without FAISS (requested: {use_faiss_index}, available: {FAISS_AVAILABLE})."
            )

    def _get_cache_path(
        self, texts: List[str], model_name_override: Optional[str] = None
    ) -> Path:
        sorted_texts_key = "".join(sorted(texts)).encode("utf-8")
        cache_key = hashlib.md5(sorted_texts_key).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        return cache_path

    def get_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        if not texts:
            return np.array([])

        cache_path = self._get_cache_path(texts)

        if use_cache and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    embeddings = pickle.load(f)
                # Basic validation of loaded embeddings
                if (
                    isinstance(embeddings, np.ndarray)
                    and embeddings.shape[0] == len(texts)
                    and embeddings.shape[1] == self.embedding_dim
                ):
                    print(f"Loaded embeddings from cache: {cache_path}")
                    return embeddings
                else:
                    print(f"Cache content mismatch for {cache_path}. Recomputing.")
            except Exception as e:
                print(
                    f"Error loading embeddings from cache {cache_path}: {e}. Recomputing."
                )

        # Compute embeddings
        embeddings_array = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        if not isinstance(embeddings_array, np.ndarray):
            # This case should ideally not happen with sentence-transformers returning numpy arrays
            raise TypeError(
                f"Expected numpy array from model encoding, got {type(embeddings_array)}"
            )

        # Save to cache if use_cache is True
        if use_cache:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(embeddings_array, f)
                print(f"Saved embeddings to cache: {cache_path}")
            except Exception as e:
                print(f"Error saving embeddings to cache {cache_path}: {e}")

        return embeddings_array

    def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Helper to get a single embedding (as a 2D array with one row)."""
        # Note: Caching for single texts via get_embeddings([text]) might create many small files.
        # Consider if this is desired or if single text caching needs a different strategy.
        # For now, it reuses the list-based caching.
        embeddings = self.get_embeddings([text], use_cache=use_cache)
        return embeddings[0]  # Return the first (and only) embedding vector

    def load_corpus(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray] = None,
        rebuild_index: bool = False,
    ) -> None:
        """Loads a corpus of texts and their embeddings, and builds the FAISS index if enabled."""
        print(f"Loading corpus with {len(texts)} documents for on-the-fly processing.")
        self.corpus_texts = texts
        if embeddings is not None:
            self.corpus_embeddings = embeddings
        else:
            self.corpus_embeddings = self.get_embeddings(texts, use_cache=True)

        if self.corpus_embeddings is None:
            raise ValueError("Failed to load or generate embeddings for the corpus.")

        if self.embedding_dim != self.corpus_embeddings.shape[1]:
            raise ValueError(
                f"Corpus embedding dimension ({self.corpus_embeddings.shape[1]}) "
                f"does not match model dimension ({self.embedding_dim})."
            )

        if self._faiss_actually_enabled and self.faiss_engine:
            current_ntotal = 0
            if self.faiss_engine.index is not None:
                current_ntotal = (
                    self.faiss_engine.index.ntotal
                    if hasattr(self.faiss_engine.index, "ntotal")
                    else 0
                )

            if rebuild_index or current_ntotal == 0:
                print("Building FAISS index for the corpus (on-the-fly)...")
                # Ensure corpus_embeddings is not None before building
                if self.corpus_embeddings is not None:
                    self.faiss_engine.build_index(self.corpus_embeddings)
                else:
                    print(
                        "Error: Cannot build FAISS index because corpus_embeddings is None."
                    )
            else:
                print(
                    f"FAISS index (on-the-fly) already has {current_ntotal} vectors and rebuild_index is False."
                )
        elif self.initial_use_faiss_request and not self._faiss_actually_enabled:
            print(
                "Warning: FAISS was requested for on-the-fly processing but is not available/initialized. Index not built."
            )

    def load_prebuilt_data(
        self,
        words_path: Path,
        embeddings_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
    ) -> None:
        """Loads pre-built words, embeddings, and FAISS index from specified file paths."""
        print(f"--- Loading Pre-built Data ---")
        print(f"Words path: {words_path}")
        if embeddings_path:
            print(f"Embeddings path: {embeddings_path}")
        if index_path:
            print(f"FAISS Index path: {index_path}")

        if not words_path.exists():
            raise FileNotFoundError(f"Required words file not found: {words_path}")

        try:
            with open(words_path, "rb") as f:
                self.corpus_texts = pickle.load(f)
            if not self.corpus_texts or not isinstance(self.corpus_texts, list):
                raise ValueError(
                    f"Words file {words_path} did not contain a valid list of texts."
                )
            print(
                f"Successfully loaded {len(self.corpus_texts)} words from {words_path}."
            )
        except Exception as e:
            raise IOError(f"Error loading or parsing words file {words_path}: {e}")

        if embeddings_path and embeddings_path.exists():
            try:
                loaded_embeddings = np.load(str(embeddings_path))
                print(
                    f"Successfully loaded {loaded_embeddings.shape[0]} embeddings (dim: {loaded_embeddings.shape[1]}) from {embeddings_path}."
                )
                if self.embedding_dim != loaded_embeddings.shape[1]:
                    raise ValueError(
                        f"Dimension mismatch: Model expects {self.embedding_dim}, loaded embeddings have {loaded_embeddings.shape[1]}. Ensure the correct model ({self.model_name_or_path}) is used for these pre-built embeddings."
                    )
                if not self.corpus_texts:
                    raise ValueError(
                        "Cannot validate embeddings count: corpus_texts not loaded before embeddings."
                    )
                if len(self.corpus_texts) != loaded_embeddings.shape[0]:
                    raise ValueError(
                        f"Data mismatch: Number of loaded words ({len(self.corpus_texts)}) does not match number of loaded embeddings ({loaded_embeddings.shape[0]})."
                    )
                self.corpus_embeddings = loaded_embeddings
            except Exception as e:
                raise IOError(f"Error loading embeddings file {embeddings_path}: {e}")
        elif not (index_path and index_path.exists()):
            print(
                "Embeddings file not found or not provided, and no FAISS index path provided. Attempting to generate embeddings for loaded words..."
            )
            if not self.corpus_texts:
                raise ValueError("Cannot generate embeddings: corpus_texts not loaded.")
            self.corpus_embeddings = self.get_embeddings(
                self.corpus_texts, use_cache=True
            )
            if self.corpus_embeddings is not None and self.corpus_embeddings.size > 0:
                print(f"Generated {self.corpus_embeddings.shape[0]} embeddings.")
            else:
                print(
                    "Warning: Failed to generate or generated empty embeddings for loaded words."
                )
                # self.corpus_embeddings might be an empty array, ensure it's None if truly failed
                if (
                    self.corpus_embeddings is not None
                    and self.corpus_embeddings.size == 0
                ):
                    self.corpus_embeddings = None

        if self._faiss_actually_enabled and self.faiss_engine and faiss is not None:
            if index_path and index_path.exists():
                try:
                    print(f"Loading FAISS index from {index_path}...")
                    loaded_index = faiss.read_index(str(index_path))
                    if self.faiss_engine.dimension != loaded_index.d:
                        raise ValueError(
                            f"Dimension mismatch: FAISS index dimension ({loaded_index.d}) does not match FAISSEngine expected dimension ({self.faiss_engine.dimension}). Ensure consistency with model."
                        )
                    self.faiss_engine.index = loaded_index
                    print(
                        f"Successfully loaded FAISS index with {self.faiss_engine.index.ntotal} vectors from {index_path}."
                    )

                    if (
                        self.corpus_texts
                        and self.faiss_engine.index is not None
                        and len(self.corpus_texts) != self.faiss_engine.index.ntotal
                    ):
                        print(
                            f"Warning: Mismatch between loaded texts count ({len(self.corpus_texts)}) and FAISS index size ({self.faiss_engine.index.ntotal})."
                        )
                except Exception as e:
                    print(
                        f"Error loading FAISS index from {index_path}: {e}. Attempting to build from embeddings if available."
                    )
                    if (
                        self.corpus_embeddings is not None
                        and self.corpus_embeddings.size > 0
                    ):
                        print(
                            "Building FAISS index from loaded/generated embeddings..."
                        )
                        self.faiss_engine.build_index(self.corpus_embeddings)
                    else:
                        print(
                            "Cannot build FAISS index: corpus_embeddings not available or empty after failed index load."
                        )
            elif self.corpus_embeddings is not None and self.corpus_embeddings.size > 0:
                print(
                    "Pre-built FAISS index not found/provided. Building index from loaded/generated embeddings..."
                )
                self.faiss_engine.build_index(self.corpus_embeddings)
            else:
                print(
                    "FAISS enabled, but no pre-built index provided and no embeddings available/valid to build one."
                )
        elif self.initial_use_faiss_request:
            print(
                "FAISS was requested but is not available/enabled. Skipping FAISS index loading/building."
            )

        print("--- Pre-built Data Loading Attempt Finished ---")
        if self.corpus_texts:
            print(f"Status: {len(self.corpus_texts)} texts loaded.")
        if self.corpus_embeddings is not None:
            print(
                f"Status: {self.corpus_embeddings.shape[0]} embeddings loaded/generated."
            )
        if self.faiss_engine and self.faiss_engine.index:
            print(
                f"Status: FAISS index active with {self.faiss_engine.index.ntotal} vectors."
            )

    def search_corpus(
        self, query_texts: List[str], k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """Searches the loaded corpus for nearest neighbors to the query texts."""
        if self.corpus_texts is None or self.corpus_embeddings is None:
            raise RuntimeError(
                "Corpus not loaded. Call load_corpus() or load_prebuilt_data() first."
            )
        if (
            not self._faiss_actually_enabled
            or not self.faiss_engine
            or self.faiss_engine.index is None
        ):
            print(
                "Warning: FAISS search unavailable in search_corpus. Returning empty results."
            )
            return [[] for _ in query_texts]

        if self.faiss_engine.index.ntotal == 0:
            print(
                "Warning: FAISS index is empty in search_corpus. Returning empty results."
            )
            return [[] for _ in query_texts]

        query_embeddings = self.get_embeddings(query_texts, use_cache=True)
        if query_embeddings.size == 0:
            return [[] for _ in query_texts]

        # Delegate to search_by_embeddings
        return self.search_by_embeddings(query_embeddings, k=k)

    def search_by_embeddings(
        self, query_embeddings: np.ndarray, k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """Searches the loaded corpus for nearest neighbors using pre-computed query embeddings."""
        if self.corpus_texts is None:  # Corpus texts are needed to return results
            raise RuntimeError(
                "Corpus texts not loaded. Cannot map search results to text. Call load_corpus() or load_prebuilt_data() first."
            )

        if (
            not self._faiss_actually_enabled
            or not self.faiss_engine
            or self.faiss_engine.index is None
        ):
            print(
                "Warning: FAISS search unavailable in search_by_embeddings. Returning empty results."
            )
            return [[] for _ in range(query_embeddings.shape[0])]

        if self.faiss_engine.index.ntotal == 0:
            print(
                "Warning: FAISS index is empty in search_by_embeddings. Returning empty results."
            )
            return [[] for _ in range(query_embeddings.shape[0])]

        if query_embeddings.ndim == 1:  # If a single embedding vector is passed
            query_embeddings = np.expand_dims(query_embeddings, axis=0)

        if query_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension ({query_embeddings.shape[1]}) does not match corpus embedding dimension ({self.embedding_dim})."
            )

        if query_embeddings.size == 0:
            return [
                [] for _ in range(query_embeddings.shape[0])
            ]  # Should match input shape

        try:
            distances, indices = self.faiss_engine.search(query_embeddings, k=k)
        except Exception as e:
            print(f"Error during FAISS search in search_by_embeddings: {e}")
            return [[] for _ in range(query_embeddings.shape[0])]

        results = []
        for i in range(query_embeddings.shape[0]):  # Iterate over each query embedding
            single_query_results = []
            for j in range(k):
                idx = indices[i][j]
                if idx < 0 or idx >= len(self.corpus_texts):
                    continue
                dist = distances[i][j]
                single_query_results.append(
                    {
                        "text": self.corpus_texts[idx],
                        "score": float(dist),
                        "id": int(idx),
                    }
                )
            results.append(single_query_results)
        return results

    def get_available_models(self) -> List[Dict[str, str]]:
        # This can be expanded to dynamically list models or fetch from a config
        # For now, returns the current model of this manager instance
        return [
            {
                "model_id": self.model_name_or_path,
                "description": f"Currently active model: {self.model_name_or_path}",
                "embedding_dim": str(self.embedding_dim),  # Pydantic model expects str
            }
        ]

    def get_corpus_status(self) -> Dict[str, Any]:
        """Returns the status of the loaded corpus."""
        return {
            "num_texts": len(self.corpus_texts) if self.corpus_texts else 0,
            "embeddings_loaded": self.corpus_embeddings is not None
            and self.corpus_embeddings.size > 0,
            "embedding_dim": (
                self.corpus_embeddings.shape[1]
                if self.corpus_embeddings is not None
                and self.corpus_embeddings.ndim == 2
                else None
            ),
            "faiss_index_active": self._faiss_actually_enabled
            and self.faiss_engine is not None
            and self.faiss_engine.index is not None
            and self.faiss_engine.index.ntotal > 0,
            "faiss_index_size": (
                self.faiss_engine.index.ntotal
                if self._faiss_actually_enabled
                and self.faiss_engine is not None
                and self.faiss_engine.index is not None
                else 0
            ),
            "faiss_metric": (
                self.faiss_engine.metric
                if self._faiss_actually_enabled and self.faiss_engine
                else None
            ),
            "model_name": self.model_name_or_path,
        }
