from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Type
import hashlib
import pickle
from pathlib import Path


# Placeholder for FAISSEngine if not available
class FAISSEnginePlaceholder:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        print(
            "Warning: FAISSEnginePlaceholder initialized. FAISS is not installed or FAISSEngine could not be imported."
        )
        # We don't raise error here, but in methods, to allow EmbeddingManager to initialize.
        self._is_placeholder = True

    def build_index(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "FAISS is not installed or FAISSEngine could not be imported."
        )

    def search(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "FAISS is not installed or FAISSEngine could not be imported."
        )

    @property
    def index(self) -> Any:
        # Allows checks like `self.faiss_engine.index is None` without erroring if placeholder
        return None

    @property
    def ntotal(self) -> int:
        return 0


# Attempt to import FAISSEngine
FAISSEngineImportType: Type[Any]
try:
    from core.search_engines import FAISSEngine as ActualFAISSEngine

    FAISSEngineImportType = ActualFAISSEngine
    print("Successfully imported FAISSEngine.")
except ImportError:
    print(
        "Warning: FAISSEngine not found during import. Using FAISSEnginePlaceholder. Search functionalities will be disabled."
    )
    FAISSEngineImportType = FAISSEnginePlaceholder


class EmbeddingManager:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "./data/cache",
        use_faiss_index: bool = False,
        faiss_metric: str = "cosine",
    ):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        if self.embedding_dim is None:  # Should not happen with a valid model
            raise ValueError(
                "Could not determine embedding dimension from the SentenceTransformer model."
            )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.corpus_embeddings: Optional[np.ndarray] = None
        self.corpus_texts: Optional[List[str]] = None
        self.faiss_engine: Optional[FAISSEngineImportType] = None
        self._faiss_actually_enabled = False
        self.initial_use_faiss_request = use_faiss_index  # Store the initial request

        if use_faiss_index:
            try:
                # Instantiate the imported type (could be placeholder or actual)
                temp_engine = FAISSEngineImportType(
                    dimension=self.embedding_dim, metric=faiss_metric
                )

                # Check if it's the placeholder by checking the attribute we added or its type
                if hasattr(temp_engine, "_is_placeholder") or isinstance(
                    temp_engine, FAISSEnginePlaceholder
                ):
                    print(
                        "FAISS features disabled as FAISSEnginePlaceholder is active."
                    )
                    self._faiss_actually_enabled = False
                    self.faiss_engine = None  # Explicitly set to None if placeholder
                else:
                    self.faiss_engine = temp_engine
                    self._faiss_actually_enabled = True
                    print("FAISS engine successfully initialized.")

            except (
                NotImplementedError
            ):  # Should be caught if placeholder methods are directly called before check
                print(
                    "FAISS is not installed (NotImplementedError during init). Disabling FAISS features."
                )
                self._faiss_actually_enabled = False
                self.faiss_engine = None
            except Exception as e:
                print(f"Error initializing FAISSEngine: {e}. Disabling FAISS features.")
                self._faiss_actually_enabled = False
                self.faiss_engine = None
        else:
            self._faiss_actually_enabled = False
            self.faiss_engine = None

    def get_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:

        sorted_texts_key = "".join(sorted(texts)).encode("utf-8")
        cache_key = hashlib.md5(sorted_texts_key).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        if use_cache and cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        embeddings = self.model.encode(texts, convert_to_numpy=True)

        if use_cache:
            with open(cache_path, "wb") as f:
                pickle.dump(embeddings, f)

        return embeddings

    def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Helper to get embedding for a single text."""
        return self.get_embeddings([text], use_cache=use_cache)[0]

    def load_corpus(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray] = None,
        rebuild_index: bool = False,
    ) -> None:
        """Loads a corpus of texts and their embeddings, and builds the FAISS index if enabled."""
        print(f"Loading corpus with {len(texts)} documents.")
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
            # Check ntotal directly on the faiss_engine.index which might be None if placeholder was used and then faiss_engine set to None
            # Or, more robustly, check if the engine itself has an index and if it needs rebuilding.
            current_ntotal = 0
            if (
                self.faiss_engine.index is not None
            ):  # FAISSEngine (actual) has .index, Placeholder also has .index property
                # The placeholder's index property returns None, actual FAISSEngine index might be an object or None before add
                # The actual FAISS index object has ntotal attribute. Placeholder now has ntotal property.
                current_ntotal = (
                    self.faiss_engine.index.ntotal
                    if hasattr(self.faiss_engine.index, "ntotal")
                    else 0
                )
                if isinstance(self.faiss_engine, FAISSEnginePlaceholder):
                    current_ntotal = (
                        self.faiss_engine.ntotal
                    )  # Use property for placeholder

            if rebuild_index or current_ntotal == 0:
                print("Building FAISS index for the corpus...")
                self.faiss_engine.build_index(
                    self.corpus_embeddings
                )  # build_index would raise if placeholder
            else:
                print(
                    f"FAISS index already has {current_ntotal} vectors and rebuild_index is False."
                )
        elif self.initial_use_faiss_request and not self._faiss_actually_enabled:
            # This condition means user wanted FAISS, but it couldn't be initialized.
            print(
                "Warning: FAISS was requested (use_faiss_index=True) but is not available/initialized. Index not built."
            )

    def search_corpus(
        self, query_texts: List[str], k: int = 5
    ) -> Optional[List[List[Dict[str, any]]]]:
        if not self._faiss_actually_enabled or not self.faiss_engine:
            print("FAISS index not available or not built. Cannot perform search.")
            return None

        if self.corpus_texts is None or self.corpus_embeddings is None:
            print("Corpus texts or embeddings not loaded. Cannot perform search.")
            return None

        # Ensure faiss_engine.index is usable (it would be None if placeholder was active and then self.faiss_engine set to None)
        # Or if it's the actual engine but index hasn't been built.
        # The FAISSEngine.search method itself should check if its index is ready.
        if self.faiss_engine.index is None or (
            hasattr(self.faiss_engine.index, "ntotal")
            and self.faiss_engine.index.ntotal == 0
        ):
            if not isinstance(
                self.faiss_engine, FAISSEnginePlaceholder
            ):  # Don't print if it's just the placeholder (already warned)
                print(
                    "FAISS index is not built or is empty. Call load_corpus to build it."
                )
            return None

        print(
            f"Searching for {k} nearest neighbors for {len(query_texts)} queries with FAISS..."
        )
        query_embeddings = self.get_embeddings(
            query_texts,
            use_cache=False,
        )

        try:
            distances, indices = self.faiss_engine.search(query_embeddings, k)
        except NotImplementedError:
            print("Search failed: FAISS is not properly installed or initialized.")
            return None
        except Exception as e:
            print(f"Error during FAISS search: {e}")
            return None

        results = []
        for i in range(len(query_texts)):
            query_results = []
            for j in range(k):
                neighbor_idx = indices[i][j]
                neighbor_dist = distances[i][j]
                if 0 <= neighbor_idx < len(self.corpus_texts):
                    query_results.append(
                        {
                            "text": self.corpus_texts[neighbor_idx],
                            "index": int(neighbor_idx),
                            "distance": float(neighbor_dist),
                        }
                    )
                else:
                    print(
                        f"Warning: Neighbor index {neighbor_idx} out of bounds for corpus size {len(self.corpus_texts)}."
                    )
            results.append(query_results)

        return results

    def get_available_models(self) -> List[Dict[str, str]]:
        return [
            {
                "model_id": (
                    self.model.tokenizer.name_or_path
                    if self.model.tokenizer.name_or_path
                    else "Unknown"
                ),
                "description": f"SentenceTransformer model: {self.model.tokenizer.name_or_path if self.model.tokenizer.name_or_path else 'N/A'}",
            }
        ]
