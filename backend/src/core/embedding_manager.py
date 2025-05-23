from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import hashlib
import pickle
from pathlib import Path

class EmbeddingManager:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = './data/cache'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:

        sorted_texts_key = ''.join(sorted(texts)).encode('utf-8')
        cache_key = hashlib.md5(sorted_texts_key).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if use_cache and cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        if use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
        
        return embeddings

    def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Helper to get embedding for a single text."""
        return self.get_embeddings([text], use_cache=use_cache)[0] 