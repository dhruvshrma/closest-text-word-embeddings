import pytest
import numpy as np
import time
from pathlib import Path
import shutil # For cleaning up cache directory
import hashlib

from core.embedding_manager import EmbeddingManager

@pytest.fixture(scope="module")
def manager():
    test_cache_dir = "./data/cache_test_temp"
    manager = EmbeddingManager(cache_dir=test_cache_dir)
    yield manager
    if Path(test_cache_dir).exists():
        shutil.rmtree(test_cache_dir)

def test_get_single_embedding(manager: EmbeddingManager):
    word = "cat"
    embedding = manager.get_embeddings(word, use_cache=True)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)

def test_get_multiple_embeddings(manager: EmbeddingManager):
    words = ["cat", "dog", "animal"]
    embeddings = manager.get_embeddings(words)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(words), 384)

def test_caching_mechanism(manager: EmbeddingManager):
    words = ["hello", "world"]
    
        # First call: should generate and cache
    start_time = time.time()
    embeddings1 = manager.get_embeddings(words, use_cache=True)
    duration1 = time.time() - start_time
    
        # Construct expected cache file path
    sorted_texts_key = ''.join(sorted(words)).encode('utf-8')
    cache_key = hashlib.md5(sorted_texts_key).hexdigest()
    cache_file = manager.cache_dir / f"{cache_key}.pkl"
    
    assert cache_file.exists(), "Cache file was not created."

    # Second call: should load from cache
    start_time = time.time()
    embeddings2 = manager.get_embeddings(words, use_cache=True)
    duration2 = time.time() - start_time
    
    assert np.array_equal(embeddings1, embeddings2), "Cached embeddings do not match original."
    
    print(f"Duration without cache: {duration1}, Duration with cache: {duration2}")

def test_get_embeddings_no_cache(manager: EmbeddingManager):
    words = ["test", "cache", "disabled"]
   
    embeddings1 = manager.get_embeddings(words, use_cache=False)
    
    # Construct expected cache file path
    sorted_texts_key = ''.join(sorted(words)).encode('utf-8')
    cache_key = hashlib.md5(sorted_texts_key).hexdigest()
    cache_file = manager.cache_dir / f"{cache_key}.pkl"
    
    assert not cache_file.exists(), "Cache file was created even when use_cache was False."
    
    # Second call with cache disabled again, should still not create a cache file
    embeddings2 = manager.get_embeddings(words, use_cache=False)
    assert not cache_file.exists(), "Cache file was created on second call with use_cache=False."
    assert np.array_equal(embeddings1, embeddings2)

def test_cache_different_word_order(manager: EmbeddingManager):
    words1 = ["apple", "banana"]
    words2 = ["banana", "apple"]

   
    embeddings1 = manager.get_embeddings(words1, use_cache=True)

    
    embeddings2 = manager.get_embeddings(words2, use_cache=True)


    assert np.array_equal(embeddings1, embeddings2), "Embeddings differ for different order of same words with caching."


    sorted_texts_key = ''.join(sorted(words1)).encode('utf-8') # or words2, doesn't matter
    cache_key = hashlib.md5(sorted_texts_key).hexdigest()
    cache_file = manager.cache_dir / f"{cache_key}.pkl"
    assert cache_file.exists(), "Cache file was not created for sorted words." 