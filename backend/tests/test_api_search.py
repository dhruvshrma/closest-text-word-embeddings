import pytest
import httpx
from typing import Dict, Any

BASE_URL = "http://localhost:8000/api/v1"

@pytest.mark.asyncio
async def test_load_corpus_nltk_success():
    """Test loading a small NLTK corpus successfully."""
    payload = {
        "dataset_name": "nltk_common_words",
        "max_words": 100, # Keep small for quick testing
        "min_word_length": 3,
        "rebuild_index": True
    }
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        response = await client.post("/search/load-corpus", json=payload, timeout=60) # Increased timeout for loading
    
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Corpus 'nltk_common_words' loaded successfully."
    assert response_data["num_documents_loaded"] == 100
    # The index_status check can be more specific if you know FAISS will be enabled/disabled
    assert "vectors" in response_data["index_status"] or "Built/Updated" in response_data["index_status"] or "FAISS disabled" in response_data["index_status"]

@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow if ConceptNet loading is inherently slow
async def test_load_corpus_conceptnet_small():
    """Test loading a very small ConceptNet corpus successfully."""
    payload = {
        "dataset_name": "conceptnet_words",
        "max_words": 50,  # Keep very small for testing
        "min_word_length": 4,
        "rebuild_index": True
    }
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        response = await client.post("/search/load-corpus", json=payload, timeout=300) # Significantly increased timeout
    
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Corpus 'conceptnet_words' loaded successfully."
    assert response_data["num_documents_loaded"] > 0 
    assert response_data["num_documents_loaded"] <= 50 # Or a more precise check if possible
    assert "vectors" in response_data["index_status"] or "Built/Updated" in response_data["index_status"] or "FAISS disabled" in response_data["index_status"]

@pytest.mark.asyncio
async def test_search_after_load_nltk():
    """Test search functionality after loading NLTK corpus."""
    load_payload = {
        "dataset_name": "nltk_common_words",
        "max_words": 100,
        "rebuild_index": True
    }
    search_payload = {
        "text": "word", # A common word likely in NLTK top 100
        "k": 5
    }
    
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # 1. Load corpus
        load_response = await client.post("/search/load-corpus", json=load_payload, timeout=60)
        assert load_response.status_code == 200
        
        # 2. Perform search
        search_response = await client.post("/search/search", json=search_payload, timeout=10)
        assert search_response.status_code == 200
        search_data = search_response.json()
        assert search_data["query_text"] == "word"
        assert len(search_data["results"]) <= 5 # Should be k or less if fewer matches
        if search_data["results"]:
            assert "text" in search_data["results"][0]
            assert "index" in search_data["results"][0]
            assert "distance" in search_data["results"][0]

