import pytest
from pathlib import Path
import os
from typing import List, Dict, Any

# Import our dataset class
from data.dataset import ConceptNetWordsDataset, _DEFAULT_HF_LOADER_CACHE_DIR

@pytest.mark.skip(reason="Skipping ConceptNet dataset tests")
class TestConceptNetDataset:
    """Test suite for ConceptNet dataset functionality"""
    
    @pytest.fixture
    def cache_dir(self) -> Path:
        """Fixture to provide the cache directory path"""
        return _DEFAULT_HF_LOADER_CACHE_DIR
    
    @pytest.fixture
    def dataset_loader(self, cache_dir: Path) -> ConceptNetWordsDataset:
        """Fixture to provide a configured dataset loader"""
        return ConceptNetWordsDataset(
            max_words=1000,  # Reasonable size for testing
            min_word_length=3,
            stream_limit=0,  # No streaming limit when using cached data
            cache_dir=cache_dir
        )
    
    def test_cache_exists(self, cache_dir: Path):
        """Verify that the cache directory exists and contains dataset files"""
        assert cache_dir.exists(), "Cache directory should exist"
        
        # Check for HuggingFace dataset cache files
        cache_files = list(cache_dir.glob("*"))
        assert len(cache_files) > 0, "Cache directory should contain dataset files"
        print(f"\nFound {len(cache_files)} files in cache directory:")
        for f in cache_files:
            print(f"  - {f.name}")
    
    def test_load_cached_dataset(self, dataset_loader: ConceptNetWordsDataset):
        """Test loading words from cached dataset"""
        words = dataset_loader.load()
        
        assert isinstance(words, list), "Should return a list of words"
        assert len(words) > 0, "Should return some words"
        assert all(isinstance(w, str) for w in words), "All items should be strings"
        assert all(len(w) >= dataset_loader.min_word_length for w in words), \
            f"All words should meet minimum length of {dataset_loader.min_word_length}"
        
        print(f"\nLoaded {len(words)} words from cached dataset")
        print(f"Sample of first 10 words: {words[:10]}")
        
        lengths = [len(w) for w in words]
        avg_len = sum(lengths) / len(lengths)
        print(f"Average word length: {avg_len:.1f}")
        print(f"Min word length: {min(lengths)}")
        print(f"Max word length: {max(lengths)}")
    
    def test_metadata(self, dataset_loader: ConceptNetWordsDataset):
        """Test metadata access"""
        metadata = dataset_loader.metadata
        
        assert isinstance(metadata, dict), "Metadata should be a dictionary"
        assert "name" in metadata, "Metadata should include dataset name"
        assert "description" in metadata, "Metadata should include description"
        
        print("\nDataset metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    def test_word_filtering(self):
        """Test dataset loading with different filtering parameters"""
        strict_loader = ConceptNetWordsDataset(
            max_words=500,
            min_word_length=5,  # Only longer words
            stream_limit=0,
            cache_dir=_DEFAULT_HF_LOADER_CACHE_DIR
        )
        
        strict_words = strict_loader.load()
        assert all(len(w) >= 5 for w in strict_words), "All words should be at least 5 characters"
        
        print(f"\nLoaded {len(strict_words)} words with min_length=5")
        print(f"Sample of longer words: {strict_words[:5]}")
    
    def test_cache_reuse(self):
        """Demonstrate that multiple loaders can use the same cache"""
        loader1 = ConceptNetWordsDataset(
            max_words=100,
            cache_dir=_DEFAULT_HF_LOADER_CACHE_DIR
        )
        
        loader2 = ConceptNetWordsDataset(
            max_words=200,  # Different max_words
            cache_dir=_DEFAULT_HF_LOADER_CACHE_DIR
        )
        
        # Both loaders should work with cached data
        words1 = loader1.load()
        words2 = loader2.load()
        
        assert len(words1) <= 100, "First loader should respect its max_words"
        assert len(words2) <= 200, "Second loader should respect its max_words"
        assert len(words2) >= len(words1), "Second loader should return more words"
        
        print(f"\nLoader1 returned {len(words1)} words")
        print(f"Loader2 returned {len(words2)} words")
        
        # Check for word overlap (they should be using the same base data)
        common_words = set(words1) & set(words2)
        print(f"Number of words in common: {len(common_words)}") 