"""
Unit tests for search module.
Tests similarity search functionality with Croatian language support.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from src.vectordb.search import (
    SemanticSearchEngine,
    SearchConfig,
    SearchResult,
    SearchResponse,
    SearchMethod,
    SearchResultFormatter,
    create_search_engine
)


class TestSearchConfig:
    """Test search configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SearchConfig()
        
        assert config.method == SearchMethod.SEMANTIC
        assert config.top_k == 5
        assert config.similarity_threshold == 0.0
        assert config.max_context_length == 2000
        assert config.rerank is True
        assert config.include_metadata is True
        assert config.include_distances is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SearchConfig(
            method=SearchMethod.HYBRID,
            top_k=10,
            similarity_threshold=0.5,
            max_context_length=1000,
            rerank=False
        )
        
        assert config.method == SearchMethod.HYBRID
        assert config.top_k == 10
        assert config.similarity_threshold == 0.5
        assert config.max_context_length == 1000
        assert config.rerank is False


class TestSearchResult:
    """Test search result structure."""
    
    def test_search_result_creation(self):
        """Test creating search result."""
        result = SearchResult(
            content="Zagreb je glavni grad Hrvatske.",
            score=0.85,
            metadata={"source": "zagreb.txt", "language": "hr"},
            id="doc_123",
            rank=1
        )
        
        assert result.content == "Zagreb je glavni grad Hrvatske."
        assert result.score == 0.85
        assert result.metadata["source"] == "zagreb.txt"
        assert result.id == "doc_123"
        assert result.rank == 1
    
    def test_search_result_to_dict(self):
        """Test converting search result to dictionary."""
        result = SearchResult(
            content="Test content",
            score=0.9,
            metadata={"test": "meta"},
            id="test_id"
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["content"] == "Test content"
        assert result_dict["score"] == 0.9
        assert result_dict["metadata"] == {"test": "meta"}
        assert result_dict["id"] == "test_id"


class TestSearchResponse:
    """Test search response structure."""
    
    def test_search_response_creation(self):
        """Test creating search response."""
        results = [
            SearchResult("Content 1", 0.9, {}, "id1", 1),
            SearchResult("Content 2", 0.8, {}, "id2", 2)
        ]
        
        response = SearchResponse(
            results=results,
            query="test query",
            method=SearchMethod.SEMANTIC,
            total_time=0.123,
            total_results=2
        )
        
        assert len(response.results) == 2
        assert response.query == "test query"
        assert response.method == SearchMethod.SEMANTIC
        assert response.total_time == 0.123
        assert response.total_results == 2
        assert response.metadata is not None  # Should be initialized


class TestSemanticSearchEngine:
    """Test semantic search engine functionality."""
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create mock embedding model."""
        mock_model = Mock()
        mock_model.encode_text.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        return mock_model
    
    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        mock_storage = Mock()
        mock_storage.collection = Mock()
        mock_storage.create_collection.return_value = Mock()
        return mock_storage
    
    @pytest.fixture
    def search_engine(self, mock_embedding_model, mock_storage):
        """Create search engine with mocks."""
        config = SearchConfig(top_k=3, rerank=False)  # Disable rerank for simpler testing
        return SemanticSearchEngine(mock_embedding_model, mock_storage, config)
    
    def test_initialization(self, mock_embedding_model, mock_storage):
        """Test search engine initialization."""
        engine = SemanticSearchEngine(mock_embedding_model, mock_storage)
        
        assert engine.embedding_model == mock_embedding_model
        assert engine.storage == mock_storage
        assert engine.config.method == SearchMethod.SEMANTIC
        mock_storage.create_collection.assert_called_once()


class TestSemanticSearch:
    """Test semantic search functionality."""
    
    @pytest.fixture
    def search_setup(self):
        """Set up search engine with Croatian test data."""
        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.encode_text.return_value = np.array([0.1, 0.2, 0.3])
        
        # Mock storage
        mock_storage = Mock()
        mock_storage.collection = Mock()
        
        # Mock ChromaDB query results
        mock_storage.query_similar.return_value = {
            "ids": [["doc1", "doc2", "doc3"]],
            "documents": [[
                "Zagreb je glavni grad Republike Hrvatske.",
                "Dubrovnik je poznat kao biser Jadrana.",
                "Plitvička jezera su nacionalni park."
            ]],
            "metadatas": [[
                {"source": "zagreb.txt", "language": "hr", "region": "Središte"},
                {"source": "dubrovnik.txt", "language": "hr", "region": "Dalmacija"}, 
                {"source": "plitvice.txt", "language": "hr", "region": "Lika"}
            ]],
            "distances": [[0.1, 0.3, 0.5]]
        }
        
        config = SearchConfig(top_k=3, rerank=False)
        engine = SemanticSearchEngine(mock_embedding_model, mock_storage, config)
        
        return engine, mock_embedding_model, mock_storage
    
    def test_semantic_search_croatian_query(self, search_setup):
        """Test semantic search with Croatian query."""
        engine, mock_embedding_model, mock_storage = search_setup
        
        query = "Koji je glavni grad Hrvatske?"
        response = engine.search(query, method=SearchMethod.SEMANTIC)
        
        # Verify embedding model was called
        mock_embedding_model.encode_text.assert_called_with(query)
        
        # Verify storage query was called
        mock_storage.query_similar.assert_called_once()
        call_kwargs = mock_storage.query_similar.call_args.kwargs
        assert "query_embeddings" in call_kwargs
        assert call_kwargs["n_results"] == 6  # top_k * 2 for filtering
        
        # Verify results
        assert isinstance(response, SearchResponse)
        assert response.query == query
        assert response.method == SearchMethod.SEMANTIC
        assert len(response.results) == 3
        
        # Check Croatian content is preserved
        zagreb_result = response.results[0]  # Should be most similar (distance 0.1)
        assert "Zagreb" in zagreb_result.content
        assert "Hrvatske" in zagreb_result.content
        assert zagreb_result.metadata["language"] == "hr"
    
    def test_semantic_search_with_filters(self, search_setup):
        """Test semantic search with metadata filters."""
        engine, mock_embedding_model, mock_storage = search_setup
        
        query = "Dalmatian coast"
        filters = {"region": "Dalmacija"}
        
        response = engine.search(query, filters=filters)
        
        # Verify filters were passed to storage
        call_kwargs = mock_storage.query_similar.call_args.kwargs
        assert call_kwargs["where"] == filters
    
    def test_distance_to_similarity_conversion(self, search_setup):
        """Test conversion of ChromaDB distance to similarity score."""
        engine, _, _ = search_setup
        
        # Test various distance values
        assert engine._distance_to_similarity(0.0) == 1.0  # Perfect similarity
        assert engine._distance_to_similarity(1.0) == 0.0  # No similarity
        assert engine._distance_to_similarity(0.5) == 0.5  # Medium similarity
        
        # Test boundary conditions
        assert engine._distance_to_similarity(-0.1) == 1.0  # Clamp to max
        assert engine._distance_to_similarity(1.5) == 0.0   # Clamp to min


class TestKeywordSearch:
    """Test keyword search functionality."""
    
    @pytest.fixture
    def keyword_search_setup(self):
        """Set up search engine for keyword testing."""
        mock_embedding_model = Mock()
        mock_storage = Mock()
        mock_storage.collection = Mock()
        
        # Mock keyword search results
        mock_storage.query_similar.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [[
                "Zagreb je glavni grad Hrvatske i najveći grad u zemlji.",
                "Split je veliki grad na Jadranu i turistički centar."
            ]],
            "metadatas": [[
                {"source": "zagreb.txt", "title": "O Zagrebu"},
                {"source": "split.txt", "title": "O Splitu"}
            ]]
        }
        
        config = SearchConfig(method=SearchMethod.KEYWORD, rerank=False)
        engine = SemanticSearchEngine(mock_embedding_model, mock_storage, config)
        
        return engine, mock_storage
    
    def test_keyword_search_croatian(self, keyword_search_setup):
        """Test keyword search with Croatian terms."""
        engine, mock_storage = keyword_search_setup
        
        query = "glavni grad"
        response = engine.search(query, method=SearchMethod.KEYWORD)
        
        # Verify document filter was used for keyword matching
        call_kwargs = mock_storage.query_similar.call_args.kwargs
        assert "where_document" in call_kwargs
        assert call_kwargs["where_document"]["$contains"] == "glavni"
        
        # Verify results
        assert len(response.results) == 2
        assert response.method == SearchMethod.KEYWORD
    
    def test_keyword_score_calculation(self, keyword_search_setup):
        """Test keyword-based scoring."""
        engine, _ = keyword_search_setup
        
        query_terms = ["glavni", "grad"]
        
        # Document with both terms
        doc_high_score = "zagreb je glavni grad hrvatske"
        score_high = engine._calculate_keyword_score(query_terms, doc_high_score)
        
        # Document with one term
        doc_low_score = "split je veliki grad"
        score_low = engine._calculate_keyword_score(query_terms, doc_low_score)
        
        # Document with no terms
        doc_no_score = "dubrovnik je lijep"
        score_none = engine._calculate_keyword_score(query_terms, doc_no_score)
        
        assert score_high > score_low > score_none
        assert 0 <= score_none <= score_low <= score_high <= 1
    
    def test_exact_phrase_boost(self, keyword_search_setup):
        """Test that exact phrase matches get score boost."""
        engine, _ = keyword_search_setup
        
        query_terms = ["glavni", "grad"]
        
        # Document with exact phrase
        doc_exact = "glavni grad hrvatske"
        score_exact = engine._calculate_keyword_score(query_terms, doc_exact)
        
        # Document with separate terms
        doc_separate = "grad je glavni u hrvatskoj"
        score_separate = engine._calculate_keyword_score(query_terms, doc_separate)
        
        assert score_exact > score_separate


class TestHybridSearch:
    """Test hybrid search combining semantic and keyword methods."""
    
    @pytest.fixture
    def hybrid_setup(self):
        """Set up hybrid search engine."""
        mock_embedding_model = Mock()
        mock_embedding_model.encode_text.return_value = np.array([0.1, 0.2, 0.3])
        
        mock_storage = Mock()
        mock_storage.collection = Mock()
        mock_storage.query_similar.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Content 1", "Content 2"]],
            "metadatas": [[{"source": "1.txt"}, {"source": "2.txt"}]],
            "distances": [[0.2, 0.4]]
        }
        
        config = SearchConfig(method=SearchMethod.HYBRID, rerank=False)
        engine = SemanticSearchEngine(mock_embedding_model, mock_storage, config)
        
        return engine, mock_storage
    
    def test_hybrid_search_combines_methods(self, hybrid_setup):
        """Test that hybrid search combines semantic and keyword results."""
        engine, mock_storage = hybrid_setup
        
        query = "test query"
        response = engine.search(query, method=SearchMethod.HYBRID)
        
        # Should call query_similar multiple times (semantic + keyword)
        assert mock_storage.query_similar.call_count >= 2
        
        assert response.method == SearchMethod.HYBRID
        assert len(response.results) >= 0


class TestSearchResultReranking:
    """Test result reranking functionality."""
    
    @pytest.fixture
    def rerank_engine(self):
        """Create search engine with reranking enabled."""
        mock_embedding_model = Mock()
        mock_storage = Mock()
        mock_storage.collection = Mock()
        
        config = SearchConfig(rerank=True, similarity_threshold=0.0)
        engine = SemanticSearchEngine(mock_embedding_model, mock_storage, config)
        
        return engine
    
    def test_reranking_boosts_term_overlap(self, rerank_engine):
        """Test that reranking boosts results with term overlap."""
        query = "Zagreb glavni grad"
        
        results = [
            SearchResult(
                content="Zagreb je glavni grad Hrvatske.",  # High overlap
                score=0.7,
                metadata={"title": "O Zagrebu"},
                id="doc1"
            ),
            SearchResult(
                content="Dubrovnik je biser Jadrana.",  # No overlap
                score=0.8,  # Initially higher score
                metadata={},
                id="doc2"
            )
        ]
        
        reranked = rerank_engine._rerank_results(query, results)
        
        # Result with term overlap should be ranked higher after reranking
        # (even if it had lower initial score)
        assert reranked[0].id == "doc1"  # Zagreb doc should be first
        assert reranked[0].score > 0.7   # Score should be boosted
    
    def test_reranking_metadata_boost(self, rerank_engine):
        """Test that results with titles get metadata boost."""
        query = "test"
        
        results = [
            SearchResult("Content without title", 0.8, {}, "doc1"),
            SearchResult("Content with title", 0.8, {"title": "Test Title"}, "doc2")
        ]
        
        reranked = rerank_engine._rerank_results(query, results)
        
        # Result with title should have higher score after reranking
        doc_with_title = next(r for r in reranked if r.id == "doc2")
        doc_without_title = next(r for r in reranked if r.id == "doc1")
        
        assert doc_with_title.score > doc_without_title.score


class TestSimilarityThresholdFiltering:
    """Test filtering results by similarity threshold."""
    
    def test_filter_by_threshold(self):
        """Test filtering results below similarity threshold."""
        config = SearchConfig(similarity_threshold=0.5)
        mock_embedding_model = Mock()
        mock_storage = Mock()
        mock_storage.collection = Mock()
        
        engine = SemanticSearchEngine(mock_embedding_model, mock_storage, config)
        
        results = [
            SearchResult("High score", 0.8, {}, "doc1"),
            SearchResult("Medium score", 0.5, {}, "doc2"), 
            SearchResult("Low score", 0.3, {}, "doc3")
        ]
        
        filtered = engine._filter_by_threshold(results)
        
        assert len(filtered) == 2  # Only scores >= 0.5
        assert all(r.score >= 0.5 for r in filtered)


class TestSearchResultFormatter:
    """Test search result formatting."""
    
    def test_format_for_display(self):
        """Test formatting search response for display."""
        results = [
            SearchResult(
                content="Zagreb je glavni grad Republike Hrvatske.",
                score=0.9,
                metadata={"title": "O Zagrebu", "source": "zagreb.txt"},
                id="doc1",
                rank=1
            ),
            SearchResult(
                content="Dubrovnik je poznat kao biser Jadrana.",
                score=0.8,
                metadata={"source": "dubrovnik.txt"},
                id="doc2", 
                rank=2
            )
        ]
        
        response = SearchResponse(
            results=results,
            query="hrvatski gradovi",
            method=SearchMethod.SEMANTIC,
            total_time=0.123,
            total_results=2
        )
        
        formatted = SearchResultFormatter.format_for_display(response)
        
        assert "Search Results for: 'hrvatski gradovi'" in formatted
        assert "Method: semantic" in formatted
        assert "Time: 0.123s" in formatted
        assert "Found 2 results" in formatted
        assert "#1 (Score: 0.900)" in formatted
        assert "#2 (Score: 0.800)" in formatted
        assert "Zagreb" in formatted
        assert "O Zagrebu" in formatted  # Title should be shown
    
    def test_extract_context_chunks(self):
        """Test extracting context chunks for RAG."""
        results = [
            SearchResult("Short content", 0.9, {}, "doc1", 1),
            SearchResult("Medium length content with more text", 0.8, {}, "doc2", 2),
            SearchResult("Very long content " + "a" * 2000, 0.7, {}, "doc3", 3)
        ]
        
        response = SearchResponse(
            results=results,
            query="test",
            method=SearchMethod.SEMANTIC,
            total_time=0.1,
            total_results=3
        )
        
        # Test normal extraction
        chunks = SearchResultFormatter.extract_context_chunks(response, max_length=100)
        
        assert len(chunks) >= 1
        assert sum(len(chunk) for chunk in chunks) <= 100
        
        # Test with very small limit (should include at least one chunk)
        chunks_small = SearchResultFormatter.extract_context_chunks(response, max_length=10)
        assert len(chunks_small) >= 1


class TestSimilarDocuments:
    """Test finding similar documents functionality."""
    
    def test_get_similar_documents(self):
        """Test finding documents similar to a given document."""
        mock_embedding_model = Mock()
        mock_storage = Mock()
        mock_storage.collection = Mock()
        
        # Mock get_documents call
        mock_storage.get_documents.return_value = {
            "documents": ["Zagreb je glavni grad Hrvatske."]
        }
        
        engine = SemanticSearchEngine(mock_embedding_model, mock_storage)
        
        # Mock the search method
        mock_results = [
            SearchResult("Similar doc 1", 0.8, {}, "similar1", 1),
            SearchResult("Similar doc 2", 0.7, {}, "similar2", 2)
        ]
        
        with patch.object(engine, 'search') as mock_search:
            mock_search.return_value = SearchResponse(
                results=[
                    SearchResult("Original doc", 1.0, {}, "original", 1),  # Self
                    *mock_results
                ],
                query="Zagreb je glavni grad Hrvatske.",
                method=SearchMethod.SEMANTIC,
                total_time=0.1,
                total_results=3
            )
            
            similar_docs = engine.get_similar_documents("original_doc_id", top_k=2)
            
            assert len(similar_docs) == 2  # Should exclude self
            assert similar_docs[0].id == "similar1"
            assert similar_docs[1].id == "similar2"


class TestFactoryFunctions:
    """Test factory and utility functions."""
    
    def test_create_search_engine(self):
        """Test search engine factory function."""
        mock_embedding_model = Mock()
        mock_storage = Mock()
        mock_storage.collection = Mock()
        
        engine = create_search_engine(
            embedding_model=mock_embedding_model,
            storage=mock_storage,
            method=SearchMethod.HYBRID,
            top_k=10
        )
        
        assert isinstance(engine, SemanticSearchEngine)
        assert engine.embedding_model == mock_embedding_model
        assert engine.storage == mock_storage
        assert engine.config.method == SearchMethod.HYBRID
        assert engine.config.top_k == 10


class TestErrorHandling:
    """Test error handling in search operations."""
    
    def test_search_with_no_collection(self):
        """Test error when storage has no collection."""
        mock_embedding_model = Mock()
        mock_storage = Mock()
        mock_storage.collection = None
        
        engine = SemanticSearchEngine(mock_embedding_model, mock_storage)
        engine.storage = mock_storage  # Override the collection creation
        
        with pytest.raises(Exception):  # Should raise some error
            engine.search("test query")
    
    def test_empty_query_handling(self):
        """Test handling of empty queries."""
        mock_embedding_model = Mock()
        mock_storage = Mock()
        mock_storage.collection = Mock()
        mock_storage.query_similar.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        engine = SemanticSearchEngine(mock_embedding_model, mock_storage)
        
        response = engine.search("")
        
        assert isinstance(response, SearchResponse)
        assert response.query == ""
        assert len(response.results) == 0


if __name__ == "__main__":
    pytest.main([__file__])