"""
Unit tests for retriever module.
Tests intelligent retrieval orchestration and strategy selection.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from src.retrieval.retriever import (
    IntelligentRetriever,
    RetrievalConfig,
    RetrievalStrategy,
    RetrievalResult,
    create_intelligent_retriever
)
from src.retrieval.query_processor import ProcessedQuery, QueryType
from src.vectordb.search import SearchResponse, SearchResult, SearchMethod


class TestRetrievalConfig:
    """Test retrieval configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RetrievalConfig()
        
        assert config.strategy == RetrievalStrategy.ADAPTIVE
        assert config.max_results == 10
        assert config.min_similarity == 0.1
        assert config.enable_reranking is True
        assert config.enable_query_expansion is True
        assert config.adaptive_top_k is True
        assert config.fallback_enabled is True
        assert config.timeout_seconds == 30
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RetrievalConfig(
            strategy=RetrievalStrategy.SIMPLE,
            max_results=5,
            min_similarity=0.3,
            enable_reranking=False
        )
        
        assert config.strategy == RetrievalStrategy.SIMPLE
        assert config.max_results == 5
        assert config.min_similarity == 0.3
        assert config.enable_reranking is False


class TestIntelligentRetriever:
    """Test intelligent retriever functionality."""
    
    @pytest.fixture
    def mock_query_processor(self):
        """Create mock query processor."""
        processor = Mock()
        processor.process_query.return_value = ProcessedQuery(
            original="test query",
            processed="test query",
            query_type=QueryType.FACTUAL,
            keywords=["test", "query"],
            expanded_terms=["testing"],
            filters={"language": "hr"},
            confidence=0.8,
            metadata={}
        )
        processor.suggest_query_improvements.return_value = ["improvement 1", "improvement 2"]
        return processor
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create mock search engine."""
        engine = Mock()
        
        # Create sample search results
        sample_results = [
            SearchResult(
                content="Zagreb je glavni grad Hrvatske.",
                score=0.9,
                metadata={"source": "zagreb.txt", "title": "O Zagrebu"},
                id="doc1",
                rank=1
            ),
            SearchResult(
                content="Dubrovnik je biser Jadrana.",
                score=0.8,
                metadata={"source": "dubrovnik.txt", "title": "Dubrovnik"},
                id="doc2",
                rank=2
            )
        ]
        
        engine.search.return_value = SearchResponse(
            results=sample_results,
            query="test query",
            method=SearchMethod.SEMANTIC,
            total_time=0.1,
            total_results=2,
            metadata={}
        )
        
        return engine
    
    @pytest.fixture
    def retriever(self, mock_query_processor, mock_search_engine):
        """Create retriever with mocks."""
        config = RetrievalConfig(max_results=5)
        return IntelligentRetriever(mock_query_processor, mock_search_engine, config)
    
    def test_initialization(self, mock_query_processor, mock_search_engine):
        """Test retriever initialization."""
        config = RetrievalConfig()
        retriever = IntelligentRetriever(mock_query_processor, mock_search_engine, config)
        
        assert retriever.query_processor == mock_query_processor
        assert retriever.search_engine == mock_search_engine
        assert retriever.config == config
        assert retriever._query_stats['total_queries'] == 0


class TestRetrievalExecution:
    """Test retrieval execution and strategy selection."""
    
    @pytest.fixture
    def setup_retriever(self):
        """Set up retriever with detailed mocks."""
        # Mock query processor
        query_processor = Mock()
        
        # Mock search engine
        search_engine = Mock()
        
        # Create retriever
        config = RetrievalConfig(strategy=RetrievalStrategy.ADAPTIVE)
        retriever = IntelligentRetriever(query_processor, search_engine, config)
        
        return retriever, query_processor, search_engine
    
    def test_successful_retrieval(self, setup_retriever):
        """Test successful document retrieval."""
        retriever, query_processor, search_engine = setup_retriever
        
        # Setup mocks
        processed_query = ProcessedQuery(
            original="Koji je glavni grad Hrvatske?",
            processed="koji je glavni grad hrvatske",
            query_type=QueryType.FACTUAL,
            keywords=["glavni", "grad", "hrvatske"],
            expanded_terms=["glavnog", "grada"],
            filters={"language": "hr"},
            confidence=0.85,
            metadata={}
        )
        query_processor.process_query.return_value = processed_query
        
        search_results = [
            SearchResult(
                content="Zagreb je glavni i najveći grad Republike Hrvatske.",
                score=0.92,
                metadata={"source": "croatia.txt", "title": "Glavni gradovi"},
                id="doc1",
                rank=1
            )
        ]
        
        search_response = SearchResponse(
            results=search_results,
            query="glavni grad hrvatske",
            method=SearchMethod.SEMANTIC,
            total_time=0.15,
            total_results=1
        )
        search_engine.search.return_value = search_response
        
        # Execute retrieval
        result = retriever.retrieve("Koji je glavni grad Hrvatske?")
        
        # Verify results
        assert isinstance(result, RetrievalResult)
        assert result.query == "Koji je glavni grad Hrvatske?"
        assert result.result_count == 1
        assert len(result.documents) == 1
        assert result.confidence > 0.0
        assert result.strategy_used in RetrievalStrategy
        
        # Check document structure
        doc = result.documents[0]
        assert doc['content'] == search_results[0].content
        assert doc['relevance_score'] == search_results[0].score
        assert 'retrieval_metadata' in doc
    
    def test_adaptive_strategy_selection(self, setup_retriever):
        """Test adaptive strategy selection based on query type."""
        retriever, query_processor, search_engine = setup_retriever
        
        # Mock search engine
        search_engine.search.return_value = SearchResponse(
            results=[], query="", method=SearchMethod.SEMANTIC, total_time=0.1, total_results=0
        )
        
        test_cases = [
            (QueryType.FACTUAL, 0.8, RetrievalStrategy.SIMPLE),
            (QueryType.EXPLANATORY, 0.7, RetrievalStrategy.HYBRID),
            (QueryType.COMPARISON, 0.6, RetrievalStrategy.HYBRID),
            (QueryType.SUMMARIZATION, 0.8, RetrievalStrategy.MULTI_PASS),
            (QueryType.GENERAL, 0.5, RetrievalStrategy.SIMPLE),
            (QueryType.FACTUAL, 0.3, RetrievalStrategy.MULTI_PASS)  # Low confidence
        ]
        
        for query_type, confidence, expected_strategy in test_cases:
            processed_query = ProcessedQuery(
                original="test",
                processed="test",
                query_type=query_type,
                keywords=["test", "word"],
                expanded_terms=[],
                filters={"language": "hr"},
                confidence=confidence,
                metadata={}
            )
            query_processor.process_query.return_value = processed_query
            
            chosen_strategy = retriever._choose_adaptive_strategy(processed_query)
            assert chosen_strategy == expected_strategy, \
                f"Wrong strategy for {query_type} with confidence {confidence}"
    
    def test_low_confidence_query_handling(self, setup_retriever):
        """Test handling of low confidence queries."""
        retriever, query_processor, search_engine = setup_retriever
        
        # Setup low confidence query
        low_confidence_query = ProcessedQuery(
            original="xyz",
            processed="xyz",
            query_type=QueryType.GENERAL,
            keywords=[],
            expanded_terms=[],
            filters={"language": "hr"},
            confidence=0.05,  # Very low confidence
            metadata={}
        )
        query_processor.process_query.return_value = low_confidence_query
        query_processor.suggest_query_improvements.return_value = ["Add more keywords"]
        
        # Execute retrieval
        result = retriever.retrieve("xyz")
        
        # Should return low confidence result
        assert result.result_count == 0
        assert result.confidence == 0.05
        assert 'suggestions' in result.metadata
        assert len(result.metadata['suggestions']) > 0
    
    def test_error_handling_with_fallback(self, setup_retriever):
        """Test error handling and fallback mechanisms."""
        retriever, query_processor, search_engine = setup_retriever
        
        # Setup successful query processing
        processed_query = ProcessedQuery(
            original="test query",
            processed="test query", 
            query_type=QueryType.FACTUAL,
            keywords=["test"],
            expanded_terms=[],
            filters={"language": "hr"},
            confidence=0.7,
            metadata={}
        )
        query_processor.process_query.return_value = processed_query
        
        # Setup search engine to fail first, then succeed on fallback
        def side_effect(*args, **kwargs):
            # First call fails
            if search_engine.search.call_count == 1:
                raise Exception("Search failed")
            # Second call (fallback) succeeds
            return SearchResponse(
                results=[],
                query="test query",
                method=SearchMethod.SEMANTIC,
                total_time=0.1,
                total_results=0
            )
        
        search_engine.search.side_effect = side_effect
        
        # Execute retrieval
        result = retriever.retrieve("test query")
        
        # Should succeed with fallback
        assert isinstance(result, RetrievalResult)
        assert search_engine.search.call_count == 2  # Original + fallback
    
    def test_performance_stats_tracking(self, setup_retriever):
        """Test performance statistics tracking."""
        retriever, query_processor, search_engine = setup_retriever
        
        # Setup successful query
        query_processor.process_query.return_value = ProcessedQuery(
            original="test", processed="test", query_type=QueryType.GENERAL,
            keywords=["test"], expanded_terms=[], filters={"language": "hr"},
            confidence=0.7, metadata={}
        )
        
        search_engine.search.return_value = SearchResponse(
            results=[], query="test", method=SearchMethod.SEMANTIC, 
            total_time=0.1, total_results=0
        )
        
        # Execute multiple retrievals
        retriever.retrieve("query 1")
        retriever.retrieve("query 2") 
        retriever.retrieve("query 3")
        
        # Check statistics
        stats = retriever.get_performance_stats()
        assert stats['total_queries'] == 3
        assert stats['success_rate'] > 0.0
        assert stats['average_time'] > 0.0
        assert 'strategy_usage' in stats


class TestSearchStrategies:
    """Test different search strategy implementations."""
    
    @pytest.fixture
    def retriever_with_mocks(self):
        """Create retriever with detailed mocks for strategy testing."""
        query_processor = Mock()
        search_engine = Mock()
        config = RetrievalConfig()
        
        retriever = IntelligentRetriever(query_processor, search_engine, config)
        
        # Mock processed query
        processed_query = ProcessedQuery(
            original="test query",
            processed="test query",
            query_type=QueryType.FACTUAL,
            keywords=["test", "query"],
            expanded_terms=["testing"],
            filters={"language": "hr"},
            confidence=0.8,
            metadata={}
        )
        
        return retriever, search_engine, processed_query
    
    def test_simple_search_strategy(self, retriever_with_mocks):
        """Test simple search strategy."""
        retriever, search_engine, processed_query = retriever_with_mocks
        
        # Setup mock response
        mock_response = SearchResponse(
            results=[SearchResult("content", 0.8, {}, "id1", 1)],
            query="test query",
            method=SearchMethod.SEMANTIC,
            total_time=0.1,
            total_results=1
        )
        search_engine.search.return_value = mock_response
        
        # Execute simple search
        response = retriever._simple_search(processed_query, {})
        
        # Verify search was called with correct parameters
        search_engine.search.assert_called_once()
        call_args = search_engine.search.call_args
        
        assert call_args.kwargs['query'] == processed_query.processed
        assert call_args.kwargs['filters'] == processed_query.filters
        assert call_args.kwargs['method'] == SearchMethod.SEMANTIC
        assert response == mock_response
    
    def test_hybrid_search_strategy(self, retriever_with_mocks):
        """Test hybrid search strategy."""
        retriever, search_engine, processed_query = retriever_with_mocks
        
        # Setup mock response
        mock_response = SearchResponse(
            results=[SearchResult("content", 0.8, {}, "id1", 1)],
            query="test query",
            method=SearchMethod.HYBRID,
            total_time=0.2,
            total_results=1
        )
        search_engine.search.return_value = mock_response
        
        # Execute hybrid search
        response = retriever._hybrid_search(processed_query, {})
        
        # Verify hybrid method was used
        search_engine.search.assert_called_once()
        call_args = search_engine.search.call_args
        assert call_args.kwargs['method'] == SearchMethod.HYBRID
        assert response == mock_response
    
    def test_multi_pass_search_strategy(self, retriever_with_mocks):
        """Test multi-pass search strategy."""
        retriever, search_engine, processed_query = retriever_with_mocks
        
        # Setup mock responses for different passes
        semantic_results = [SearchResult("semantic content", 0.9, {}, "sem1", 1)]
        keyword_results = [SearchResult("keyword content", 0.7, {}, "key1", 1)]
        expanded_results = [SearchResult("expanded content", 0.6, {}, "exp1", 1)]
        
        def search_side_effect(*args, **kwargs):
            method = kwargs.get('method')
            if method == SearchMethod.SEMANTIC:
                if 'testing' in kwargs.get('query', ''):  # Expanded terms query
                    return SearchResponse(expanded_results, "", method, 0.1, 1)
                else:  # Original semantic query
                    return SearchResponse(semantic_results, "", method, 0.1, 1)
            elif method == SearchMethod.KEYWORD:
                return SearchResponse(keyword_results, "", method, 0.1, 1)
            return SearchResponse([], "", method, 0.1, 0)
        
        search_engine.search.side_effect = search_side_effect
        
        # Execute multi-pass search
        response = retriever._multi_pass_search(processed_query, {})
        
        # Should have made multiple search calls
        assert search_engine.search.call_count >= 2
        
        # Should combine results from different passes
        assert len(response.results) <= 3  # Max one from each pass
        assert response.method == SearchMethod.HYBRID  # Multi-pass is hybrid
        
        # Results should be sorted by score
        if len(response.results) > 1:
            scores = [result.score for result in response.results]
            assert scores == sorted(scores, reverse=True)


class TestResultPostProcessing:
    """Test result post-processing functionality."""
    
    @pytest.fixture
    def sample_search_response(self):
        """Create sample search response for testing."""
        results = [
            SearchResult(
                content="Zagreb je glavni grad Hrvatske s 800.000 stanovnika.",
                score=0.9,
                metadata={"source": "zagreb.txt", "title": "Zagreb Info"},
                id="doc1",
                rank=1
            ),
            SearchResult(
                content="Split je drugi najveći grad u Hrvatskoj.",
                score=0.7,
                metadata={"source": "split.txt", "city": "Split"},
                id="doc2", 
                rank=2
            ),
            SearchResult(
                content="Mala bilježka o gradu.",
                score=0.05,  # Below threshold
                metadata={"source": "note.txt"},
                id="doc3",
                rank=3
            )
        ]
        
        return SearchResponse(
            results=results,
            query="hrvatski gradovi",
            method=SearchMethod.SEMANTIC,
            total_time=0.2,
            total_results=3
        )
    
    def test_similarity_threshold_filtering(self):
        """Test filtering by similarity threshold."""
        # Create retriever with higher threshold
        config = RetrievalConfig(min_similarity=0.6)
        query_processor = Mock()
        search_engine = Mock()
        
        retriever = IntelligentRetriever(query_processor, search_engine, config)
        
        processed_query = ProcessedQuery(
            original="test", processed="test", query_type=QueryType.GENERAL,
            keywords=["test"], expanded_terms=[], filters={"language": "hr"},
            confidence=0.8, metadata={}
        )
        
        # Create search response with mixed scores
        search_response = SearchResponse(
            results=[
                SearchResult("high score", 0.8, {}, "high", 1),
                SearchResult("medium score", 0.5, {}, "medium", 2),  # Below threshold
                SearchResult("low score", 0.2, {}, "low", 3)  # Below threshold
            ],
            query="test",
            method=SearchMethod.SEMANTIC,
            total_time=0.1,
            total_results=3
        )
        
        # Post-process results
        documents = retriever._post_process_results(search_response, processed_query)
        
        # Only high score result should remain
        assert len(documents) == 1
        assert documents[0]['relevance_score'] == 0.8
        assert documents[0]['id'] == "high"
    
    def test_keyword_matching_analysis(self, sample_search_response):
        """Test keyword matching analysis in post-processing."""
        query_processor = Mock()
        search_engine = Mock() 
        retriever = IntelligentRetriever(query_processor, search_engine)
        
        processed_query = ProcessedQuery(
            original="glavni grad",
            processed="glavni grad", 
            query_type=QueryType.FACTUAL,
            keywords=["glavni", "grad"],
            expanded_terms=[],
            filters={"language": "hr"},
            confidence=0.8,
            metadata={}
        )
        
        documents = retriever._post_process_results(sample_search_response, processed_query)
        
        # Check keyword matching analysis
        for doc in documents:
            assert 'retrieval_metadata' in doc
            assert 'matching_keywords' in doc['retrieval_metadata']
            
            # First document should match both keywords
            if doc['id'] == 'doc1':
                matching = doc['retrieval_metadata']['matching_keywords']
                assert 'glavni' in matching
                assert 'grad' in matching
    
    def test_retrieval_metadata_enrichment(self, sample_search_response):
        """Test retrieval metadata enrichment."""
        query_processor = Mock()
        search_engine = Mock()
        retriever = IntelligentRetriever(query_processor, search_engine)
        
        processed_query = ProcessedQuery(
            original="test", processed="test", query_type=QueryType.EXPLANATORY,
            keywords=["test"], expanded_terms=[], filters={"language": "hr"},
            confidence=0.8, metadata={}
        )
        
        documents = retriever._post_process_results(sample_search_response, processed_query)
        
        for doc in documents:
            metadata = doc['retrieval_metadata']
            
            # Check required metadata fields
            assert 'query_type' in metadata
            assert metadata['query_type'] == 'explanatory'
            assert 'content_length' in metadata
            assert 'has_title' in metadata
            assert 'source' in metadata
            
            # Verify metadata accuracy
            assert metadata['content_length'] == len(doc['content'])
            assert isinstance(metadata['has_title'], bool)


class TestConfidenceCalculation:
    """Test confidence calculation functionality."""
    
    @pytest.fixture
    def retriever(self):
        """Create retriever for confidence testing."""
        return IntelligentRetriever(Mock(), Mock())
    
    def test_confidence_with_good_results(self, retriever):
        """Test confidence calculation with good results."""
        processed_query = ProcessedQuery(
            original="test", processed="test", query_type=QueryType.FACTUAL,
            keywords=["test", "query"], expanded_terms=[], filters={},
            confidence=0.8, metadata={}
        )
        
        documents = [
            {
                'relevance_score': 0.9,
                'retrieval_metadata': {
                    'matching_keywords': ['test', 'query'],
                    'source': 'source1.txt'
                }
            },
            {
                'relevance_score': 0.8,
                'retrieval_metadata': {
                    'matching_keywords': ['test'],
                    'source': 'source2.txt'
                }
            }
        ]
        
        confidence = retriever._calculate_retrieval_confidence(
            processed_query, None, documents
        )
        
        # Should have high confidence with good results
        assert confidence > 0.7
        assert confidence <= 1.0
    
    def test_confidence_with_poor_results(self, retriever):
        """Test confidence calculation with poor results."""
        processed_query = ProcessedQuery(
            original="test", processed="test", query_type=QueryType.GENERAL,
            keywords=["test"], expanded_terms=[], filters={},
            confidence=0.3, metadata={}  # Low initial confidence
        )
        
        documents = [
            {
                'relevance_score': 0.2,  # Low relevance
                'retrieval_metadata': {
                    'matching_keywords': [],  # No keyword matches
                    'source': 'source1.txt'
                }
            }
        ]
        
        confidence = retriever._calculate_retrieval_confidence(
            processed_query, None, documents
        )
        
        # Should have low confidence
        assert confidence < 0.5
    
    def test_confidence_with_no_results(self, retriever):
        """Test confidence calculation with no results."""
        processed_query = ProcessedQuery(
            original="test", processed="test", query_type=QueryType.GENERAL,
            keywords=["test"], expanded_terms=[], filters={},
            confidence=0.6, metadata={}
        )
        
        documents = []
        
        confidence = retriever._calculate_retrieval_confidence(
            processed_query, None, documents
        )
        
        # Should be zero confidence with no results
        assert confidence == 0.0


class TestFactoryFunction:
    """Test factory function."""
    
    @patch('src.retrieval.retriever.create_query_processor')
    @patch('src.retrieval.retriever.create_search_engine')
    def test_create_intelligent_retriever(self, mock_create_search, mock_create_query):
        """Test intelligent retriever factory function."""
        # Mock components
        mock_embedding_model = Mock()
        mock_storage = Mock()
        
        mock_query_processor = Mock()
        mock_search_engine = Mock()
        
        mock_create_query.return_value = mock_query_processor
        mock_create_search.return_value = mock_search_engine
        
        # Create retriever
        retriever = create_intelligent_retriever(
            mock_embedding_model, 
            mock_storage,
            RetrievalStrategy.HYBRID
        )
        
        # Verify factory calls
        mock_create_query.assert_called_once_with(language="hr")
        mock_create_search.assert_called_once_with(mock_embedding_model, mock_storage)
        
        # Verify retriever configuration
        assert isinstance(retriever, IntelligentRetriever)
        assert retriever.config.strategy == RetrievalStrategy.HYBRID


if __name__ == "__main__":
    pytest.main([__file__])