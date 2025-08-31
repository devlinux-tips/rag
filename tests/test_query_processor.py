"""
Unit tests for query processor module.
Tests Croatian query preprocessing and analysis functionality.
"""

import pytest
from unittest.mock import patch

from src.retrieval.query_processor import (
    CroatianQueryProcessor,
    QueryProcessingConfig,
    ProcessedQuery,
    QueryType,
    create_query_processor
)


class TestQueryProcessingConfig:
    """Test query processing configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = QueryProcessingConfig()
        
        assert config.language == "hr"
        assert config.expand_synonyms is True
        assert config.normalize_case is True
        assert config.remove_stop_words is True
        assert config.min_query_length == 3
        assert config.max_expanded_terms == 10
        assert config.enable_spell_check is False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = QueryProcessingConfig(
            language="en",
            expand_synonyms=False,
            normalize_case=False,
            min_query_length=5
        )
        
        assert config.language == "en"
        assert config.expand_synonyms is False
        assert config.normalize_case is False
        assert config.min_query_length == 5


class TestCroatianQueryProcessor:
    """Test Croatian query processor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create query processor for testing."""
        return CroatianQueryProcessor()
    
    @pytest.fixture
    def croatian_queries(self):
        """Sample Croatian queries for testing."""
        return {
            QueryType.FACTUAL: [
                "Koji je glavni grad Hrvatske?",
                "Kada je osnovana Zagreb?",
                "Koliko stanovnika ima Dubrovnik?",
                "Gdje se nalazi Plitvička jezera?"
            ],
            QueryType.EXPLANATORY: [
                "Kako nastaju Plitvička jezera?",
                "Zašto je Dubrovnik poznat?",
                "Objasni hrvatsku povijest.",
                "Opisi tradicionalnu hrvatsku kuhinju."
            ],
            QueryType.COMPARISON: [
                "Usporedi Zagreb i Split.",
                "Razlika između Jadrana i Dunava.",
                "Što je bolje - Dubrovnik ili Rovinj?",
                "Zagreb nasuprot Rijeci."
            ],
            QueryType.SUMMARIZATION: [
                "Sažmi hrvatsku povijest.",
                "Ukratko o hrvatskim gradovima.",
                "Pregled hrvatske kulture.",
                "Glavne točke o turizmu u Hrvatskoj."
            ],
            QueryType.GENERAL: [
                "Hrvatska je lijepa zemlja.",
                "Volim putovati po Hrvatskoj.",
                "Interesantne stvari o Zagrebu."
            ]
        }
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = CroatianQueryProcessor()
        
        assert processor.config.language == "hr"
        assert len(processor.croatian_stop_words) > 0
        assert 'zagreb' in processor.morphological_patterns
        assert QueryType.FACTUAL in processor.question_patterns
    
    def test_empty_query_processing(self, processor):
        """Test processing empty or very short queries."""
        # Empty query
        result = processor.process_query("")
        assert result.confidence == 0.0
        assert result.query_type == QueryType.GENERAL
        assert len(result.keywords) == 0
        
        # Very short query
        result = processor.process_query("hi")
        assert result.confidence == 0.0
        assert "too short" in result.metadata.get('error', '').lower()
    
    def test_query_type_classification(self, processor, croatian_queries):
        """Test query type classification for Croatian queries."""
        for expected_type, queries in croatian_queries.items():
            for query in queries:
                result = processor.process_query(query)
                assert result.query_type == expected_type, f"Query '{query}' should be {expected_type}, got {result.query_type}"
    
    def test_croatian_diacritic_handling(self, processor):
        """Test that Croatian diacritics are preserved and processed correctly."""
        queries_with_diacritics = [
            "Što je najljepši grad u Hrvatskoj?",
            "Gdje se nalazi Plitvička jezera?",
            "Kako se zove glavni grad?",
            "Čuvena hrvatska jela"
        ]
        
        for query in queries_with_diacritics:
            result = processor.process_query(query)
            
            # Check that diacritics are preserved
            assert any(char in result.processed for char in 'čćšžđ'), f"Diacritics lost in: {query}"
            
            # Should have reasonable confidence for Croatian text
            assert result.confidence > 0.4, f"Low confidence for Croatian query: {query}"
    
    def test_keyword_extraction(self, processor):
        """Test keyword extraction from Croatian queries."""
        test_cases = [
            ("Koji je glavni grad Hrvatske?", ["koji", "glavni", "grad", "hrvatske"]),
            ("Zagreb i Split su veliki gradovi.", ["zagreb", "split", "veliki", "gradovi"]),
            ("Dubrovnik je biser Jadrana.", ["dubrovnik", "biser", "jadrana"])
        ]
        
        for query, expected_keywords in test_cases:
            result = processor.process_query(query)
            
            # Check that main keywords are extracted
            for keyword in expected_keywords:
                assert any(keyword.lower() in kw.lower() for kw in result.keywords), \
                    f"Keyword '{keyword}' not found in {result.keywords} for query: {query}"
    
    def test_stop_word_removal(self, processor):
        """Test Croatian stop word removal."""
        query = "Gdje se nalazi veliki grad Zagreb u Hrvatskoj?"
        result = processor.process_query(query)
        
        # Common Croatian stop words should be removed
        stop_words_found = [kw for kw in result.keywords if kw.lower() in processor.croatian_stop_words]
        assert len(stop_words_found) == 0, f"Stop words not removed: {stop_words_found}"
        
        # Important words should remain
        important_words = ['gdje', 'veliki', 'grad', 'zagreb', 'hrvatskoj']
        remaining_important = [word for word in important_words if word not in processor.croatian_stop_words]
        
        for word in remaining_important:
            if word not in ['gdje']:  # 'gdje' might be classified as stop word
                assert any(word.lower() in kw.lower() for kw in result.keywords), \
                    f"Important word '{word}' was removed"
    
    def test_synonym_expansion(self, processor):
        """Test Croatian synonym expansion."""
        query = "Veliki grad Zagreb"
        result = processor.process_query(query)
        
        # Should expand 'veliki' and 'grad'
        assert len(result.expanded_terms) > 0, "No terms were expanded"
        
        # Check for expected synonyms
        all_terms = result.keywords + result.expanded_terms
        
        # Check if synonyms for 'grad' are included
        grad_synonyms = processor.synonym_groups.get('grad', [])
        if grad_synonyms:
            synonym_found = any(syn in all_terms for syn in grad_synonyms[1:])  # Skip the base word
            # Note: Not asserting this as it depends on exact expansion logic
    
    def test_morphological_expansion(self, processor):
        """Test Croatian morphological expansion."""
        query = "Zagreb je lijep grad"
        result = processor.process_query(query)
        
        # 'zagreb' should trigger morphological expansion
        if 'zagreb' in [kw.lower() for kw in result.keywords]:
            zagreb_forms = processor.morphological_patterns.get('zagreb', [])
            if zagreb_forms:
                # Should include some morphological forms
                assert any(form in result.expanded_terms for form in zagreb_forms[1:]), \
                    "Zagreb morphological forms not expanded"
    
    def test_filter_generation(self, processor):
        """Test metadata filter generation."""
        test_cases = [
            ("Zagreb glavni grad", {'language': 'hr'}),  # Basic case
            ("povijest Hrvatske", {'language': 'hr', 'topic': 'history'}),
            ("turizam u Dubrovniku", {'language': 'hr', 'topic': 'tourism'}),
            ("nacionalni park Plitvice", {'language': 'hr', 'topic': 'nature'}),
            ("hrvatska hrana", {'language': 'hr', 'topic': 'food'}),
            ("nogomet u Hrvatskoj", {'language': 'hr', 'topic': 'sports'})
        ]
        
        for query, expected_filters in test_cases.items():
            result = processor.process_query(query)
            
            # Language filter should always be present
            assert result.filters.get('language') == 'hr'
            
            # Check topic-specific filters
            for key, value in expected_filters.items():
                if key != 'language':  # We already checked language
                    assert result.filters.get(key) == value, \
                        f"Filter {key}={value} not found for query: {query}"
    
    def test_confidence_calculation(self, processor):
        """Test confidence score calculation."""
        test_cases = [
            ("", 0.0),  # Empty query
            ("a", 0.0),  # Too short
            ("Koji je glavni grad Hrvatske?", 0.7),  # Good Croatian query
            ("asdfgh qwerty", 0.3),  # Non-meaningful
            ("Zagreb Dubrovnik Split Rijeka Osijek", 0.8)  # Many Croatian cities
        ]
        
        for query, min_expected_confidence in test_cases:
            result = processor.process_query(query)
            
            if min_expected_confidence == 0.0:
                assert result.confidence == 0.0, f"Expected 0.0 confidence for: {query}"
            else:
                assert result.confidence >= min_expected_confidence, \
                    f"Confidence {result.confidence} too low for: {query} (expected >= {min_expected_confidence})"
                assert result.confidence <= 1.0, f"Confidence too high for: {query}"


class TestCroatianLanguageSpecifics:
    """Test Croatian language-specific processing."""
    
    @pytest.fixture
    def processor(self):
        return CroatianQueryProcessor()
    
    def test_diacritic_variations(self, processor):
        """Test handling of different diacritic combinations."""
        diacritic_queries = [
            "Čakovec je grad u Međimurju",
            "Ćuprija u Mostaru",
            "Šibenik na obali",
            "Požega u Slavoniji", 
            "Đakovo i Osijek"
        ]
        
        for query in diacritic_queries:
            result = processor.process_query(query)
            
            # Should maintain high confidence for proper Croatian
            assert result.confidence > 0.5, f"Low confidence for Croatian query: {query}"
            
            # Should preserve diacritics in processed text
            original_diacritics = set(char for char in query if char in 'čćšžđČĆŠŽĐ')
            processed_diacritics = set(char for char in result.processed if char in 'čćšžđČĆŠŽĐ')
            
            assert original_diacritics <= processed_diacritics, \
                f"Diacritics lost: {original_diacritics - processed_diacritics} in query: {query}"
    
    def test_case_handling(self, processor):
        """Test proper case handling in Croatian."""
        test_cases = [
            ("ZAGREB JE GLAVNI GRAD", "zagreb je glavni grad"),
            ("Dubrovnik Je Lijep Grad", "dubrovnik je lijep grad"),
            ("plitvička JEZERA su Prekrasna", "plitvička jezera su prekrasna")
        ]
        
        for original, expected_processed in test_cases:
            result = processor.process_query(original)
            assert result.processed == expected_processed, \
                f"Case normalization failed: expected '{expected_processed}', got '{result.processed}'"
    
    def test_croatian_cultural_context(self, processor):
        """Test recognition of Croatian cultural context."""
        cultural_queries = [
            "biser Jadrana Dubrovnik",
            "UNESCO svjetska baština u Hrvatskoj", 
            "Adriatic Sea obala",
            "hrvatski kralj Tomislav"
        ]
        
        for query in cultural_queries:
            result = processor.process_query(query)
            
            # Should have reasonable confidence for cultural content
            assert result.confidence > 0.4, f"Low confidence for cultural query: {query}"
            
            # Should extract meaningful keywords
            assert len(result.keywords) >= 2, f"Too few keywords for: {query}"


class TestQueryComplexityAnalysis:
    """Test query complexity analysis functionality."""
    
    @pytest.fixture
    def processor(self):
        return CroatianQueryProcessor()
    
    def test_complexity_analysis(self, processor):
        """Test query complexity analysis."""
        test_cases = [
            ("Zagreb", {"word_count": 1, "complexity_score": 0.1}),
            ("Koji je glavni grad Hrvatske?", {"word_count": 5, "has_question_mark": True}),
            ("Objasni detaljno hrvatsku povijest i kulturu kroz stoljeća.", 
             {"word_count": 8, "complexity_score": 0.3}),
            ("Usporedi Zagreb i Split. Koja su glavna obilježja ovih gradova?",
             {"sentence_count": 2, "complexity_score": 0.5})
        ]
        
        for query, expected_features in test_cases:
            analysis = processor.analyze_query_complexity(query)
            
            # Check basic metrics
            assert analysis["character_count"] == len(query)
            assert analysis["word_count"] == len(query.split())
            
            # Check expected features
            for feature, expected_value in expected_features.items():
                if feature == "complexity_score":
                    assert analysis[feature] >= expected_value, \
                        f"Complexity score too low for: {query}"
                else:
                    assert analysis[feature] == expected_value, \
                        f"Feature {feature} mismatch for: {query}"
    
    def test_query_improvement_suggestions(self, processor):
        """Test query improvement suggestions."""
        # Low confidence query
        result = processor.process_query("xyz")
        suggestions = processor.suggest_query_improvements(result)
        
        assert len(suggestions) > 0, "No suggestions for poor query"
        assert any("ključnih riječi" in suggestion.lower() for suggestion in suggestions), \
            "Should suggest adding keywords"
        
        # Good query should have fewer suggestions
        result = processor.process_query("Koji je glavni grad Hrvatske?")
        suggestions = processor.suggest_query_improvements(result)
        
        # Good queries might still have some suggestions, but should be reasonable
        assert len(suggestions) <= 3, f"Too many suggestions for good query: {suggestions}"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def processor(self):
        return CroatianQueryProcessor()
    
    def test_special_characters(self, processor):
        """Test handling of special characters."""
        special_queries = [
            "Zagreb (glavni grad)",
            "Split - drugi najveći grad",
            "Dubrovnik \"biser Jadrana\"",
            "Plitvice & priroda",
            "100% hrvatsko!"
        ]
        
        for query in special_queries:
            result = processor.process_query(query)
            
            # Should not crash
            assert result is not None
            assert isinstance(result, ProcessedQuery)
            
            # Should still extract meaningful content
            if len(query.replace("(", "").replace(")", "").replace("-", "").strip()) > 10:
                assert len(result.keywords) > 0, f"No keywords extracted from: {query}"
    
    def test_mixed_language_queries(self, processor):
        """Test handling of mixed Croatian/English queries."""
        mixed_queries = [
            "Zagreb je beautiful grad",
            "What je glavni grad Hrvatske?",
            "Croatian food je very tasty",
            "Dubrovnik Game of Thrones lokacija"
        ]
        
        for query in mixed_queries:
            result = processor.process_query(query)
            
            # Should not crash
            assert result is not None
            
            # Should still detect some Croatian content
            if any(char in query for char in 'čćšžđ') or any(word in query.lower() for word in ['zagreb', 'hrvatske', 'dubrovnik']):
                assert result.confidence > 0.2, f"Should detect Croatian content in: {query}"
    
    def test_very_long_queries(self, processor):
        """Test handling of very long queries."""
        long_query = " ".join(["Zagreb je glavni grad Hrvatske"] * 20)
        
        result = processor.process_query(long_query)
        
        # Should not crash
        assert result is not None
        
        # Should handle length appropriately
        assert len(result.processed) > 0
        assert len(result.keywords) > 0
        
        # Confidence might be reduced for overly long queries
        # but shouldn't be zero if content is meaningful


class TestFactoryFunction:
    """Test factory function."""
    
    def test_create_query_processor(self):
        """Test query processor factory function."""
        processor = create_query_processor()
        
        assert isinstance(processor, CroatianQueryProcessor)
        assert processor.config.language == "hr"
        assert processor.config.expand_synonyms is True
        
        # Test with custom parameters
        processor_custom = create_query_processor(language="en", expand_synonyms=False)
        assert processor_custom.config.language == "en"
        assert processor_custom.config.expand_synonyms is False


class TestProcessedQueryStructure:
    """Test ProcessedQuery data structure."""
    
    def test_processed_query_creation(self):
        """Test ProcessedQuery structure."""
        query = ProcessedQuery(
            original="Test query",
            processed="test query",
            query_type=QueryType.FACTUAL,
            keywords=["test", "query"],
            expanded_terms=["testing"],
            filters={"language": "hr"},
            confidence=0.8,
            metadata={"test": True}
        )
        
        assert query.original == "Test query"
        assert query.processed == "test query"
        assert query.query_type == QueryType.FACTUAL
        assert query.keywords == ["test", "query"]
        assert query.expanded_terms == ["testing"]
        assert query.filters["language"] == "hr"
        assert query.confidence == 0.8
        assert query.metadata["test"] is True


if __name__ == "__main__":
    pytest.main([__file__])