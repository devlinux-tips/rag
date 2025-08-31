"""
Unit tests for generation system module.
Tests Ollama client, prompt templates, and response parsing.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import httpx

from src.generation.ollama_client import (
    OllamaClient, 
    OllamaConfig, 
    GenerationRequest, 
    GenerationResponse,
    create_ollama_client
)
from src.generation.prompt_templates import (
    CroatianRAGPrompts,
    PromptBuilder,
    get_prompt_for_query_type,
    create_prompt_builder
)
from src.generation.response_parser import (
    CroatianResponseParser,
    ParsedResponse,
    create_response_parser
)


class TestOllamaConfig:
    """Test Ollama configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OllamaConfig()
        
        assert config.base_url == "http://localhost:11434"
        assert config.model == "llama3.1:8b"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.preserve_diacritics is True
        assert config.prefer_formal_style is True
        assert config.include_cultural_context is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = OllamaConfig(
            base_url="http://custom:8080",
            model="custom-model",
            temperature=0.3,
            preserve_diacritics=False
        )
        
        assert config.base_url == "http://custom:8080"
        assert config.model == "custom-model"
        assert config.temperature == 0.3
        assert config.preserve_diacritics is False


class TestGenerationRequest:
    """Test generation request structure."""
    
    def test_generation_request_creation(self):
        """Test creation of generation request."""
        request = GenerationRequest(
            prompt="Test prompt",
            context=["Context 1", "Context 2"],
            query="Test query",
            query_type="factual",
            language="hr"
        )
        
        assert request.prompt == "Test prompt"
        assert request.context == ["Context 1", "Context 2"]
        assert request.query == "Test query"
        assert request.query_type == "factual"
        assert request.language == "hr"


class TestGenerationResponse:
    """Test generation response structure."""
    
    def test_generation_response_creation(self):
        """Test creation of generation response."""
        response = GenerationResponse(
            text="Test response",
            model="test-model",
            tokens_used=100,
            generation_time=1.5,
            confidence=0.8,
            metadata={"test": True}
        )
        
        assert response.text == "Test response"
        assert response.model == "test-model"
        assert response.tokens_used == 100
        assert response.generation_time == 1.5
        assert response.confidence == 0.8
        assert response.metadata["test"] is True
    
    def test_has_croatian_content(self):
        """Test Croatian content detection."""
        with patch('src.generation.ollama_client.detect_croatian_content') as mock_detect:
            mock_detect.return_value = 0.7
            
            response = GenerationResponse(
                text="Hrvatski text sa č, ć, š, ž, đ",
                model="test-model",
                tokens_used=50,
                generation_time=1.0,
                confidence=0.8,
                metadata={}
            )
            
            assert response.has_croatian_content is True
            mock_detect.assert_called_once()


class TestOllamaClient:
    """Test Ollama client functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return OllamaConfig(
            base_url="http://test:11434",
            model="test-model",
            temperature=0.5
        )
    
    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return OllamaClient(config)
    
    def test_client_initialization(self, client, config):
        """Test client initialization."""
        assert client.config == config
        assert client._async_client is None
        assert client._model_info is None
    
    def test_synchronous_generation(self, client):
        """Test synchronous response generation."""
        with patch.object(client, 'generate_text_async') as mock_async:
            mock_response = GenerationResponse(
                text="Test Croatian response",
                model="test-model",
                tokens_used=50,
                generation_time=1.0,
                confidence=0.8,
                metadata={}
            )
            
            # Mock asyncio loop
            mock_async.return_value = mock_response
            
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_loop.run_until_complete.return_value = mock_response
                mock_get_loop.return_value = mock_loop
                
                result = client.generate_response(
                    prompt="Test prompt",
                    context=["Test context"],
                    system_prompt="System prompt"
                )
                
                assert result == "Test Croatian response"
                mock_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_generation_success(self, client):
        """Test successful async generation."""
        request = GenerationRequest(
            prompt="Test prompt",
            context=["Context chunk"],
            query="Test query",
            language="hr"
        )
        
        mock_response_data = {
            "response": "Croatian test response",
            "model": "test-model",
            "eval_count": 50,
            "prompt_eval_count": 20,
            "total_duration": 1000000
        }
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock successful availability check
            mock_client.get.return_value = Mock(status_code=200)
            
            # Mock successful generation
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = mock_response_data
            mock_client.post.return_value = mock_response
            
            client._async_client = mock_client
            
            with patch('src.generation.ollama_client.preserve_croatian_encoding') as mock_preserve:
                mock_preserve.return_value = "Croatian test response"
                
                result = await client.generate_text_async(request)
                
                assert result.text == "Croatian test response"
                assert result.model == "test-model"
                assert result.tokens_used == 50
                assert result.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_async_generation_error(self, client):
        """Test async generation error handling."""
        request = GenerationRequest(
            prompt="Test prompt",
            context=["Context chunk"],
            query="Test query"
        )
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock service unavailable
            mock_client.get.return_value = Mock(status_code=500)
            client._async_client = mock_client
            
            result = await client.generate_text_async(request)
            
            assert "Greška" in result.text
            assert result.confidence == 0.0
            assert "error" in result.metadata
    
    def test_confidence_calculation(self, client):
        """Test confidence calculation."""
        request = GenerationRequest(
            prompt="Test prompt",
            context=["test context with relevant words"],
            query="test query",
            language="hr"
        )
        
        with patch('src.generation.ollama_client.detect_croatian_content') as mock_detect:
            mock_detect.return_value = 0.8
            
            # Test good response
            good_text = "Ovo je detaljan hrvatski odgovor sa relevantnim informacijama test query"
            confidence = client._calculate_confidence(good_text, request)
            assert confidence > 0.5
            
            # Test poor response
            poor_text = "Ne znam"
            confidence = client._calculate_confidence(poor_text, request)
            assert confidence < 0.5
    
    def test_health_check(self, client):
        """Test health check functionality."""
        with patch('requests.get') as mock_get:
            # Test successful health check
            mock_get.return_value = Mock(status_code=200)
            assert client.health_check() is True
            
            # Test failed health check
            mock_get.side_effect = Exception("Connection error")
            assert client.health_check() is False


class TestPromptTemplates:
    """Test Croatian prompt templates."""
    
    def test_base_prompt_structure(self):
        """Test base prompt template structure."""
        template = CroatianRAGPrompts.QUESTION_ANSWERING
        
        assert "hrvatski" in template.system_prompt.lower()
        assert "kontekst" in template.context_template.lower()
        assert "pitanje" in template.user_template.lower()
    
    def test_cultural_context_template(self):
        """Test cultural context template."""
        template = CroatianRAGPrompts.CULTURAL_CONTEXT
        
        assert "kultura" in template.system_prompt.lower()
        assert "povijest" in template.system_prompt.lower()
        assert "cultural context" in template.user_template.lower() or "kulturni kontekst" in template.user_template.lower()
    
    def test_tourism_template(self):
        """Test tourism template."""
        template = CroatianRAGPrompts.TOURISM
        
        assert "turistički" in template.system_prompt.lower()
        assert "odredišt" in template.system_prompt.lower()
    
    def test_query_type_selection(self):
        """Test automatic query type selection."""
        # Test cultural queries
        cultural_queries = [
            "Kakva je povijest Zagreba?",
            "Objasni hrvatsku kulturu",
            "Što je biser Jadrana?"
        ]
        
        for query in cultural_queries:
            template = get_prompt_for_query_type(query)
            assert template == CroatianRAGPrompts.CULTURAL_CONTEXT
        
        # Test tourism queries
        tourism_queries = [
            "Najbolja mjesta za turizam u Hrvatskoj",
            "Koje plaže posjetiti u Istri?",
            "Nacionalni parkovi u Hrvatskoj"
        ]
        
        for query in tourism_queries:
            template = get_prompt_for_query_type(query)
            assert template == CroatianRAGPrompts.TOURISM
        
        # Test factual queries
        factual_queries = [
            "Koji je glavni grad Hrvatske?",
            "Kada je osnovana Hrvatska?",
            "Koliko stanovnika ima Zagreb?"
        ]
        
        for query in factual_queries:
            template = get_prompt_for_query_type(query)
            assert template == CroatianRAGPrompts.FACTUAL_QA
        
        # Test summary queries
        summary_queries = [
            "Sažmi povijest Hrvatske",
            "Ukratko o hrvatskim gradovima"
        ]
        
        for query in summary_queries:
            template = get_prompt_for_query_type(query)
            assert template == CroatianRAGPrompts.SUMMARIZATION
    
    def test_prompt_builder(self):
        """Test prompt builder functionality."""
        template = CroatianRAGPrompts.QUESTION_ANSWERING
        builder = PromptBuilder(template)
        
        query = "Što je Zagreb?"
        context = ["Zagreb je glavni grad Hrvatske.", "Zagreb ima 800.000 stanovnika."]
        
        system_prompt, user_prompt = builder.build_prompt(query, context)
        
        assert system_prompt == template.system_prompt
        assert query in user_prompt
        assert all(chunk in user_prompt for chunk in context)
        assert "Zagreb" in user_prompt
    
    def test_context_length_limiting(self):
        """Test context length limiting in prompt builder."""
        template = CroatianRAGPrompts.QUESTION_ANSWERING
        builder = PromptBuilder(template)
        
        # Create very long context
        long_context = ["Very long context chunk " * 100 for _ in range(10)]
        
        system_prompt, user_prompt = builder.build_prompt(
            "Test query", 
            long_context, 
            max_context_length=500
        )
        
        # Check that prompt is within reasonable length
        assert len(user_prompt) < 1000  # Should be truncated
        assert "[Dokument 1]" in user_prompt  # Should have document headers


class TestResponseParser:
    """Test Croatian response parser."""
    
    @pytest.fixture
    def parser(self):
        """Create test parser."""
        return CroatianResponseParser()
    
    def test_basic_parsing(self, parser):
        """Test basic response parsing."""
        response = "Zagreb je glavni grad Hrvatske sa oko 800.000 stanovnika."
        result = parser.parse_response(response, "Što je Zagreb?")
        
        assert result.content == response
        assert result.has_answer is True
        assert result.language == "hr"
        assert result.confidence is not None
    
    def test_no_answer_detection(self, parser):
        """Test no answer detection."""
        no_answer_responses = [
            "Ne mogu pronaći tu informaciju.",
            "Nije dostupno u dokumentima.",
            "Ne znam odgovor na to pitanje."
        ]
        
        for response in no_answer_responses:
            result = parser.parse_response(response)
            assert result.has_answer is False
    
    def test_confidence_estimation(self, parser):
        """Test confidence estimation."""
        # High confidence indicators
        high_conf_response = "Jasno je da je Zagreb glavni grad Hrvatske."
        result = parser.parse_response(high_conf_response)
        assert result.confidence > 0.7
        
        # Low confidence indicators
        low_conf_response = "Možda je Zagreb glavni grad, teško je reći."
        result = parser.parse_response(low_conf_response)
        assert result.confidence < 0.5
    
    def test_source_extraction(self, parser):
        """Test source reference extraction."""
        response = "Prema dokumentu, Zagreb je glavni grad. [Dokument 1] spominje populaciju."
        result = parser.parse_response(response)
        
        assert len(result.sources_mentioned) > 0
        assert any("dokument" in source.lower() for source in result.sources_mentioned)
    
    def test_language_detection(self, parser):
        """Test language detection."""
        # Croatian text
        croatian_response = "Zagreb je lijepi grad sa puno zanimljivih stvari."
        result = parser.parse_response(croatian_response)
        assert result.language == "hr"
        
        # English text
        english_response = "Zagreb is the capital city and the largest city of Croatia."
        result = parser.parse_response(english_response)
        assert result.language == "en"
    
    def test_response_cleaning(self, parser):
        """Test response text cleaning."""
        dirty_response = "   Answer:    Zagreb   je    glavni grad.   "
        result = parser.parse_response(dirty_response)
        
        assert result.content == "Zagreb je glavni grad."
        assert not result.content.startswith("Answer:")
    
    def test_display_formatting(self, parser):
        """Test display formatting."""
        response = ParsedResponse(
            content="Zagreb je glavni grad Hrvatske.",
            confidence=0.9,
            sources_mentioned=["[Dokument 1]"],
            has_answer=True
        )
        
        formatted = parser.format_for_display(response)
        
        assert "Zagreb je glavni grad Hrvatske." in formatted
        assert "visoka pouzdanost" in formatted
        assert "[Dokument 1]" in formatted


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_ollama_client(self):
        """Test Ollama client factory."""
        client = create_ollama_client(
            model="custom-model",
            base_url="http://custom:8080",
            temperature=0.3
        )
        
        assert isinstance(client, OllamaClient)
        assert client.config.model == "custom-model"
        assert client.config.base_url == "http://custom:8080"
        assert client.config.temperature == 0.3
    
    def test_create_prompt_builder(self):
        """Test prompt builder factory."""
        query = "Objasni hrvatsku povijest"
        builder = create_prompt_builder(query)
        
        assert isinstance(builder, PromptBuilder)
        # Should select explanatory template for this query
        assert builder.template == CroatianRAGPrompts.EXPLANATORY
    
    def test_create_response_parser(self):
        """Test response parser factory."""
        parser = create_response_parser()
        
        assert isinstance(parser, CroatianResponseParser)
        assert len(parser.no_answer_patterns) > 0
        assert len(parser.confidence_indicators) > 0


class TestIntegration:
    """Test integration between generation components."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Ollama client."""
        client = Mock(spec=OllamaClient)
        client.generate_response.return_value = "Zagreb je glavni grad Hrvatske sa oko 800.000 stanovnika."
        return client
    
    def test_end_to_end_generation_flow(self, mock_client):
        """Test complete generation flow."""
        # 1. Create prompt
        query = "Što je Zagreb?"
        context = ["Zagreb je glavni i najveći grad Republike Hrvatske."]
        
        builder = create_prompt_builder(query)
        system_prompt, user_prompt = builder.build_prompt(query, context)
        
        # 2. Generate response
        raw_response = mock_client.generate_response(user_prompt)
        
        # 3. Parse response
        parser = create_response_parser()
        parsed_response = parser.parse_response(raw_response, query, context)
        
        # 4. Format for display
        formatted = parser.format_for_display(parsed_response)
        
        assert "Zagreb" in formatted
        assert parsed_response.has_answer is True
        assert parsed_response.language == "hr"
    
    def test_cultural_query_integration(self):
        """Test integration for Croatian cultural queries."""
        query = "Objasni kulturu Dubrovnika"
        
        # Should select cultural context template
        template = get_prompt_for_query_type(query)
        assert template == CroatianRAGPrompts.CULTURAL_CONTEXT
        
        # Template should include cultural context instructions
        assert "kultura" in template.system_prompt.lower()
        assert "traditional" in template.system_prompt.lower() or "tradicij" in template.system_prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__])