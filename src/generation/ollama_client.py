"""
Ollama client for local LLM integration in Croatian RAG system.
Handles communication with Ollama API for answer generation.
"""

import json
import logging
from typing import Dict, List, Optional, Any
import requests
from dataclasses import dataclass


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30


class OllamaClient:
    """Client for interacting with Ollama API for text generation."""
    
    def __init__(self, config: OllamaConfig = None):
        """
        Initialize Ollama client.
        
        Args:
            config: Configuration object for Ollama settings
        """
        self.config = config or OllamaConfig()
        self.logger = logging.getLogger(__name__)
        
    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request to Ollama API.
        
        Args:
            endpoint: API endpoint path
            payload: Request payload
            
        Returns:
            API response as dictionary
            
        Raises:
            ConnectionError: If Ollama is not running
            requests.RequestException: For other API errors
        """
        url = f"{self.config.base_url}/{endpoint}"
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Make sure Ollama is running with: ollama serve"
            )
        except requests.RequestException as e:
            self.logger.error(f"Ollama API request failed: {e}")
            raise
    
    def is_model_available(self) -> bool:
        """
        Check if the configured model is available in Ollama.
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            response = self._make_request("api/tags", {})
            models = [model["name"] for model in response.get("models", [])]
            return self.config.model in models
            
        except Exception as e:
            self.logger.error(f"Failed to check model availability: {e}")
            return False
    
    def pull_model(self) -> bool:
        """
        Download the configured model if not available.
        
        Returns:
            True if model was pulled successfully, False otherwise
        """
        if self.is_model_available():
            return True
            
        try:
            self.logger.info(f"Pulling model {self.config.model}...")
            payload = {"name": self.config.model}
            self._make_request("api/pull", payload)
            self.logger.info(f"Successfully pulled {self.config.model}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to pull model {self.config.model}: {e}")
            return False
    
    def generate_response(
        self, 
        prompt: str, 
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response using Ollama model.
        
        Args:
            prompt: User query or prompt
            context: List of context chunks from retrieval
            system_prompt: Optional system prompt for behavior control
            
        Returns:
            Generated response text
            
        Raises:
            ValueError: If model is not available
            ConnectionError: If cannot connect to Ollama
        """
        if not self.is_model_available():
            if not self.pull_model():
                raise ValueError(f"Model {self.config.model} is not available")
        
        # Build full prompt with context
        full_prompt = self._build_prompt(prompt, context, system_prompt)
        
        payload = {
            "model": self.config.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        try:
            response = self._make_request("api/generate", payload)
            return response.get("response", "").strip()
            
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            raise
    
    def _build_prompt(
        self, 
        query: str, 
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build complete prompt from query, context, and system instructions.
        
        Args:
            query: User query
            context: Retrieved document chunks
            system_prompt: System instructions
            
        Returns:
            Complete formatted prompt
        """
        parts = []
        
        # Add system prompt if provided
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        
        # Add context if provided
        if context:
            context_text = "\n\n".join(context)
            parts.append(f"Context:\n{context_text}")
        
        # Add the main query
        parts.append(f"Question: {query}")
        parts.append("Answer:")
        
        return "\n\n".join(parts)
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models in Ollama.
        
        Returns:
            List of model names
        """
        try:
            response = self._make_request("api/tags", {})
            return [model["name"] for model in response.get("models", [])]
            
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return []
    
    def health_check(self) -> bool:
        """
        Check if Ollama service is healthy.
        
        Returns:
            True if service is running, False otherwise
        """
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            return response.status_code == 200
            
        except Exception:
            return False


def create_ollama_client(
    model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7
) -> OllamaClient:
    """
    Factory function to create configured Ollama client.
    
    Args:
        model: Model name to use
        base_url: Ollama API base URL
        temperature: Generation temperature
        
    Returns:
        Configured OllamaClient instance
    """
    config = OllamaConfig(
        model=model,
        base_url=base_url,
        temperature=temperature
    )
    return OllamaClient(config)