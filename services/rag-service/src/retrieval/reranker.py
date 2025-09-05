"""
Multilingual reranker using BAAI/bge-reranker-v2-m3 for multilingual RAG
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..utils.config_loader import get_reranking_config
from ..utils.error_handler import handle_config_error


@dataclass
class RerankerResult:
    """Result from reranking."""

    content: str
    score: float
    original_rank: int
    new_rank: int
    metadata: Dict[str, Any]


class MultilingualReranker:
    """Multilingual cross-encoder reranker using BGE-reranker-v2-m3."""

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        max_length: int = None,
        batch_size: int = None,
    ):
        """
        Initialize multilingual reranker with config.

        Args:
            model_name: HuggingFace model name (from config if None)
            device: Device to run on (from config if None)
            max_length: Maximum sequence length (from config if None)
            batch_size: Batch size for processing (from config if None)
        """
        # Load configuration with DRY pattern
        config = handle_config_error(
            operation=lambda: get_reranking_config(),
            fallback_value={
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "device": "cpu",
                "batch_size": 4,
                "confidence_threshold": 0.5,
            },
            config_file="config/config.toml",
            section="[reranking]",
        )

        self.model_name = model_name or config["model_name"]
        self.device = device or config["device"]
        self.max_length = max_length or 512  # Not in config yet
        self.batch_size = batch_size or config["batch_size"]

        self.tokenizer = None
        self.model = None
        self.is_loaded = False

    def load_model(self):
        """Load the reranker model and tokenizer."""
        try:
            print(f"🔧 Loading reranker model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            # Determine optimal dtype based on device
            model_dtype = torch.float32 if self.device == "cpu" else torch.float16

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Convert to appropriate dtype and device
            self.model = self.model.to(device=self.device, dtype=model_dtype)
            self.model.eval()

            self.is_loaded = True
            print("✅ Reranker model loaded")

        except Exception as e:
            print(f"❌ Failed to load reranker: {e}")
            print("💡 Falling back to score-based reranking")
            self.is_loaded = False

    def rerank(
        self,
        query: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[RerankerResult]:
        """
        Rerank documents using cross-encoder model.

        Args:
            query: Search query
            documents: List of document texts to rerank
            metadatas: Optional metadata for each document
            top_k: Number of top results to return

        Returns:
            List of RerankerResult objects
        """
        if metadatas is None:
            metadatas = [{"index": i} for i in range(len(documents))]

        if not self.is_loaded:
            # Fallback: return documents in original order with dummy scores
            return [
                RerankerResult(
                    content=doc,
                    score=1.0 - (i * 0.1),  # Decreasing scores
                    original_rank=i,
                    new_rank=i,
                    metadata=meta,
                )
                for i, (doc, meta) in enumerate(zip(documents[:top_k], metadatas[:top_k]))
            ]

        # Prepare query-document pairs
        pairs = [(query, doc) for doc in documents]

        # Get reranker scores in batches
        all_scores = []

        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            batch_scores = self._score_batch(batch_pairs)
            all_scores.extend(batch_scores)

        # Create results with scores
        results = []
        for i, (doc, meta, score) in enumerate(zip(documents, metadatas, all_scores)):
            results.append(
                RerankerResult(
                    content=doc,
                    score=score,
                    original_rank=i,
                    new_rank=-1,  # Will be set after sorting
                    metadata=meta,
                )
            )

        # Sort by score (descending) and assign new ranks
        results.sort(key=lambda x: x.score, reverse=True)
        for new_rank, result in enumerate(results):
            result.new_rank = new_rank

        return results[:top_k]

    def _score_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score a batch of query-document pairs."""
        if not self.is_loaded:
            return [0.5] * len(pairs)

        try:
            # Tokenize pairs
            encoded = self.tokenizer(
                pairs,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Get scores
            with torch.no_grad():
                outputs = self.model(**encoded)
                scores = torch.nn.functional.sigmoid(outputs.logits).squeeze(-1)

                # Convert to numpy
                if self.device == "cuda":
                    scores = scores.cpu()
                scores = scores.numpy()

                # Ensure it's a list of floats
                if scores.ndim == 0:
                    scores = [float(scores)]
                else:
                    scores = scores.tolist()

            return scores

        except Exception as e:
            print(f"❌ Reranker scoring error: {e}")
            return [0.5] * len(pairs)

    def explain_reranking(self, results: List[RerankerResult]) -> str:
        """Generate explanation of reranking for debugging."""
        explanation = "🎯 Reranking Results:\n"
        explanation += f"Model: {self.model_name}\n"
        explanation += f"Device: {self.device}\n\n"

        for result in results:
            rank_change = result.original_rank - result.new_rank
            change_symbol = "📈" if rank_change > 0 else "📉" if rank_change < 0 else "➡️"

            explanation += f"Rank {result.new_rank + 1}: Score {result.score:.3f} {change_symbol}\n"
            explanation += f"  Original rank: {result.original_rank + 1}\n"
            explanation += f"  Content: {result.content[:100]}...\n\n"

        return explanation


def create_lightweight_reranker() -> MultilingualReranker:
    """Create a CPU-friendly reranker instance."""
    return MultilingualReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        device="cpu",
        max_length=512,
        batch_size=4,  # Small batch for CPU
    )
