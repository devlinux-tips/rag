# Incomplete Features

This document tracks features that have partial implementations or interfaces but are not fully functional.

## Reranker System

**Status:** Interface only, no concrete implementation
**Location:** `src/retrieval/reranker.py`

**What exists:**
- `ModelLoader` Protocol (interface)
- `ScoreCalculator` Protocol (interface)
- `MultilingualReranker` class (expects concrete implementations)
- Mock implementations for testing

**What's missing:**
- Concrete `ModelLoader` implementation
- Concrete `ScoreCalculator` implementation
- Integration with actual ML models (e.g., cross-encoder models)

**Current workaround:**
- `factories.py` sets `ranker = None`
- RAG system accepts `ranker: RankerProtocol | None`
- Ranker is stored but never called in production code

**To implement:**
1. Create concrete ModelLoader class that loads cross-encoder models
2. Create concrete ScoreCalculator that uses loaded models for scoring
3. Update factories.py to use real implementations
4. Add configuration for reranker models
