#!/bin/bash
echo "🐍 Starting RAG development environment..."
source venv/bin/activate
cd services/rag-service
python -c "
import asyncio
from src.pipeline.rag_system import RAGSystem

async def test_rag():
    print('🧪 Testing RAG system...')
    rag = RAGSystem(language='hr')
    await rag.initialize()
    print('✅ RAG system initialized successfully')

if __name__ == '__main__':
    asyncio.run(test_rag())
"
