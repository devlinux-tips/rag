I'll provide you with a comprehensive list of resources for learning RAG (Retrieval-Augmented Generation) with LLMs, including books, code repositories, and practical resources.

## Essential Books on RAG and LLMs

### Core RAG-Focused Resources

**"Building LLM Apps: Create Intelligent Apps and Agents with Large Language Models" by Valentina Alto** (2024)
- Covers RAG architecture patterns in detail
- Includes practical examples with LangChain and vector databases
- Great for understanding embedding strategies and retrieval optimization

**"Hands-On Large Language Models" by Jay Alammar and Maarten Grootendorst** (2024)
- Dedicated chapters on RAG implementation
- Covers both dense and sparse retrieval methods
- Excellent visualizations of RAG pipelines

**"Generative AI with LangChain" by Ben Auffarth** (2023)
- Practical RAG implementations using LangChain
- Covers document loaders, text splitters, and vector stores
- Real-world examples with different embedding models

### Foundational LLM Books with RAG Coverage

**"Natural Language Processing with Transformers" by Lewis Tunstall, Leandro von Werra, and Thomas Wolf** (Revised Edition)
- While not RAG-specific, provides essential transformer knowledge
- Includes sections on retrieval-augmented approaches
- Hugging Face ecosystem integration

**"Designing Machine Learning Systems" by Chip Huyen**
- Covers production RAG system design patterns
- Excellent for understanding scalability and monitoring
- Discusses real-world deployment challenges

## Key RAG Patterns and Techniques

### Advanced RAG Patterns You Should Master:

1. **Hybrid Search**: Combining dense (semantic) and sparse (keyword) retrieval
2. **Reranking**: Using cross-encoders to reorder retrieved documents
3. **Query Expansion**: Enhancing user queries for better retrieval
4. **Contextual Compression**: Reducing retrieved content to relevant portions
5. **Multi-hop Reasoning**: Iterative retrieval for complex queries
6. **Adaptive Retrieval**: Determining when retrieval is necessary

## Code Repositories and Implementations

### Must-Explore GitHub Repositories:

**LangChain** (github.com/langchain-ai/langchain)
- Extensive RAG templates and examples
- Production-ready implementations
- Multiple vector store integrations

**LlamaIndex** (github.com/run-llama/llama_index)
- Advanced RAG techniques like sentence window retrieval
- Auto-merging retrieval strategies
- Knowledge graph integration

**RAG Techniques** (github.com/NirDiamant/RAG_Techniques)
- Comprehensive collection of RAG optimization methods
- Includes evaluation metrics and benchmarks

**OpenAI Cookbook** (github.com/openai/openai-cookbook)
- RAG examples with OpenAI models
- Question-answering implementations
- Best practices for prompt engineering with retrieval

## Online Resources and Courses

### High-Quality Learning Platforms:

**DeepLearning.AI Courses**
- "LangChain for LLM Application Development"
- "Building and Evaluating Advanced RAG Applications"
- "Vector Databases: from Embeddings to Applications"

**Hugging Face Course**
- Free course with RAG sections
- Practical notebooks and exercises

### Technical Blogs and Papers:

**Pinecone Learning Center** (pinecone.io/learn)
- Excellent RAG tutorials and case studies
- Vector database optimization guides

**Weaviate Blog** (weaviate.io/blog)
- Advanced RAG architectures
- Hybrid search implementations

## Practical Implementation Advice

### Key Recommendations for RAG Development:

1. **Start with Chunking Strategy**
   - Experiment with different chunk sizes (typically 200-1000 tokens)
   - Consider overlap between chunks (10-20% is common)
   - Use semantic chunking for better context preservation

2. **Embedding Model Selection**
   - Start with OpenAI's text-embedding-3-small for cost-effectiveness
   - Consider open-source alternatives like BGE or Instructor models
   - Fine-tune embeddings on your domain if possible

3. **Vector Database Choice**
   - Development: ChromaDB or FAISS for simplicity
   - Production: Pinecone, Weaviate, or Qdrant for scalability
   - Consider pgvector for PostgreSQL integration

4. **Evaluation Metrics**
   - Implement RAGAS (Retrieval Augmented Generation Assessment)
   - Track retrieval precision, recall, and MRR
   - Monitor answer relevancy and faithfulness

5. **Common Pitfalls to Avoid**
   - Don't ignore metadata filtering
   - Avoid over-relying on semantic search alone
   - Test with adversarial queries
   - Implement proper document versioning

## Research Papers Worth Reading

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020) - The foundational RAG paper
- "REALM: Retrieval-Augmented Language Model Pre-Training" (Guu et al., 2020)
- "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)

## Tools and Frameworks Comparison

**For Beginners**: Start with LangChain + ChromaDB + OpenAI
**For Production**: Consider LlamaIndex + Pinecone/Weaviate + Multiple LLM providers
**For Research**: Explore Haystack or custom implementations with Transformers library

Remember that RAG is rapidly evolving. Join communities like r/LocalLLaMA on Reddit, the LangChain Discord, or the Hugging Face forums to stay updated with the latest patterns and best practices. The key to mastering RAG is iterative experimentation with different retrieval strategies, chunk sizes, and reranking methods on your specific use case.

## Links

- https://github.com/RagView/RagView
- https://github.com/kruskal-labs/toolfront
- https://github.com/D-Star-AI/dsRAG (test it)
- https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC/ (metadata extraction)
- https://github.com/NirDiamant/RAG_Techniques


## GraphRAG

- https://github.com/getzep/graphiti
- https://github.com/microsoft/graphrag
- https://github.com/ChristopherLyon/graphrag-workbench (visualizer for ms graphrag)
