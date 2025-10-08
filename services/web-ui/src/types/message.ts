/**
 * Type definitions for RAG chat messages and metadata
 */

export interface Source {
  citationId: number;
  title: string;
  issue: string;
  eli: string;
  publisher?: string;
  year?: string;
  relevance?: number;
}

export interface RAGContext {
  documentsRetrieved?: number;
  documentsUsed?: number;
  searchTimeMs?: number;
  responseTimeMs?: number;
  tokensUsed?: {
    input?: number;
    output?: number;
    total?: number;
  };
}

export interface MessageMetadata {
  ragContext?: RAGContext;
  nnSources?: Source[];
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: string;
  isError?: boolean;
  metadata?: MessageMetadata;
}
