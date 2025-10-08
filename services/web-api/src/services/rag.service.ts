import { AuthContext } from '../types/auth.types';

interface RAGQueryRequest {
  query: string;
  tenant: string;
  user: string;
  language: string;
  scope: string;
  feature?: string;
  max_documents?: number;
  min_confidence?: number;
  temperature?: number;
}

interface RAGQueryResponse {
  response: string;
  sources: Array<{
    documentId: string;
    title: string;
    relevance: number;
    chunk: string;
  }>;
  documentsRetrieved: number;
  documentsUsed: number;
  confidence: number;
  searchTimeMs: number;
  responseTimeMs: number;
  model: string;
  tokensUsed: {
    input: number;
    output: number;
    total: number;
  };
  nnSources?: Array<{
    title: string;
    issue: string;
    eli: string;
    year?: string;
    publisher?: string;
    start_page?: string | null;
    end_page?: string | null;
  }>;
}

export class RAGService {
  private baseUrl: string;
  private timeout: number;

  constructor() {
    this.baseUrl = process.env.PYTHON_RAG_URL || 'http://localhost:8082';
    this.timeout = parseInt(process.env.PYTHON_RAG_TIMEOUT || '30000');
  }

  async query(
    content: string,
    auth: AuthContext,
    feature: string,
    ragConfig?: {
      maxDocuments?: number;
      minConfidence?: number;
      temperature?: number;
    }
  ): Promise<RAGQueryResponse> {
    const request: RAGQueryRequest = {
      query: content,
      tenant: auth.tenant.slug,
      user: auth.user.id,
      language: auth.language,
      scope: "feature",
      feature: feature,
      max_documents: ragConfig?.maxDocuments || 5,
      min_confidence: ragConfig?.minConfidence || 0.7,
      temperature: ragConfig?.temperature || 0.7,
    };

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(`${this.baseUrl}/api/v1/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorMessage = `RAG service error: ${response.status}`;
        try {
          // Try to parse as JSON first
          const errorData = await response.json() as { detail?: string; message?: string };
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch {
          // If not JSON, try to get text (might be HTML)
          try {
            const errorText = await response.text();
            // Check if it's HTML
            if (errorText.trim().startsWith('<')) {
              errorMessage = `RAG service returned HTML error (status ${response.status})`;
            } else {
              errorMessage = errorText.substring(0, 200); // Limit error message length
            }
          } catch {
            // Can't read response body, use status code only
            errorMessage = `RAG service error: ${response.status}`;
          }
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      return data as RAGQueryResponse;
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error(`RAG service timeout after ${this.timeout}ms`);
        }
        throw error;
      }
      throw new Error('Unknown error calling RAG service');
    }
  }

  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}

// Singleton instance
export const ragService = new RAGService();