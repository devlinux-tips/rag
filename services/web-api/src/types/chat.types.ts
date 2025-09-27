export interface Chat {
  id: string;
  title: string;
  feature: string;
  visibility: 'private' | 'tenant_shared';
  tenantId: string;
  userId: string;
  ragConfig?: {
    language: string;
    maxDocuments: number;
    minConfidence: number;
    temperature: number;
  };
  metadata?: {
    tags?: string[];
    description?: string;
  };
  createdAt: Date;
  updatedAt: Date;
  lastMessageAt?: Date;
  messageCount: number;
}

export interface Message {
  id: string;
  chatId: string;
  role: 'user' | 'assistant';
  /**
   * IMPORTANT: This field contains raw Markdown text that must be preserved
   * exactly as received from the LLM. This includes:
   * - Headers (###), bold (**), italic (*), links
   * - Tables, lists (numbered and bulleted)
   * - Code blocks and inline code
   * - Emojis and UTF-8 characters (📌, ✅, etc.)
   * - Line breaks and all formatting
   */
  content: string;
  feature?: string;
  metadata?: {
    // For user messages
    edited?: boolean;
    editedAt?: Date;

    // For assistant messages
    ragContext?: {
      documentsRetrieved: number;
      documentsUsed: number;
      confidence: number;
      searchTimeMs: number;
      sources: Array<{
        documentId: string;
        title: string;
        relevance: number;
        chunk: string;
      }>;
    };
    model?: string;
    provider?: string;
    tokensUsed?: {
      input: number;
      output: number;
      total: number;
    };
    responseTimeMs?: number;
    processingTaskId?: string;
  };
  createdAt: Date;
}

export interface CreateChatInput {
  title: string;
  feature: string;
  visibility?: 'private' | 'tenant_shared';
  ragConfig?: Chat['ragConfig'];
  metadata?: Chat['metadata'];
}

export interface UpdateChatInput {
  title?: string;
  visibility?: 'private' | 'tenant_shared';
  ragConfig?: Partial<Chat['ragConfig']>;
  metadata?: Chat['metadata'];
}

export interface SendMessageInput {
  content: string;
  ragConfig?: {
    maxDocuments?: number;
    minConfidence?: number;
    language?: string;
  };
  modelConfig?: {
    model?: string;
    temperature?: number;
    maxTokens?: number;
  };
  stream?: boolean;
}

export interface MessageResponse {
  userMessage: Message;
  assistantMessage: Message;
  processingTask?: {
    id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    duration?: number;
  };
}