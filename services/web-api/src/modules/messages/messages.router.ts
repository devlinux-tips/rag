import { z } from 'zod';
import { router, protectedProcedure } from '../../trpc/trpc';
import { TRPCError } from '@trpc/server';
import { prisma } from '../../lib/prisma';
import { ragService } from '../../services/rag.service';

// Validation schemas
const sendMessageSchema = z.object({
  chatId: z.string(),
  content: z.string().min(1).max(10000),
  ragConfig: z.object({
    maxDocuments: z.number().min(1).max(20).optional(),
    minConfidence: z.number().min(0).max(1).optional(),
    temperature: z.number().min(0).max(2).optional(),
    language: z.string().optional(),
  }).optional(),
});

export const messagesRouter = router({
  /**
   * List messages in a chat
   */
  list: protectedProcedure
    .input(z.object({
      chatId: z.string(),
      limit: z.number().min(1).max(100).default(50),
      cursor: z.string().optional(),
      order: z.enum(['asc', 'desc']).default('desc'),
    }))
    .query(async ({ input, ctx }) => {
      // First verify the user has access to the chat
      const chat = await prisma.chat.findUnique({
        where: { id: input.chatId },
      });

      if (!chat) {
        throw new TRPCError({
          code: 'NOT_FOUND',
          message: 'Chat not found',
        });
      }

      // Check access
      const hasAccess =
        chat.userId === ctx.auth.user.id ||
        (chat.tenantId === ctx.auth.tenant.id && chat.visibility === 'tenant_shared');

      if (!hasAccess) {
        throw new TRPCError({
          code: 'FORBIDDEN',
          message: 'Access denied to this chat',
        });
      }

      // Get messages with pagination
      const messages = await prisma.message.findMany({
        where: { chatId: input.chatId },
        take: input.limit,
        skip: input.cursor ? 1 : 0,
        cursor: input.cursor ? { id: input.cursor } : undefined,
        orderBy: { createdAt: input.order },
      });

      // Get total count
      const totalCount = await prisma.message.count({
        where: { chatId: input.chatId },
      });

      return {
        messages,
        pagination: {
          hasMore: messages.length === input.limit,
          nextCursor: messages.length > 0
            ? messages[messages.length - 1].id
            : null,
          totalCount,
        },
      };
    }),

  /**
   * Send a message and get RAG response
   */
  send: protectedProcedure
    .input(sendMessageSchema)
    .mutation(async ({ input, ctx }) => {
      // First verify the user has access to the chat
      const chat = await prisma.chat.findUnique({
        where: { id: input.chatId },
      });

      if (!chat) {
        throw new TRPCError({
          code: 'NOT_FOUND',
          message: 'Chat not found',
        });
      }

      // Check access (only owner or tenant members can send messages)
      const hasAccess =
        chat.userId === ctx.auth.user.id ||
        (chat.tenantId === ctx.auth.tenant.id && chat.visibility === 'tenant_shared');

      if (!hasAccess) {
        throw new TRPCError({
          code: 'FORBIDDEN',
          message: 'Access denied to this chat',
        });
      }

      // Create user message
      const userMessage = await prisma.message.create({
        data: {
          chatId: input.chatId,
          role: 'user',
          content: input.content,
          metadata: {
            edited: false,
          },
        },
      });

      // Create processing task
      const processingTask = await prisma.processingTask.create({
        data: {
          messageId: userMessage.id,
          type: 'rag_query',
          status: 'processing',
          request: {
            content: input.content,
            ragConfig: input.ragConfig,
            feature: chat.feature,
          },
          startedAt: new Date(),
        },
      });

      try {
        // Merge chat's ragConfig with request ragConfig
        const ragConfig = {
          ...(chat.ragConfig as any),
          ...input.ragConfig,
        };

        // Call RAG service
        const startTime = Date.now();

        // Call RAG service
        const ragResponse = await ragService.query(
          input.content,
          ctx.auth,
          chat.feature,
          {
            maxDocuments: ragConfig.maxDocuments,
            minConfidence: ragConfig.minConfidence,
            temperature: ragConfig.temperature,
          }
        );

        const endTime = Date.now();

        // Create assistant message with RAG response
        const assistantMessage = await prisma.message.create({
          data: {
            chatId: input.chatId,
            role: 'assistant',
            content: ragResponse.response,
            feature: chat.feature,
            metadata: {
              ragContext: {
                documentsRetrieved: ragResponse.documentsRetrieved,
                documentsUsed: ragResponse.documentsUsed,
                confidence: ragResponse.confidence,
                searchTimeMs: ragResponse.searchTimeMs,
                responseTimeMs: endTime - startTime,
                tokensUsed: ragResponse.tokensUsed,
              },
              nnSources: ragResponse.nnSources || undefined,
              model: ragResponse.model,
              provider: 'ollama',
              processingTaskId: processingTask.id,
            },
          },
        });

        // Update processing task
        await prisma.processingTask.update({
          where: { id: processingTask.id },
          data: {
            status: 'completed',
            response: ragResponse as any,
            completedAt: new Date(),
            durationMs: endTime - startTime,
          },
        });

        // Update chat's lastMessageAt
        await prisma.chat.update({
          where: { id: input.chatId },
          data: { lastMessageAt: new Date() },
        });

        return {
          userMessage,
          assistantMessage,
          processingTask: {
            id: processingTask.id,
            status: 'completed',
            duration: endTime - startTime,
          },
        };
      } catch (error) {
        // Update processing task with error
        await prisma.processingTask.update({
          where: { id: processingTask.id },
          data: {
            status: 'failed',
            error: {
              message: error instanceof Error ? error.message : 'Unknown error',
              timestamp: new Date().toISOString(),
            },
            completedAt: new Date(),
          },
        });

        // Create user-friendly error message based on language
        const language = ctx.auth.language || 'hr';
        let errorContent: string;

        if (language === 'hr') {
          errorContent = 'Žao mi je, dogodila se greška pri obradi pitanja. Molim pokušajte ponovno.';
        } else {
          errorContent = 'Sorry, I encountered an error processing your request. Please try again.';
        }

        // Create error message
        const errorMessage = await prisma.message.create({
          data: {
            chatId: input.chatId,
            role: 'assistant',
            content: errorContent,
            feature: chat.feature,
            status: 'failed',
            errorMessage: error instanceof Error ? error.message : 'Unknown error',
            metadata: {
              processingTaskId: processingTask.id,
              error: error instanceof Error ? error.message : 'Unknown error',
              errorType: error instanceof Error ? error.name : 'UnknownError',
            },
          },
        });

        // Update chat's lastMessageAt
        await prisma.chat.update({
          where: { id: input.chatId },
          data: { lastMessageAt: new Date() },
        });

        // Return the error message to the frontend instead of throwing
        return {
          userMessage,
          assistantMessage: errorMessage,
          processingTask: {
            id: processingTask.id,
            status: 'failed',
            duration: 0,
          },
        };
      }
    }),

  /**
   * Delete a message
   */
  delete: protectedProcedure
    .input(z.object({
      messageId: z.string(),
    }))
    .mutation(async ({ input, ctx }) => {
      // Get the message and its chat
      const message = await prisma.message.findUnique({
        where: { id: input.messageId },
        include: {
          chat: true,
        },
      });

      if (!message) {
        throw new TRPCError({
          code: 'NOT_FOUND',
          message: 'Message not found',
        });
      }

      // Only chat owner can delete messages
      if (message.chat.userId !== ctx.auth.user.id) {
        throw new TRPCError({
          code: 'FORBIDDEN',
          message: 'Only chat owner can delete messages',
        });
      }

      // Delete the message
      await prisma.message.delete({
        where: { id: input.messageId },
      });

      return { success: true };
    }),

  /**
   * Edit a user message (not assistant messages)
   */
  edit: protectedProcedure
    .input(z.object({
      messageId: z.string(),
      content: z.string().min(1).max(10000),
    }))
    .mutation(async ({ input, ctx }) => {
      // Get the message and its chat
      const message = await prisma.message.findUnique({
        where: { id: input.messageId },
        include: {
          chat: true,
        },
      });

      if (!message) {
        throw new TRPCError({
          code: 'NOT_FOUND',
          message: 'Message not found',
        });
      }

      // Only allow editing user messages
      if (message.role !== 'user') {
        throw new TRPCError({
          code: 'BAD_REQUEST',
          message: 'Only user messages can be edited',
        });
      }

      // Only chat owner can edit messages
      if (message.chat.userId !== ctx.auth.user.id) {
        throw new TRPCError({
          code: 'FORBIDDEN',
          message: 'Only chat owner can edit messages',
        });
      }

      // Update the message
      const updatedMessage = await prisma.message.update({
        where: { id: input.messageId },
        data: {
          content: input.content,
          metadata: {
            ...(message.metadata as any),
            edited: true,
            editedAt: new Date().toISOString(),
          },
        },
      });

      return updatedMessage;
    }),
});