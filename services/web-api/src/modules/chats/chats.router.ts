import { z } from 'zod';
import { router, protectedProcedure } from '../../trpc/trpc';
import { TRPCError } from '@trpc/server';
import { prisma } from '../../lib/prisma';
import { Prisma } from '@prisma/client';

// Validation schemas
const createChatSchema = z.object({
  title: z.string().min(1).max(255),
  feature: z.string(),
  visibility: z.enum(['private', 'tenant_shared']).optional().default('private'),
  ragConfig: z.object({
    language: z.string(),
    maxDocuments: z.number().min(1).max(20),
    minConfidence: z.number().min(0).max(1),
    temperature: z.number().min(0).max(2),
  }).optional(),
  metadata: z.object({
    tags: z.array(z.string()).optional(),
    description: z.string().optional(),
  }).optional(),
});

const updateChatSchema = z.object({
  title: z.string().min(1).max(255).optional(),
  visibility: z.enum(['private', 'tenant_shared']).optional(),
  ragConfig: z.object({
    language: z.string().optional(),
    maxDocuments: z.number().min(1).max(20).optional(),
    minConfidence: z.number().min(0).max(1).optional(),
    temperature: z.number().min(0).max(2).optional(),
  }).optional(),
  metadata: z.object({
    tags: z.array(z.string()).optional(),
    description: z.string().optional(),
  }).optional(),
});

export const chatsRouter = router({
  /**
   * List chats accessible to the user
   */
  list: protectedProcedure
    .input(z.object({
      feature: z.string().optional(),
      visibility: z.enum(['private', 'tenant_shared', 'all']).optional(),
      search: z.string().optional(),
      limit: z.number().min(1).max(100).default(20),
      cursor: z.string().optional(),
      sortBy: z.enum(['createdAt', 'updatedAt', 'lastMessageAt']).optional(),
      sortOrder: z.enum(['asc', 'desc']).default('desc'),
    }))
    .query(async ({ input, ctx }) => {
      const where: Prisma.ChatWhereInput = {
        tenantId: ctx.auth.tenant.id,
        AND: [
          // User access: own chats OR shared chats
          {
            OR: [
              { userId: ctx.auth.user.id },
              { visibility: 'tenant_shared' }
            ]
          }
        ]
      };

      // Feature filtering
      if (input.feature) {
        where.feature = input.feature;
      }

      // Visibility filtering
      if (input.visibility && input.visibility !== 'all') {
        where.visibility = input.visibility;
      }

      // Search filtering
      if (input.search) {
        where.OR = [
          { title: { contains: input.search, mode: 'insensitive' } },
          // Note: JSON path search would require raw query or different approach
          // Simplified to just title search for now
        ];
      }

      // Get total count
      const totalCount = await prisma.chat.count({ where });

      // Get chats with pagination
      const sortField = input.sortBy || 'createdAt';
      const chats = await prisma.chat.findMany({
        where,
        take: input.limit,
        skip: input.cursor ? 1 : 0,
        cursor: input.cursor ? { id: input.cursor } : undefined,
        orderBy: { [sortField]: input.sortOrder },
        include: {
          _count: {
            select: { messages: true }
          }
        }
      });

      // Transform to include message count
      const transformedChats = chats.map(chat => ({
        ...chat,
        messageCount: chat._count.messages,
        _count: undefined
      }));

      return {
        chats: transformedChats,
        pagination: {
          hasMore: chats.length === input.limit,
          nextCursor: chats.length > 0 ? chats[chats.length - 1].id : null,
          totalCount,
        },
      };
    }),

  /**
   * Create a new chat
   */
  create: protectedProcedure
    .input(createChatSchema)
    .mutation(async ({ input, ctx }) => {
      // Validate feature access
      if (!ctx.auth.features.includes(input.feature)) {
        throw new TRPCError({
          code: 'FORBIDDEN',
          message: `Feature '${input.feature}' not enabled for your account`,
          cause: {
            feature: input.feature,
            availableFeatures: ctx.auth.features,
          },
        });
      }

      // Create the chat
      const chat = await prisma.chat.create({
        data: {
          title: input.title,
          feature: input.feature,
          visibility: input.visibility || 'private',
          tenantId: ctx.auth.tenant.id,
          userId: ctx.auth.user.id,
          language: ctx.auth.language,
          ragConfig: input.ragConfig || {
            language: ctx.auth.language,
            maxDocuments: 5,
            minConfidence: 0.7,
            temperature: 0.7,
          },
          metadata: input.metadata || {},
        },
      });

      return chat;
    }),

  /**
   * Get chat details
   */
  getById: protectedProcedure
    .input(z.object({
      chatId: z.string(),
    }))
    .query(async ({ input, ctx }) => {
      const chat = await prisma.chat.findUnique({
        where: { id: input.chatId },
        include: {
          _count: {
            select: { messages: true }
          }
        }
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

      // Add permissions info
      const permissions = {
        canEdit: chat.userId === ctx.auth.user.id,
        canDelete: chat.userId === ctx.auth.user.id,
        canShare: chat.userId === ctx.auth.user.id,
      };

      return {
        ...chat,
        messageCount: chat._count.messages,
        _count: undefined,
        permissions,
      };
    }),

  /**
   * Update chat
   */
  update: protectedProcedure
    .input(z.object({
      chatId: z.string(),
      updates: updateChatSchema,
    }))
    .mutation(async ({ input, ctx }) => {
      // Check if chat exists and user owns it
      const existingChat = await prisma.chat.findUnique({
        where: { id: input.chatId },
      });

      if (!existingChat) {
        throw new TRPCError({
          code: 'NOT_FOUND',
          message: 'Chat not found',
        });
      }

      // Only owner can update
      if (existingChat.userId !== ctx.auth.user.id) {
        throw new TRPCError({
          code: 'FORBIDDEN',
          message: 'Only chat owner can update',
        });
      }

      // Prepare update data
      const updateData: Prisma.ChatUpdateInput = {};

      if (input.updates.title !== undefined) {
        updateData.title = input.updates.title;
      }

      if (input.updates.visibility !== undefined) {
        updateData.visibility = input.updates.visibility;
      }

      if (input.updates.ragConfig) {
        updateData.ragConfig = {
          ...(existingChat.ragConfig as any),
          ...input.updates.ragConfig,
        };
      }

      if (input.updates.metadata !== undefined) {
        updateData.metadata = input.updates.metadata;
      }

      // Update the chat
      const updatedChat = await prisma.chat.update({
        where: { id: input.chatId },
        data: updateData,
      });

      return updatedChat;
    }),

  /**
   * Delete chat
   */
  delete: protectedProcedure
    .input(z.object({
      chatId: z.string(),
    }))
    .mutation(async ({ input, ctx }) => {
      // Check if chat exists and user owns it
      const chat = await prisma.chat.findUnique({
        where: { id: input.chatId },
      });

      if (!chat) {
        throw new TRPCError({
          code: 'NOT_FOUND',
          message: 'Chat not found',
        });
      }

      // Only owner can delete
      if (chat.userId !== ctx.auth.user.id) {
        throw new TRPCError({
          code: 'FORBIDDEN',
          message: 'Only chat owner can delete',
        });
      }

      // Delete the chat (messages will cascade delete)
      await prisma.chat.delete({
        where: { id: input.chatId },
      });

      return { success: true };
    }),
});