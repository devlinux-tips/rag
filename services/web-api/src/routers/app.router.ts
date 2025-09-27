import { router } from '../trpc/trpc';
import { chatsRouter } from '../modules/chats/chats.router';
import { messagesRouter } from '../modules/messages/messages.router';

/**
 * Main application router
 * Combines all module routers into a single API
 */
export const appRouter = router({
  chats: chatsRouter,
  messages: messagesRouter,
  // TODO: Add more module routers
  // users: usersRouter,
  // tenants: tenantsRouter,
  // features: featuresRouter,
  // settings: settingsRouter,
});

// Export type for client
export type AppRouter = typeof appRouter;