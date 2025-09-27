import express from 'express';
import cors from 'cors';
import * as trpcExpress from '@trpc/server/adapters/express';
import { createContext } from './trpc/context';
import { appRouter } from './routers/app.router';
import { validateAuthConfig } from './config/auth.config';
import { authMiddleware } from './middleware/auth.middleware';

// Load environment and validate config
validateAuthConfig();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors({
  origin: (process.env.CORS_ORIGIN || 'http://localhost:3001').split(','),
  credentials: true,
}));

app.use(express.json({ limit: '10mb' })); // Support large Markdown content

// Health check endpoint
app.get('/api/v1/health', (_req, res) => {
  res.json({
    status: 'healthy',
    version: '1.0.0',
    services: {
      database: 'connected', // TODO: Actual health checks
      redis: 'connected',
      pythonRag: 'connected',
    },
    timestamp: new Date().toISOString(),
  });
});

// API info endpoint
app.get('/api/v1/info', (_req, res) => {
  res.json({
    version: '1.0.0',
    environment: process.env.NODE_ENV || 'development',
    authMode: process.env.AUTH_MODE || 'mock',
    features: {
      streaming: true,
      websockets: true,
      maxMessageLength: 10000,
      maxFileSizeMB: 10,
    },
    rateLimits: {
      messagesPerMinute: 30,
      messagesPerHour: 500,
      chatsPerDay: 100,
    },
  });
});

// User info endpoint (requires auth)
app.get('/api/v1/user', authMiddleware, (req, res) => {
  res.json({
    user: req.auth?.user,
    tenant: req.auth?.tenant,
    language: req.auth?.language,
    features: req.auth?.features,
    permissions: req.auth?.permissions,
  });
});

// tRPC router
app.use(
  '/api/trpc',
  trpcExpress.createExpressMiddleware({
    router: appRouter,
    createContext,
    onError: ({ path, error }) => {
      console.error(`tRPC Error on ${path}:`, error);
    },
  })
);

// Error handler
app.use((err: any, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  console.error('Error:', err);
  res.status(err.status || 500).json({
    error: {
      code: err.code || 'INTERNAL_ERROR',
      message: err.message || 'Internal server error',
    },
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════════════╗
║           Web API Server Started               ║
╠════════════════════════════════════════════════╣
║  Port: ${PORT}                                    ║
║  Environment: ${process.env.NODE_ENV || 'development'}                   ║
║  Auth Mode: ${process.env.AUTH_MODE || 'mock'}                       ║
╠════════════════════════════════════════════════╣
║  Endpoints:                                    ║
║  - Health: http://localhost:${PORT}/api/v1/health ║
║  - Info: http://localhost:${PORT}/api/v1/info     ║
║  - tRPC: http://localhost:${PORT}/api/trpc        ║
╚════════════════════════════════════════════════╝

📌 Markdown Support: ENABLED
📌 Multi-tenant: ENABLED
📌 Mock Auth: ${process.env.AUTH_MODE === 'mock' ? 'ACTIVE (Development Mode)' : 'DISABLED'}
  `);
});

export { appRouter };
export type { AppRouter } from './routers/app.router';