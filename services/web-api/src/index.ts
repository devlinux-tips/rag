import express from 'express';
import cors from 'cors';
import * as trpcExpress from '@trpc/server/adapters/express';
import { createContext } from './trpc/context';
import { appRouter } from './routers/app.router';
import { authRouter } from './routers/auth.router';
import { validateAuthConfig } from './config/auth.config';
import { authMiddleware } from './middleware/auth.middleware';
import { prisma } from './lib/prisma';

// Load environment and validate config
validateAuthConfig();

// Database connection check - FAIL-FAST
async function checkDatabaseConnection(): Promise<void> {
  try {
    await prisma.$connect();
    console.log('✅ Database connection established');
  } catch (error) {
    console.error('❌ Database connection failed:', error);
    throw new Error('Failed to connect to database. Please check DATABASE_URL configuration.');
  }
}

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors({
  origin: (process.env.CORS_ORIGIN || 'http://localhost:3001').split(','),
  credentials: true,
}));

app.use(express.json({ limit: '10mb' })); // Support large Markdown content

// Health check endpoint
app.get('/api/v1/health', async (_req, res) => {
  const health = {
    status: 'healthy',
    version: '1.0.0',
    services: {
      database: 'unknown',
      redis: 'unknown',
      pythonRag: 'unknown',
    },
    timestamp: new Date().toISOString(),
  };

  // Check database connection
  try {
    await prisma.$queryRaw`SELECT 1`;
    health.services.database = 'connected';
  } catch (error) {
    health.services.database = 'disconnected';
    health.status = 'degraded';
  }

  // Return health status
  const statusCode = health.status === 'healthy' ? 200 : 503;
  res.status(statusCode).json(health);
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

// Authentication routes
app.use('/api/v1/auth', authRouter);

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

// Start server with database initialization
async function startServer(): Promise<void> {
  try {
    // Check database connection before starting server - FAIL-FAST
    await checkDatabaseConnection();

    app.listen(PORT, () => {
      console.log(`
╔════════════════════════════════════════════════╗
║           Web API Server Started               ║
╠════════════════════════════════════════════════╣
║  Port: ${PORT}                                    ║
║  Environment: ${process.env.NODE_ENV || 'development'}                   ║
║  Auth Mode: JWT                                ║
╠════════════════════════════════════════════════╣
║  Endpoints:                                    ║
║  - Health: http://localhost:${PORT}/api/v1/health ║
║  - Info: http://localhost:${PORT}/api/v1/info     ║
║  - Auth: http://localhost:${PORT}/api/v1/auth     ║
║  - tRPC: http://localhost:${PORT}/api/trpc        ║
╚════════════════════════════════════════════════╝

📌 Markdown Support: ENABLED
📌 Multi-tenant: ENABLED
📌 JWT Authentication: ENABLED
  `);
    });
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n🔄 Shutting down gracefully...');
  await prisma.$disconnect();
  console.log('✅ Database connection closed');
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\n🔄 Shutting down gracefully...');
  await prisma.$disconnect();
  console.log('✅ Database connection closed');
  process.exit(0);
});

// Start the server
startServer();

export { appRouter };
export type { AppRouter } from './routers/app.router';