import { initTRPC, TRPCError } from '@trpc/server';
import { Context } from './context';
import { ZodError } from 'zod';

/**
 * Initialize tRPC with context
 */
const t = initTRPC.context<Context>().create({
  errorFormatter({ shape, error }) {
    return {
      ...shape,
      data: {
        ...shape.data,
        zodError:
          error.cause instanceof ZodError
            ? error.cause.flatten()
            : null,
      },
    };
  },
});

export const router = t.router;
export const publicProcedure = t.procedure;

/**
 * Protected procedure that requires authentication
 */
export const protectedProcedure = t.procedure.use(async ({ ctx, next }) => {
  if (!ctx.auth) {
    throw new TRPCError({
      code: 'UNAUTHORIZED',
      message: 'Authentication required',
    });
  }

  return next({
    ctx: {
      ...ctx,
      auth: ctx.auth, // auth is guaranteed to exist
    },
  });
});

/**
 * Procedure that requires specific permissions
 */
export function requirePermission(...permissions: string[]) {
  return protectedProcedure.use(async ({ ctx, next }) => {
    const hasPermission = permissions.some(permission =>
      ctx.auth.permissions.includes(permission)
    );

    if (!hasPermission) {
      throw new TRPCError({
        code: 'FORBIDDEN',
        message: 'Insufficient permissions',
        cause: {
          required: permissions,
          available: ctx.auth.permissions,
        },
      });
    }

    return next();
  });
}

/**
 * Procedure that requires specific features
 */
export function requireFeature(...features: string[]) {
  return protectedProcedure.use(async ({ ctx, next }) => {
    const missingFeatures = features.filter(
      feature => !ctx.auth.features.includes(feature)
    );

    if (missingFeatures.length > 0) {
      throw new TRPCError({
        code: 'FORBIDDEN',
        message: `Features not enabled: ${missingFeatures.join(', ')}`,
        cause: {
          required: features,
          missing: missingFeatures,
          available: ctx.auth.features,
        },
      });
    }

    return next();
  });
}