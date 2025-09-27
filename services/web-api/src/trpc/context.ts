import { CreateExpressContextOptions } from '@trpc/server/adapters/express';
import { extractAuthContext } from '../middleware/auth.middleware';
import { AuthContext } from '../types/auth.types';

export interface Context {
  auth: AuthContext | null;
  req: CreateExpressContextOptions['req'];
  res: CreateExpressContextOptions['res'];
}

/**
 * Create tRPC context with authentication
 */
export async function createContext({
  req,
  res,
}: CreateExpressContextOptions): Promise<Context> {
  let auth: AuthContext | null = null;

  try {
    // Try to extract auth context, but don't fail if not present
    // Individual procedures can require auth if needed
    auth = await extractAuthContext(req);
  } catch {
    // Auth extraction failed, continue with null auth
    // Protected procedures will handle this
  }

  return {
    auth,
    req,
    res,
  };
}