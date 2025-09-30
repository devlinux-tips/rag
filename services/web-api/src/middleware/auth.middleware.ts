import { Request, Response, NextFunction } from 'express';
import { authService } from '../services/auth.service';
import { AuthContext, UnauthorizedError } from '../types/auth.types';

// Extend Express Request type to include auth context
declare global {
  namespace Express {
    interface Request {
      auth?: AuthContext;
    }
  }
}

/**
 * Extract authentication context from JWT token
 * FAIL-FAST: No token = immediate rejection
 */
export async function extractAuthContext(req: Request): Promise<AuthContext> {
  // JWT mode - validate and extract from token
  const token = extractTokenFromHeader(req);
  if (!token) {
    throw new UnauthorizedError('No authentication token provided');
  }

  // Use auth service to verify token - FAIL-FAST validation
  const decoded = authService.verifyAccessToken(token);

  return {
    user: {
      id: decoded.userId,
      email: decoded.email,
      name: decoded.name,
      role: decoded.role,
    },
    tenant: {
      id: decoded.tenantId,
      slug: decoded.tenantSlug,
      name: decoded.tenantName,
    },
    language: decoded.language,
    features: decoded.features,
    permissions: decoded.permissions
  };
}

/**
 * Express middleware for authentication
 */
export async function authMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    req.auth = await extractAuthContext(req);
    next();
  } catch (error) {
    if (error instanceof UnauthorizedError) {
      res.status(401).json({
        error: {
          code: 'UNAUTHORIZED',
          message: error.message
        }
      });
    } else {
      next(error);
    }
  }
}

/**
 * Express middleware to require specific permissions
 */
export function requirePermission(...permissions: string[]) {
  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      if (!req.auth) {
        req.auth = await extractAuthContext(req);
      }

      const hasPermission = permissions.some(permission =>
        req.auth!.permissions.includes(permission)
      );

      if (!hasPermission) {
        res.status(403).json({
          error: {
            code: 'FORBIDDEN',
            message: 'Insufficient permissions',
            details: {
              required: permissions,
              available: req.auth.permissions
            }
          }
        });
        return;
      }

      next();
    } catch (error) {
      next(error);
    }
  };
}

/**
 * Express middleware to require specific features
 */
export function requireFeature(...features: string[]) {
  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      if (!req.auth) {
        req.auth = await extractAuthContext(req);
      }

      const missingFeatures = features.filter(feature =>
        !req.auth!.features.includes(feature)
      );

      if (missingFeatures.length > 0) {
        res.status(403).json({
          error: {
            code: 'FEATURE_NOT_ENABLED',
            message: `The following features are not enabled: ${missingFeatures.join(', ')}`,
            details: {
              required: features,
              missing: missingFeatures,
              available: req.auth.features
            }
          }
        });
        return;
      }

      next();
    } catch (error) {
      next(error);
    }
  };
}

/**
 * Extract JWT token from Authorization header
 */
function extractTokenFromHeader(req: Request): string | null {
  const authHeader = req.headers.authorization;
  if (!authHeader) return null;

  const parts = authHeader.split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') return null;

  return parts[1];
}