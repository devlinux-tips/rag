import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { authConfig } from '../config/auth.config';
import { AuthContext, JWTPayload, UnauthorizedError } from '../types/auth.types';

// Extend Express Request type to include auth context
declare global {
  namespace Express {
    interface Request {
      auth?: AuthContext;
    }
  }
}

/**
 * Extract authentication context from request
 * Supports both mock mode (development) and JWT mode (production)
 */
export async function extractAuthContext(req: Request): Promise<AuthContext> {
  // Mock mode - return predefined context
  if (authConfig.mode === 'mock') {
    return {
      user: {
        id: authConfig.mockUser.id,
        email: authConfig.mockUser.email
      },
      tenant: {
        id: authConfig.mockUser.tenantId,
        slug: authConfig.mockUser.tenantSlug
      },
      language: authConfig.mockUser.language,
      features: authConfig.mockUser.features,
      permissions: authConfig.mockUser.permissions
    };
  }

  // JWT mode - validate and extract from token
  const token = extractTokenFromHeader(req);
  if (!token) {
    throw new UnauthorizedError('No authentication token provided');
  }

  try {
    const decoded = jwt.verify(token, authConfig.jwtSecret) as JWTPayload;

    return {
      user: {
        id: decoded.userId,
        email: decoded.email
      },
      tenant: {
        id: decoded.tenantId,
        slug: decoded.tenantSlug
      },
      language: decoded.language,
      features: decoded.features || [],
      permissions: decoded.permissions || []
    };
  } catch (error) {
    if (error instanceof jwt.TokenExpiredError) {
      throw new UnauthorizedError('Token expired');
    }
    if (error instanceof jwt.JsonWebTokenError) {
      throw new UnauthorizedError('Invalid token');
    }
    throw error;
  }
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