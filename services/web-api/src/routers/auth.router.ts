import { Router } from 'express';
import { authService } from '../services/auth.service';
import { authMiddleware } from '../middleware/auth.middleware';
import {
  RegisterRequest,
  LoginRequest,
  RefreshTokenRequest,
  UpdateProfileRequest,
  ChangePasswordRequest,
  UnauthorizedError,
  ForbiddenError,
} from '../types/auth.types';

const router = Router();

/**
 * POST /api/v1/auth/register
 * Register a new user account
 */
router.post('/register', async (req, res, next): Promise<void> => {
  try {
    const request: RegisterRequest = req.body;

    // Validate required fields - FAIL-FAST
    if (!request.email || !request.password || !request.name) {
      res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Email, password, and name are required',
          details: {
            email: !request.email ? 'Email is required' : undefined,
            password: !request.password ? 'Password is required' : undefined,
            name: !request.name ? 'Name is required' : undefined,
          },
        },
      });
      return;
    }

    const response = await authService.register(request);

    res.status(201).json({
      success: true,
      data: response,
    });
  } catch (error: any) {
    if (error.message.includes('already exists')) {
      res.status(409).json({
        error: {
          code: 'USER_EXISTS',
          message: error.message,
        },
      });
      return;
    }

    if (error.message.includes('validation failed') || error.message.includes('Invalid email')) {
      res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: error.message,
        },
      });
      return;
    }

    if (error.message.includes('not found')) {
      res.status(404).json({
        error: {
          code: 'TENANT_NOT_FOUND',
          message: error.message,
        },
      });
      return;
    }

    next(error);
  }
});

/**
 * POST /api/v1/auth/login
 * Login with email and password
 */
router.post('/login', async (req, res, next): Promise<void> => {
  try {
    const request: LoginRequest = req.body;

    // Validate required fields - FAIL-FAST
    if (!request.email || !request.password) {
      res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Email and password are required',
          details: {
            email: !request.email ? 'Email is required' : undefined,
            password: !request.password ? 'Password is required' : undefined,
          },
        },
      });
      return;
    }

    // Extract client IP for security logging
    const ipAddress = req.ip || req.connection.remoteAddress || 'unknown';

    const response = await authService.login(request, ipAddress);

    res.json({
      success: true,
      data: response,
    });
  } catch (error: any) {
    if (error instanceof UnauthorizedError) {
      res.status(401).json({
        error: {
          code: 'UNAUTHORIZED',
          message: error.message,
        },
      });
      return;
    }

    if (error instanceof ForbiddenError) {
      res.status(403).json({
        error: {
          code: 'ACCOUNT_SUSPENDED',
          message: error.message,
        },
      });
      return;
    }

    next(error);
  }
});

/**
 * POST /api/v1/auth/refresh
 * Refresh access token
 */
router.post('/refresh', async (req, res, next): Promise<void> => {
  try {
    const request: RefreshTokenRequest = req.body;

    // Validate required fields - FAIL-FAST
    if (!request.refreshToken) {
      res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Refresh token is required',
        },
      });
      return;
    }

    const response = await authService.refreshToken(request);

    res.json({
      success: true,
      data: response,
    });
  } catch (error: any) {
    if (error instanceof UnauthorizedError) {
      res.status(401).json({
        error: {
          code: 'INVALID_REFRESH_TOKEN',
          message: error.message,
        },
      });
      return;
    }

    if (error instanceof ForbiddenError) {
      res.status(403).json({
        error: {
          code: 'ACCOUNT_SUSPENDED',
          message: error.message,
        },
      });
      return;
    }

    next(error);
  }
});

/**
 * POST /api/v1/auth/logout
 * Logout (revoke refresh token)
 */
router.post('/logout', async (req, res, next) => {
  try {
    const { refreshToken } = req.body;

    if (refreshToken) {
      await authService.revokeRefreshToken(refreshToken);
    }

    res.json({
      success: true,
      message: 'Logged out successfully',
    });
  } catch (error) {
    next(error);
  }
});

/**
 * POST /api/v1/auth/logout-all
 * Logout from all devices (revoke all refresh tokens)
 */
router.post('/logout-all', authMiddleware, async (req, res, next) => {
  try {
    if (!req.auth) {
      throw new UnauthorizedError('Authentication required');
    }

    await authService.revokeAllRefreshTokens(req.auth.user.id);

    res.json({
      success: true,
      message: 'Logged out from all devices successfully',
    });
  } catch (error) {
    next(error);
  }
});

/**
 * GET /api/v1/auth/profile
 * Get current user profile
 */
router.get('/profile', authMiddleware, async (req, res, next) => {
  try {
    if (!req.auth) {
      throw new UnauthorizedError('Authentication required');
    }

    const user = await authService.getUserProfile(req.auth.user.id);

    // Return profile without sensitive information
    res.json({
      success: true,
      data: {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role,
        language: user.language,
        timezone: user.timezone,
        settings: user.settings,
        features: user.features,
        permissions: user.permissions,
        emailVerified: user.emailVerified,
        emailVerifiedAt: user.emailVerifiedAt,
        lastLoginAt: user.lastLoginAt,
        createdAt: user.createdAt,
        updatedAt: user.updatedAt,
      },
    });
  } catch (error) {
    next(error);
  }
});

/**
 * PUT /api/v1/auth/profile
 * Update user profile
 */
router.put('/profile', authMiddleware, async (req, res, next) => {
  try {
    if (!req.auth) {
      throw new UnauthorizedError('Authentication required');
    }

    const updates: UpdateProfileRequest = req.body;

    // Validate at least one field is provided
    if (!updates.name && !updates.language && !updates.timezone && !updates.settings) {
      res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'At least one field must be provided for update',
        },
      });
      return;
    }

    const updatedUser = await authService.updateProfile(req.auth.user.id, updates);

    res.json({
      success: true,
      data: {
        id: updatedUser.id,
        email: updatedUser.email,
        name: updatedUser.name,
        role: updatedUser.role,
        language: updatedUser.language,
        timezone: updatedUser.timezone,
        settings: updatedUser.settings,
        updatedAt: updatedUser.updatedAt,
      },
    });
  } catch (error: any) {
    if (error.message.includes('Invalid timezone') || error.message.includes('Language must be')) {
      res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: error.message,
        },
      });
      return;
    }

    next(error);
  }
});

/**
 * POST /api/v1/auth/change-password
 * Change user password
 */
router.post('/change-password', authMiddleware, async (req, res, next) => {
  try {
    if (!req.auth) {
      throw new UnauthorizedError('Authentication required');
    }

    const request: ChangePasswordRequest = req.body;

    // Validate required fields - FAIL-FAST
    if (!request.currentPassword || !request.newPassword) {
      res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Current password and new password are required',
          details: {
            currentPassword: !request.currentPassword ? 'Current password is required' : undefined,
            newPassword: !request.newPassword ? 'New password is required' : undefined,
          },
        },
      });
      return;
    }

    await authService.changePassword(req.auth.user.id, request);

    res.json({
      success: true,
      message: 'Password changed successfully. Please login again on all devices.',
    });
  } catch (error: any) {
    if (error instanceof UnauthorizedError) {
      res.status(401).json({
        error: {
          code: 'INVALID_PASSWORD',
          message: error.message,
        },
      });
      return;
    }

    if (error.message.includes('validation failed')) {
      res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: error.message,
        },
      });
      return;
    }

    next(error);
  }
});

export { router as authRouter };