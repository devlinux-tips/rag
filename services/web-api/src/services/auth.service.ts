import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { nanoid } from 'nanoid';
import { prisma } from '../lib/prisma';
import { authConfig } from '../config/auth.config';
import {
  RegisterRequest,
  RegisterResponse,
  LoginRequest,
  LoginResponse,
  RefreshTokenRequest,
  RefreshTokenResponse,
  UpdateProfileRequest,
  ChangePasswordRequest,
  JWTPayload,
  UnauthorizedError,
  ForbiddenError,
  User,
  Tenant,
} from '../types/auth.types';

// Configuration constants - FAIL-FAST: No fallbacks
const SALT_ROUNDS = 12;
const ACCESS_TOKEN_EXPIRES_IN = '15m';

// Password strength validation
function validatePassword(password: string): string[] {
  const errors: string[] = [];

  if (password.length < 8) {
    errors.push('Password must be at least 8 characters long');
  }

  if (!/[A-Z]/.test(password)) {
    errors.push('Password must contain at least one uppercase letter');
  }

  if (!/[a-z]/.test(password)) {
    errors.push('Password must contain at least one lowercase letter');
  }

  if (!/[0-9]/.test(password)) {
    errors.push('Password must contain at least one number');
  }

  if (!/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) {
    errors.push('Password must contain at least one special character');
  }

  return errors;
}

// Email validation
function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

// Generate JWT access token
function generateAccessToken(user: User, tenant: Tenant): string {
  if (!authConfig.jwtSecret) {
    throw new Error('JWT_SECRET not configured');
  }

  const payload: JWTPayload = {
    userId: user.id,
    email: user.email,
    name: user.name,
    role: user.role,
    tenantId: tenant.id,
    tenantSlug: tenant.slug,
    tenantName: tenant.name,
    language: user.language,
    features: Array.isArray(user.features) ? user.features : [],
    permissions: Array.isArray(user.permissions) ? user.permissions : [],
  };

  return jwt.sign(payload, authConfig.jwtSecret, {
    expiresIn: ACCESS_TOKEN_EXPIRES_IN,
    issuer: 'rag-web-api',
    audience: 'rag-client',
  });
}

// Generate refresh token
async function generateRefreshToken(
  userId: string,
  deviceInfo?: string,
  ipAddress?: string
): Promise<string> {
  const token = nanoid(64); // Cryptographically secure random token
  const expiresAt = new Date();
  expiresAt.setDate(expiresAt.getDate() + 7); // 7 days from now

  await prisma.refreshToken.create({
    data: {
      token,
      userId,
      deviceInfo,
      ipAddress,
      expiresAt,
    },
  });

  return token;
}

// Create tenant slug from name
function createTenantSlug(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .substring(0, 50); // Limit length
}

// Default permissions based on role
function getDefaultPermissions(role: string): string[] {
  switch (role) {
    case 'owner':
      return [
        'chat:create', 'chat:read', 'chat:write', 'chat:delete',
        'user:create', 'user:read', 'user:write', 'user:delete',
        'tenant:read', 'tenant:write',
      ];
    case 'admin':
      return [
        'chat:create', 'chat:read', 'chat:write', 'chat:delete',
        'user:create', 'user:read', 'user:write',
        'tenant:read',
      ];
    case 'user':
    default:
      return ['chat:create', 'chat:read', 'chat:write', 'chat:delete'];
  }
}

// Default features for new tenants
function getDefaultFeatures(): string[] {
  return ['narodne-novine']; // Basic feature set
}

export class AuthService {
  /**
   * Register a new user and optionally create a tenant
   */
  async register(request: RegisterRequest): Promise<RegisterResponse> {
    // Validate input - FAIL-FAST
    if (!request.email || !request.password || !request.name) {
      throw new Error('Email, password, and name are required');
    }

    if (!validateEmail(request.email)) {
      throw new Error('Invalid email format');
    }

    const passwordErrors = validatePassword(request.password);
    if (passwordErrors.length > 0) {
      throw new Error(`Password validation failed: ${passwordErrors.join(', ')}`);
    }

    // Check if user already exists
    const existingUser = await prisma.user.findUnique({
      where: { email: request.email },
    });

    if (existingUser) {
      throw new Error('User with this email already exists');
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(request.password, SALT_ROUNDS);

    // Handle tenant logic
    let tenant: Tenant;

    if (request.tenantSlug) {
      // Join existing tenant
      const existingTenant = await prisma.tenant.findUnique({
        where: { slug: request.tenantSlug },
      });

      if (!existingTenant) {
        throw new Error(`Tenant with slug '${request.tenantSlug}' not found`);
      }

      tenant = existingTenant as Tenant;
    } else {
      // Create personal tenant
      const tenantSlug = createTenantSlug(request.name);

      // Ensure slug is unique
      let uniqueSlug = tenantSlug;
      let counter = 1;
      while (await prisma.tenant.findUnique({ where: { slug: uniqueSlug } })) {
        uniqueSlug = `${tenantSlug}-${counter}`;
        counter++;
      }

      const createdTenant = await prisma.tenant.create({
        data: {
          name: `${request.name}'s Organization`,
          slug: uniqueSlug,
          features: getDefaultFeatures(),
          settings: {},
        },
      });

      tenant = createdTenant as Tenant;
    }

    // Determine user role
    const role = request.tenantSlug ? 'user' : 'owner'; // Owner of personal tenant, user of existing tenant

    // Create user
    const user = await prisma.user.create({
      data: {
        email: request.email,
        password: hashedPassword,
        name: request.name,
        role,
        tenantId: tenant.id,
        features: tenant.features as string[],
        permissions: getDefaultPermissions(role),
        language: 'hr', // Default language
        timezone: 'Europe/Zagreb',
        settings: {},
      },
    }) as User;

    // Generate tokens
    const accessToken = generateAccessToken(user, tenant);
    const refreshToken = await generateRefreshToken(user.id);

    return {
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
      },
      tenant: {
        id: tenant.id,
        slug: tenant.slug,
        name: tenant.name,
      },
      tokens: {
        accessToken,
        refreshToken,
      },
    };
  }

  /**
   * Login user with email and password
   */
  async login(request: LoginRequest, ipAddress?: string): Promise<LoginResponse> {
    // Validate input - FAIL-FAST
    if (!request.email || !request.password) {
      throw new UnauthorizedError('Email and password are required');
    }

    // Find user with tenant
    const user = await prisma.user.findUnique({
      where: { email: request.email },
      include: { tenant: true },
    });

    if (!user) {
      throw new UnauthorizedError('Invalid email or password');
    }

    // Verify password
    const isPasswordValid = await bcrypt.compare(request.password, user.password);
    if (!isPasswordValid) {
      throw new UnauthorizedError('Invalid email or password');
    }

    // Check if tenant is active
    if (user.tenant.status !== 'active') {
      throw new ForbiddenError(`Account is ${user.tenant.status}. Please contact support.`);
    }

    // Update last login
    await prisma.user.update({
      where: { id: user.id },
      data: { lastLoginAt: new Date() },
    });

    // Generate tokens
    const accessToken = generateAccessToken(user as User, user.tenant as Tenant);
    const refreshToken = await generateRefreshToken(
      user.id,
      request.deviceInfo,
      ipAddress
    );

    return {
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role,
        language: user.language,
      },
      tenant: {
        id: user.tenant.id,
        slug: user.tenant.slug,
        name: user.tenant.name,
      },
      tokens: {
        accessToken,
        refreshToken,
      },
    };
  }

  /**
   * Refresh access token using refresh token
   */
  async refreshToken(request: RefreshTokenRequest): Promise<RefreshTokenResponse> {
    if (!request.refreshToken) {
      throw new UnauthorizedError('Refresh token is required');
    }

    // Find and validate refresh token
    const tokenRecord = await prisma.refreshToken.findUnique({
      where: { token: request.refreshToken },
      include: {
        user: {
          include: { tenant: true },
        },
      },
    });

    if (!tokenRecord) {
      throw new UnauthorizedError('Invalid refresh token');
    }

    if (tokenRecord.isRevoked) {
      throw new UnauthorizedError('Refresh token has been revoked');
    }

    if (new Date() > tokenRecord.expiresAt) {
      // Clean up expired token
      await prisma.refreshToken.delete({
        where: { id: tokenRecord.id },
      });
      throw new UnauthorizedError('Refresh token has expired');
    }

    // Check if tenant is still active
    if (tokenRecord.user.tenant.status !== 'active') {
      throw new ForbiddenError(`Account is ${tokenRecord.user.tenant.status}`);
    }

    // Generate new tokens
    const accessToken = generateAccessToken(
      tokenRecord.user as User,
      tokenRecord.user.tenant as Tenant
    );
    const newRefreshToken = await generateRefreshToken(
      tokenRecord.user.id,
      tokenRecord.deviceInfo || undefined,
      tokenRecord.ipAddress || undefined
    );

    // Revoke old refresh token
    await prisma.refreshToken.update({
      where: { id: tokenRecord.id },
      data: {
        isRevoked: true,
        revokedAt: new Date(),
      },
    });

    return {
      tokens: {
        accessToken,
        refreshToken: newRefreshToken,
      },
    };
  }

  /**
   * Revoke refresh token (logout)
   */
  async revokeRefreshToken(refreshToken: string): Promise<void> {
    await prisma.refreshToken.updateMany({
      where: { token: refreshToken },
      data: {
        isRevoked: true,
        revokedAt: new Date(),
      },
    });
  }

  /**
   * Revoke all refresh tokens for user (logout from all devices)
   */
  async revokeAllRefreshTokens(userId: string): Promise<void> {
    await prisma.refreshToken.updateMany({
      where: { userId, isRevoked: false },
      data: {
        isRevoked: true,
        revokedAt: new Date(),
      },
    });
  }

  /**
   * Get user profile by ID
   */
  async getUserProfile(userId: string): Promise<User> {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      include: { tenant: true },
    });

    if (!user) {
      throw new Error('User not found');
    }

    return user as User;
  }

  /**
   * Update user profile
   */
  async updateProfile(userId: string, updates: UpdateProfileRequest): Promise<User> {
    // Validate timezone if provided
    if (updates.timezone) {
      try {
        Intl.DateTimeFormat(undefined, { timeZone: updates.timezone });
      } catch {
        throw new Error('Invalid timezone');
      }
    }

    // Validate language if provided
    if (updates.language && !['hr', 'en'].includes(updates.language)) {
      throw new Error('Language must be "hr" or "en"');
    }

    const user = await prisma.user.update({
      where: { id: userId },
      data: {
        ...(updates.name && { name: updates.name }),
        ...(updates.language && { language: updates.language }),
        ...(updates.timezone && { timezone: updates.timezone }),
        ...(updates.settings && { settings: updates.settings }),
      },
    });

    return user as User;
  }

  /**
   * Change user password
   */
  async changePassword(userId: string, request: ChangePasswordRequest): Promise<void> {
    if (!request.currentPassword || !request.newPassword) {
      throw new Error('Current password and new password are required');
    }

    // Validate new password
    const passwordErrors = validatePassword(request.newPassword);
    if (passwordErrors.length > 0) {
      throw new Error(`Password validation failed: ${passwordErrors.join(', ')}`);
    }

    // Get current user
    const user = await prisma.user.findUnique({
      where: { id: userId },
    });

    if (!user) {
      throw new Error('User not found');
    }

    // Verify current password
    const isCurrentPasswordValid = await bcrypt.compare(
      request.currentPassword,
      user.password
    );

    if (!isCurrentPasswordValid) {
      throw new UnauthorizedError('Current password is incorrect');
    }

    // Hash new password
    const hashedNewPassword = await bcrypt.hash(request.newPassword, SALT_ROUNDS);

    // Update password and revoke all refresh tokens
    await prisma.$transaction([
      prisma.user.update({
        where: { id: userId },
        data: {
          password: hashedNewPassword,
          passwordChangedAt: new Date(),
        },
      }),
      prisma.refreshToken.updateMany({
        where: { userId, isRevoked: false },
        data: {
          isRevoked: true,
          revokedAt: new Date(),
        },
      }),
    ]);
  }

  /**
   * Verify JWT token and return payload
   */
  verifyAccessToken(token: string): JWTPayload {
    try {
      if (!authConfig.jwtSecret) {
        throw new Error('JWT_SECRET not configured');
      }

      const payload = jwt.verify(token, authConfig.jwtSecret, {
        issuer: 'rag-web-api',
        audience: 'rag-client',
      }) as JWTPayload;

      return payload;
    } catch (error) {
      if (error instanceof jwt.TokenExpiredError) {
        throw new UnauthorizedError('Access token expired');
      }
      if (error instanceof jwt.JsonWebTokenError) {
        throw new UnauthorizedError('Invalid access token');
      }
      throw error;
    }
  }
}

// Export singleton instance
export const authService = new AuthService();