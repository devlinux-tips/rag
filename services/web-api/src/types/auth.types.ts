// Database model types
export interface User {
  id: string;
  email: string;
  name: string;
  role: string;
  tenantId: string;
  features: string[];
  permissions: string[];
  language: string;
  timezone: string;
  settings: Record<string, any>;
  emailVerified: boolean;
  emailVerifiedAt: Date | null;
  lastLoginAt: Date | null;
  passwordChangedAt: Date;
  createdAt: Date;
  updatedAt: Date;
}

export interface Tenant {
  id: string;
  name: string;
  slug: string;
  status: string;
  features: string[];
  settings: Record<string, any>;
  createdAt: Date;
  updatedAt: Date;
}

export interface RefreshToken {
  id: string;
  token: string;
  userId: string;
  deviceInfo: string | null;
  ipAddress: string | null;
  expiresAt: Date;
  isRevoked: boolean;
  revokedAt: Date | null;
  createdAt: Date;
  updatedAt: Date;
}

// Request/Response types
export interface RegisterRequest {
  email: string;
  password: string;
  name: string;
  tenantSlug?: string; // If not provided, create personal tenant
}

export interface RegisterResponse {
  user: {
    id: string;
    email: string;
    name: string;
  };
  tenant: {
    id: string;
    slug: string;
    name: string;
  };
  tokens: {
    accessToken: string;
    refreshToken: string;
  };
}

export interface LoginRequest {
  email: string;
  password: string;
  deviceInfo?: string;
}

export interface LoginResponse {
  user: {
    id: string;
    email: string;
    name: string;
    role: string;
    language: string;
  };
  tenant: {
    id: string;
    slug: string;
    name: string;
  };
  tokens: {
    accessToken: string;
    refreshToken: string;
  };
}

export interface RefreshTokenRequest {
  refreshToken: string;
}

export interface RefreshTokenResponse {
  tokens: {
    accessToken: string;
    refreshToken: string;
  };
}

export interface UpdateProfileRequest {
  name?: string;
  language?: string;
  timezone?: string;
  settings?: Record<string, any>;
}

export interface ChangePasswordRequest {
  currentPassword: string;
  newPassword: string;
}

// Runtime context (used in middleware)
export interface AuthContext {
  user: {
    id: string;
    email: string;
    name: string;
    role: string;
  };
  tenant: {
    id: string;
    slug: string;
    name: string;
  };
  language: string;
  features: string[];
  permissions: string[];
}

export interface JWTPayload {
  userId: string;
  email: string;
  name: string;
  role: string;
  tenantId: string;
  tenantSlug: string;
  tenantName: string;
  language: string;
  features: string[];
  permissions: string[];
  iat?: number;
  exp?: number;
}

export class UnauthorizedError extends Error {
  constructor(message: string = 'Unauthorized') {
    super(message);
    this.name = 'UnauthorizedError';
  }
}

export class ForbiddenError extends Error {
  constructor(message: string = 'Forbidden') {
    super(message);
    this.name = 'ForbiddenError';
  }
}

export class FeatureNotEnabledError extends Error {
  constructor(feature: string, availableFeatures: string[]) {
    super(`The '${feature}' feature is not enabled for your account`);
    this.name = 'FeatureNotEnabledError';
    this.feature = feature;
    this.availableFeatures = availableFeatures;
  }

  feature: string;
  availableFeatures: string[];
}