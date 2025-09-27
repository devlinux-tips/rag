import jwt from 'jsonwebtoken';
import { authConfig } from '../config/auth.config';
import { JWTPayload } from '../types/auth.types';

/**
 * Generate a mock JWT token for testing
 */
export function generateMockJWT(overrides?: Partial<JWTPayload>): string {
  const payload: JWTPayload = {
    userId: 'test_user_123',
    email: 'test@example.com',
    tenantId: 'test_tenant',
    tenantSlug: 'testing',
    language: 'hr',
    features: ['narodne-novine'],
    permissions: ['chat:create', 'chat:read', 'chat:write'],
    iat: Math.floor(Date.now() / 1000),
    exp: Math.floor(Date.now() / 1000) + 3600, // 1 hour
    ...overrides
  };

  return jwt.sign(payload, authConfig.jwtSecret);
}

/**
 * Pre-generated mock tokens for different testing scenarios
 */
export const mockTokens = {
  // Full access token with all features and permissions
  fullAccess: generateMockJWT({
    userId: 'admin_user',
    email: 'admin@example.com',
    tenantId: 'test_tenant',
    tenantSlug: 'testing',
    language: 'hr',
    permissions: [
      'chat:create', 'chat:read', 'chat:write', 'chat:delete',
      'user:read', 'user:write',
      'tenant:read', 'tenant:write'
    ],
    features: ['narodne-novine', 'financial-reports']
  }),

  // Read-only token
  readOnly: generateMockJWT({
    userId: 'viewer_user',
    email: 'viewer@example.com',
    language: 'hr',
    permissions: ['chat:read', 'user:read', 'tenant:read'],
    features: ['narodne-novine']
  }),

  // No features enabled
  noFeatures: generateMockJWT({
    userId: 'limited_user',
    email: 'limited@example.com',
    language: 'hr',
    features: [],
    permissions: ['chat:read']
  }),

  // Different tenant
  differentTenant: generateMockJWT({
    userId: 'other_user',
    email: 'other@example.com',
    tenantId: 'other_tenant',
    tenantSlug: 'other',
    language: 'en',
    features: ['narodne-novine']
  }),

  // Expired token (expired 1 hour ago)
  expired: jwt.sign({
    userId: 'expired_user',
    email: 'expired@example.com',
    tenantId: 'test_tenant',
    tenantSlug: 'testing',
    language: 'hr',
    iat: Math.floor(Date.now() / 1000) - 7200,
    exp: Math.floor(Date.now() / 1000) - 3600
  }, authConfig.jwtSecret)
};

/**
 * Decode and print token for debugging
 */
export function debugToken(token: string): void {
  try {
    const decoded = jwt.decode(token) as JWTPayload;
    console.log('Token Payload:', JSON.stringify(decoded, null, 2));

    // Verify without checking expiration
    jwt.verify(token, authConfig.jwtSecret, { ignoreExpiration: true });
    console.log('✅ Token signature valid');
  } catch (error) {
    console.error('❌ Invalid token:', error);
  }
}

/**
 * Generate a token that expires in N seconds (for testing expiration)
 */
export function generateExpiringToken(expiresInSeconds: number): string {
  return generateMockJWT({
    exp: Math.floor(Date.now() / 1000) + expiresInSeconds
  });
}