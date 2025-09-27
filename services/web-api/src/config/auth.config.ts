import dotenv from 'dotenv';
import path from 'path';

// Load environment variables based on NODE_ENV
const envFile = process.env.NODE_ENV === 'production' ? '.env' : '.env.development';
dotenv.config({ path: path.resolve(process.cwd(), envFile) });

export interface AuthConfig {
  mode: 'mock' | 'jwt' | 'production';
  jwtSecret: string;
  mockUser: {
    id: string;
    email: string;
    tenantId: string;
    tenantSlug: string;
    language: string;
    features: string[];
    permissions: string[];
  };
}

export const authConfig: AuthConfig = {
  mode: (process.env.AUTH_MODE || 'mock') as AuthConfig['mode'],
  jwtSecret: process.env.JWT_SECRET || 'dev-secret-change-in-production',
  mockUser: {
    id: process.env.MOCK_USER_ID || 'dev_user_123',
    email: process.env.MOCK_USER_EMAIL || 'dev@example.com',
    tenantId: process.env.MOCK_TENANT_ID || 'dev_tenant',
    tenantSlug: process.env.MOCK_TENANT_SLUG || 'development',
    language: process.env.MOCK_LANGUAGE || 'hr',
    features: (process.env.MOCK_FEATURES || 'narodne-novine').split(','),
    permissions: (process.env.MOCK_PERMISSIONS || 'chat:create,chat:read,chat:write,chat:delete').split(',')
  }
};

// Validate configuration at startup
export function validateAuthConfig(): void {
  if (authConfig.mode === 'production' && authConfig.jwtSecret === 'dev-secret-change-in-production') {
    throw new Error('JWT_SECRET must be changed for production environment');
  }

  if (authConfig.mode === 'mock') {
    console.log('⚠️  Running in MOCK authentication mode - DO NOT USE IN PRODUCTION');
    console.log(`📌 Mock user: ${authConfig.mockUser.email} (${authConfig.mockUser.id})`);
    console.log(`📌 Mock tenant: ${authConfig.mockUser.tenantSlug} (${authConfig.mockUser.tenantId})`);
    console.log(`📌 Mock language: ${authConfig.mockUser.language}`);
    console.log(`📌 Mock features: ${authConfig.mockUser.features.join(', ')}`);
  }
}