import dotenv from 'dotenv';
import path from 'path';

// Load environment variables based on NODE_ENV
const envFile = process.env.NODE_ENV === 'production' ? '.env' : '.env.development';
dotenv.config({ path: path.resolve(process.cwd(), envFile) });

export interface AuthConfig {
  mode: 'jwt' | 'production';
  jwtSecret: string;
}

// FAIL-FAST: Direct access without fallbacks
function getRequiredEnvVar(name: string, fallbackForDev?: string): string {
  const value = process.env[name];

  // In production, ALL required env vars must be set
  if (process.env.NODE_ENV === 'production') {
    if (!value) {
      throw new Error(`Missing required environment variable: ${name}`);
    }
    return value;
  }

  // In development, allow fallbacks but warn
  if (!value) {
    if (fallbackForDev) {
      console.warn(`⚠️  Using default value for ${name}. Set explicitly for production.`);
      return fallbackForDev;
    } else {
      throw new Error(`Missing required environment variable: ${name}`);
    }
  }

  return value;
}

export const authConfig: AuthConfig = {
  mode: (getRequiredEnvVar('AUTH_MODE', 'jwt')) as AuthConfig['mode'],
  jwtSecret: getRequiredEnvVar('JWT_SECRET', 'dev-secret-change-in-production'),
};

// Validate configuration at startup - FAIL-FAST
export function validateAuthConfig(): void {
  // Validate mode
  const validModes = ['jwt', 'production'];
  if (!validModes.includes(authConfig.mode)) {
    throw new Error(`Invalid AUTH_MODE: ${authConfig.mode}. Must be one of: ${validModes.join(', ')}`);
  }

  // JWT Secret validation
  if (!authConfig.jwtSecret) {
    throw new Error('JWT_SECRET is required for JWT authentication');
  }

  // Production mode validation
  if (authConfig.mode === 'production') {
    if (authConfig.jwtSecret === 'dev-secret-change-in-production') {
      throw new Error('JWT_SECRET must be set to a secure value for production environment');
    }

    if (authConfig.jwtSecret.length < 32) {
      throw new Error('JWT_SECRET must be at least 32 characters long for production');
    }
  }

  console.log(`✅ Authentication mode: JWT`);

  if (authConfig.mode !== 'production' && authConfig.jwtSecret === 'dev-secret-change-in-production') {
    console.warn('⚠️  Using default JWT_SECRET for development. Change for production!');
  }
}