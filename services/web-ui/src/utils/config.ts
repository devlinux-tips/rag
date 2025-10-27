/**
 * Centralized configuration for the web-ui application
 * Handles API URLs and other environment-specific settings
 */

/**
 * Get the API base URL with proper fallbacks
 * Priority:
 * 1. VITE_API_URL (explicit API URL)
 * 2. VITE_WEB_API_URL + VITE_WEB_API_PORT (host + port)
 * 3. Development defaults
 * 4. Production relative paths
 */
export function getApiBaseUrl(): string {
  const env = import.meta.env;

  // 1. Explicit API URL (highest priority)
  if (env.VITE_API_URL) {
    return env.VITE_API_URL.replace(/\/$/, ''); // Remove trailing slash
  }

  // 2. Construct from host and port
  const host = env.VITE_API_HOST || 'localhost';
  const port = env.VITE_WEB_API_PORT || '3000';
  const protocol = env.MODE === 'production' ? 'https' : 'http';

  if (env.VITE_WEB_API_URL) {
    return `${env.VITE_WEB_API_URL.replace(/\/$/, '')}`;
  }

  // 3. Default to localhost:port in development
  if (env.MODE === 'development') {
    return `${protocol}://${host}:${port}`;
  }

  // 4. Production: use relative path (assumes same domain)
  return '';
}

/**
 * Get the tRPC API URL
 */
export function getTrpcUrl(): string {
  const baseUrl = getApiBaseUrl();
  return `${baseUrl}/api/trpc`;
}

/**
 * Get the REST API URL (for non-tRPC endpoints)
 */
export function getRestApiUrl(): string {
  const baseUrl = getApiBaseUrl();
  return `${baseUrl}/api/v1`;
}

/**
 * Application configuration object
 */
export const config = {
  api: {
    baseUrl: getApiBaseUrl(),
    trpcUrl: getTrpcUrl(),
    restUrl: getRestApiUrl(),
  },
  // Add other config as needed
  isDevelopment: import.meta.env.MODE === 'development',
  isProduction: import.meta.env.MODE === 'production',
} as const;

// Export individual functions for convenience
export const { api } = config;

// Debug logging in development
if (config.isDevelopment) {
  console.log('🔧 Configuration loaded:', {
    apiBaseUrl: api.baseUrl,
    trpcUrl: api.trpcUrl,
    restUrl: api.restUrl,
    mode: import.meta.env.MODE,
  });
}