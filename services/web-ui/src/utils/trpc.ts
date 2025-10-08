import { createTRPCReact } from '@trpc/react-query';
import { httpBatchLink } from '@trpc/client';

// Use any for now - in production we'd share types from web-api
export const trpc = createTRPCReact<any>();

// Global function to handle authentication errors
function handleAuthError(error: any) {
  const isAuthError =
    error?.data?.code === 'UNAUTHORIZED' ||
    error?.message?.includes('Authentication required') ||
    error?.message?.includes('Invalid token') ||
    error?.message?.includes('Token expired');

  if (isAuthError) {
    // AI-FRIENDLY LOG: Session expired
    console.log('INFO_SESSION_EXPIRED | redirecting_to_login=true | timestamp=' + new Date().toISOString());

    // Clear auth data
    localStorage.removeItem('accessToken');
    localStorage.removeItem('refreshToken');

    // Redirect to login
    window.location.href = '/login';
    return true;
  }
  return false;
}

export const trpcClient = (trpc as any).createClient({
  links: [
    httpBatchLink({
      url: `${window.location.origin}/api/trpc`,
      headers() {
        // Get token from localStorage for authenticated requests
        const token = localStorage.getItem('accessToken');
        return {
          authorization: token ? `Bearer ${token}` : '',
        };
      },
      // Add response interceptor to handle auth errors globally
      fetch(url, options) {
        return fetch(url, options).then(async (response) => {
          // Check if response indicates auth error
          if (response.status === 401) {
            const data = await response.clone().json().catch(() => ({}));
            handleAuthError(data);
          }
          return response;
        });
      },
    }),
  ],
});