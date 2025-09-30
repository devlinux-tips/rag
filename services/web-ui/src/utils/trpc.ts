import { createTRPCReact } from '@trpc/react-query';
import { httpBatchLink } from '@trpc/client';

// Use any for now - in production we'd share types from web-api
export const trpc = createTRPCReact<any>();

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
    }),
  ],
});