import { createTRPCReact } from '@trpc/react-query';
import { httpBatchLink } from '@trpc/client';

// Use any for now - in production we'd share types from web-api
export const trpc = createTRPCReact<any>();

export const trpcClient = (trpc as any).createClient({
  links: [
    httpBatchLink({
      url: '/api/trpc',
      headers() {
        return {
          authorization: 'Bearer test-token',
        };
      },
    }),
  ],
});