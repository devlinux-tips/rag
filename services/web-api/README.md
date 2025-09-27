# Web API

Unified Web API for the multi-tenant RAG platform. This is a modular monolith that handles all user-facing operations.

## Architecture

```
src/
├── modules/           # Domain modules (each with own router, services, types)
│   ├── auth/         # Authentication & authorization
│   ├── users/        # User management
│   ├── tenants/      # Tenant management
│   ├── chats/        # Chat functionality (preserves Markdown)
│   ├── messages/     # Message handling
│   ├── features/     # Feature flags & management
│   └── settings/     # User/tenant settings
├── middleware/       # Shared middleware
│   ├── auth.ts      # JWT/Mock auth extraction
│   ├── rateLimit.ts # Rate limiting
│   └── validate.ts  # Request validation
├── lib/             # Shared utilities
│   ├── prisma.ts    # Database client
│   ├── redis.ts     # Redis client
│   └── ragClient.ts # Python RAG service client
├── types/           # Shared TypeScript types
└── index.ts         # Main entry point
```

## Development

```bash
# Install dependencies
npm install

# Run in development mode with mock auth
npm run dev

# Run tests
npm test

# Build for production
npm run build
```

## Mock Authentication

By default, the API runs with `AUTH_MODE=mock` which automatically provides:
- Mock user: `dev@example.com`
- Mock tenant: `development`
- All features enabled: `narodne-novine`, `financial-reports`
- Full permissions

To test with real JWT:
```bash
AUTH_MODE=jwt npm run dev
```

## API Endpoints

See `docs/2025-09-25_web-api-specification.md` for complete API documentation.

### Core Resources
- `/api/v1/chats` - Chat management
- `/api/v1/chats/{id}/messages` - Messages (with Markdown support)
- `/api/v1/users` - User management
- `/api/v1/tenants` - Tenant administration
- `/api/v1/features` - Feature flags
- `/api/v1/settings` - User/tenant settings

## Important Notes

### Markdown Preservation
The chat/message system preserves **raw Markdown** from LLMs including:
- Headers, bold, italic, links
- Tables and lists
- Code blocks
- Emojis and UTF-8 characters
- All formatting must be preserved as-is

### Multi-tenancy
All operations are automatically scoped to the authenticated user's tenant. The tenant/user context comes from the JWT token, not URL parameters.