# Web API with Node 22 for better SSL compatibility
FROM node:22-slim

WORKDIR /app

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY services/web-api/package*.json ./

# Install all dependencies including Prisma
RUN npm ci

# Copy Prisma schema (for future use)
COPY services/web-api/prisma ./prisma

# Skip Prisma generation for now to avoid SSL issues
# RUN npx prisma generate

# Copy source code
COPY services/web-api/src ./src
COPY services/web-api/tsconfig.json .

# Build TypeScript
RUN npm run build

# Create non-root user
RUN addgroup --gid 1001 --system nodejs && \
    adduser --system --uid 1001 nodejs

# Change ownership
RUN chown -R nodejs:nodejs /app
USER nodejs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000/api/v1/health || exit 1

# Run the application
CMD ["npm", "start"]