# Phoenix LiveView Dockerfile
FROM elixir:1.16-alpine

# Install system dependencies
RUN apk add --no-cache \
    build-base \
    nodejs \
    npm \
    git \
    curl \
    inotify-tools \
    postgresql-client

# Set environment variables
ENV MIX_ENV=dev
ENV PHX_SERVER=true

# Create app directory
WORKDIR /app

# Install hex and rebar
RUN mix local.hex --force && \
    mix local.rebar --force

# Install Phoenix
RUN mix archive.install hex phx_new --force

# Copy mix files
COPY mix.exs mix.lock ./

# Install dependencies
RUN mix deps.get

# Copy application code
COPY . .

# Re-run deps.get after copying all files to fix lock mismatches
RUN mix deps.get

# Install node dependencies and compile assets
RUN if [ -d "assets" ]; then cd assets && npm install; fi
RUN mix assets.deploy

# Compile the application
RUN mix compile

# Expose Phoenix port
EXPOSE 4000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:4000 || exit 1

# Start Phoenix server
CMD ["sh", "-c", "mix deps.get && mix ecto.create && mix ecto.migrate && mix phx.server"]