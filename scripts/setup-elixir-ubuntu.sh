#!/bin/bash

# Elixir + Phoenix Setup for Ubuntu 24.04
# Uses official Erlang Solutions repository for latest stable versions

set -e

echo "🚀 Setting up Elixir development environment on Ubuntu 24.04..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install prerequisites
echo "🔧 Installing prerequisites..."
sudo apt install -y curl wget gnupg2 software-properties-common apt-transport-https ca-certificates

# Add Erlang Solutions repository (official)
echo "📋 Adding Erlang Solutions repository..."
curl -fsSL https://packages.erlang-solutions.com/ubuntu/erlang_solutions.asc | sudo gpg --dearmor -o /usr/share/keyrings/erlang-solutions.gpg

echo "deb [signed-by=/usr/share/keyrings/erlang-solutions.gpg] https://packages.erlang-solutions.com/ubuntu $(lsb_release -cs) contrib" | sudo tee /etc/apt/sources.list.d/erlang-solutions.list

# Update package list
sudo apt update

# Install Erlang/OTP (latest stable)
echo "🔥 Installing Erlang/OTP..."
sudo apt install -y erlang

# Install Elixir (latest stable)
echo "⚡ Installing Elixir..."
sudo apt install -y elixir

# Install development tools
echo "🛠️ Installing development tools..."
sudo apt install -y build-essential autoconf m4 libncurses5-dev libwxgtk3.2-dev libwxgtk-webview3.2-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libssh-dev unixodbc-dev xsltproc fop libxml2-utils libncurses-dev openjdk-11-jdk

# Install PostgreSQL (required for Phoenix)
echo "🐘 Installing PostgreSQL..."
sudo apt install -y postgresql postgresql-contrib

# Start and enable PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Install Node.js (for Phoenix assets)
echo "📦 Installing Node.js via NodeSource..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Install Hex (Elixir package manager)
echo "💎 Installing Hex..."
mix local.hex --force

# Install Phoenix application generator
echo "🔥 Installing Phoenix..."
mix archive.install hex phx_new --force

# Install Rebar3 (Erlang build tool)
echo "🔨 Installing Rebar3..."
mix local.rebar --force

# Setup PostgreSQL for development
echo "🔧 Setting up PostgreSQL..."
sudo -u postgres psql -c "CREATE USER phoenix WITH PASSWORD 'postgres' CREATEDB;"

# Verify installations
echo "✅ Verifying installations..."
echo "Erlang version:"
erl -version
echo ""
echo "Elixir version:"
elixir --version
echo ""
echo "Phoenix version:"
mix phx.new --version
echo ""
echo "Node.js version:"
node --version
echo ""
echo "PostgreSQL version:"
psql --version

echo ""
echo "🎉 Elixir development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Create a new Phoenix project: mix phx.new my_app"
echo "2. Setup database: mix ecto.setup"
echo "3. Start Phoenix server: mix phx.server"
echo ""
echo "📚 Useful commands:"
echo "  - mix phx.new --help        # Phoenix project options"
echo "  - mix deps.get             # Install dependencies"
echo "  - mix ecto.migrate         # Run database migrations"
echo "  - mix test                 # Run tests"
echo "  - iex -S mix               # Interactive Elixir shell"
