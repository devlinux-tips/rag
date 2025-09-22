#!/bin/bash
# SurrealDB Production Demo - Quick Start Script

set -e

echo "🚀 Starting SurrealDB Production Demo..."
echo "======================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available (prefer modern 'docker compose')
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
    echo "✅ Using modern 'docker compose'"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
    echo "✅ Using legacy 'docker-compose'"
else
    echo "❌ Neither 'docker compose' nor 'docker-compose' is available."
    exit 1
fi

echo "✅ Docker is ready"

# Build and start all services
echo "🏗️  Building and starting services..."
$DOCKER_COMPOSE -f docker-compose.production.yml up --build -d

echo "⏳ Waiting for services to start..."
sleep 15

# Check if services are running
echo "🔍 Checking service status..."
$DOCKER_COMPOSE -f docker-compose.production.yml ps

# Test connectivity
echo "🧪 Testing connectivity..."

# Test SurrealDB
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✅ SurrealDB is responding on port 8000"
else
    echo "⚠️  SurrealDB might still be starting..."
fi

# Test Web App
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✅ Web App is responding on port 5000"

    # Show health check details
    echo "📊 Health check details:"
    curl -s http://localhost:5000/health | python3 -m json.tool 2>/dev/null || echo "   (health endpoint available)"
else
    echo "⚠️  Web App might still be starting..."
fi

echo ""
echo "🎉 Production Demo is ready!"
echo "============================"
echo ""
echo "📱 Access the application:"
echo "   🌐 Web App:       http://localhost:5000"
echo "   🗄️  SurrealDB:     http://localhost:8000"
echo ""
echo "📊 API Endpoints:"
echo "   🏢 Companies:     http://localhost:5000/api/companies"
echo "   📈 Statistics:    http://localhost:5000/api/stats"
echo "   📦 Products:      http://localhost:5000/api/products/<company_id>"
echo "   🏥 Health Check:  http://localhost:5000/health"
echo ""
echo "📋 Demo Features:"
echo "   ✅ Production-ready SurrealDB with RocksDB storage"
echo "   ✅ Real data from SurrealDB (no hardcoded fallbacks)"
echo "   ✅ Python SDK integration"
echo "   ✅ Secure authentication"
echo "   ✅ Persistent data storage"
echo ""
echo "🛠️  Add sample products:"
echo "   python3 add_products.py"
echo ""
echo "🛑 To stop the demo:"
echo "   $DOCKER_COMPOSE -f docker-compose.production.yml down"
echo ""
echo "📖 For more details, see README.md"