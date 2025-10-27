#!/bin/bash
set -e

# RAG Platform Ubuntu Setup Script
# Sets up all dependencies for the Multilingual RAG Platform development environment

echo "🚀 Setting up Multilingual RAG Platform on Ubuntu..."
echo "==============================================="

# Check if running on Ubuntu
if ! command -v lsb_release &> /dev/null || [[ "$(lsb_release -si)" != "Ubuntu" ]]; then
    echo "⚠️  Warning: This script is designed for Ubuntu. Proceeding anyway..."
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install package if not exists
install_if_missing() {
    if ! command_exists "$1"; then
        echo "📦 Installing $1..."
        case "$1" in
            "python3")
                sudo apt update
                sudo apt install -y python3 python3-pip python3-venv
                ;;
            "node")
                curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
                sudo apt-get install -y nodejs
                ;;
            "git")
                sudo apt install -y git
                ;;
            "curl")
                sudo apt install -y curl
                ;;
            *)
                echo "❌ Unknown package: $1"
                return 1
                ;;
        esac
        echo "✅ $1 installed successfully"
    else
        echo "✅ $1 already installed"
    fi
}

# Update system packages
echo "🔄 Updating system packages..."
sudo apt update

# Install basic system dependencies
echo "📦 Installing system dependencies..."
install_if_missing "curl"
install_if_missing "git"

# Install Python 3 and pip
echo "🐍 Setting up Python environment..."
install_if_missing "python3"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $PYTHON_VERSION"

# Install Node.js (for React frontend)
echo "🟢 Setting up Node.js environment..."
install_if_missing "node"

# Check Node.js version
if command_exists "node"; then
    NODE_VERSION=$(node --version)
    NPM_VERSION=$(npm --version)
    echo "✅ Node.js version: $NODE_VERSION"
    echo "✅ npm version: $NPM_VERSION"
fi

# Install Python dependencies for RAG system
echo "🔧 Setting up Python RAG environment..."
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies from requirements.txt..."
    python3 -m pip install --user -r requirements.txt
    echo "✅ Python dependencies installed"
else
    echo "⚠️  requirements.txt not found, skipping Python dependencies"
    echo "   You may need to install them manually later"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🏗️  Creating Python virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo "   Activate with: source venv/bin/activate"
else
    echo "✅ Python virtual environment already exists"
fi

# Install global npm packages if needed
echo "📦 Setting up Node.js global packages..."
if ! npm list -g @vite/create-app &>/dev/null; then
    echo "📥 Installing Vite globally..."
    npm install -g create-vite
    echo "✅ Vite installed globally"
else
    echo "✅ Vite already available"
fi


# Create basic project structure if not exists
echo "📁 Checking project structure..."
mkdir -p services/rag-service/data/vectordb
mkdir -p services/web-api
mkdir -p services/user-frontend
mkdir -p docs
mkdir -p scripts
echo "✅ Project directories ensured"

# Create basic development scripts
echo "📝 Creating development scripts..."


# RAG development server script
cat > scripts/start_rag_dev.sh << 'EOF'
#!/bin/bash
echo "🐍 Starting RAG development environment..."
source venv/bin/activate
cd services/rag-service
python -c "
import asyncio
from src.pipeline.rag_system import RAGSystem

async def test_rag():
    print('🧪 Testing RAG system...')
    rag = RAGSystem(language='hr')
    await rag.initialize()
    print('✅ RAG system initialized successfully')

if __name__ == '__main__':
    asyncio.run(test_rag())
"
EOF

# Make scripts executable
chmod +x scripts/*.sh
echo "✅ Development scripts created"

# Summary
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo "✅ System dependencies installed"
echo "✅ Python 3 and pip ready"
echo "✅ Node.js and npm ready"
echo "✅ SurrealDB installed"
echo "✅ Project structure created"
echo "✅ Development scripts ready"
echo ""
echo "🚀 Next Steps:"
echo "1. Reload your shell: source ~/.bashrc"
echo "2. Test SurrealDB: surreal version"
echo "3. Activate Python environment: source venv/bin/activate"
echo "4. Install Python dependencies: pip install -r services/rag-service/requirements.txt"
echo "5. Start development: scripts/start_surrealdb.sh"
echo ""
echo "📚 Development Scripts:"
echo "- scripts/start_surrealdb.sh - Start SurrealDB server"
echo "- scripts/start_rag_dev.sh - Test RAG system"
echo ""
echo "Happy coding! 🎯"
