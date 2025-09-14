# PyProject configuration optimized for macOS M4 Pro + uv
# This is a template - rename to pyproject.toml if starting fresh

[project]
name = "croatian-rag"
version = "0.1.0"
description = "Croatian RAG system with Apple Silicon optimization"
authors = [
    {name = "RAG Developer", email = "developer@example.com"}
]
dependencies = [
    # Core RAG dependencies
    "torch>=2.0.0",
    "torchvision>=0.15.0", 
    "torchaudio>=2.0.0",
    "sentence-transformers>=2.2.0",
    "chromadb>=0.4.0",
    
    # Document processing
    "PyPDF2>=3.0.0",
    "python-docx>=0.8.11",
    "python-pptx>=0.6.21",
    
    # Web framework
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    
    # Data processing
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    
    # Configuration
    "pydantic>=2.0.0",
    "toml>=0.10.2",
    "pyyaml>=6.0",
    
    # HTTP client
    "requests>=2.31.0",
    "httpx>=0.24.0",
    
    # Development tools
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.13.0",
    "mypy>=1.5.0",
    
    # Type stubs
    "types-requests>=2.31.0",
    "types-PyYAML>=6.0.0",
    "types-toml>=0.10.0",
]

requires-python = ">=3.13"

[project.optional-dependencies]
dev = [
    "jupyter>=1.0.0",
    "ipython>=8.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
# uv-specific configuration for M4 Pro optimization
dev-dependencies = [
    "pytest>=7.4.0",
    "black>=23.0.0", 
    "ruff>=0.13.0",
    "mypy>=1.5.0",
]

# Apple Silicon specific optimizations
[tool.uv.sources]
# Use Apple Silicon optimized wheels when available
torch = { index = "https://download.pytorch.org/whl/cpu" }

[tool.ruff]
line-length = 120
target-version = "py313"
exclude = [
    ".git",
    "__pycache__",
    "build", 
    "dist",
    ".eggs",
    "*.egg-info",
    ".tox",
    ".venv",
    "temp_cache",
    "data",
    "notebooks",
]

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings  
    "F",   # pyflakes
    "I",   # isort
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
]
# M4 Pro optimized ignores: focus on performance over aesthetics
ignore = [
    "E203",  # whitespace before ':' 
    "E501",  # line too long (handled by formatter)
    "W291",  # trailing whitespace
    "W293",  # blank line contains whitespace
    "E711",  # comparison to None (== None vs is None - both work)
    "E712",  # comparison to bool (== True vs is True - both work)
    "B008",  # function call in argument defaults (often intentional)
]

[tool.ruff.lint.isort]
known-first-party = ["src"]
split-on-trailing-comma = false

[tool.ruff.format]
# Apple Silicon friendly formatting
quote-style = "double"
indent-style = "space"
line-ending = "auto"
skip-magic-trailing-comma = true

[tool.mypy]
python_version = "3.13"
warn_return_any = false  # M4 Pro: focus on performance over pedantic checks
warn_unused_configs = false
disallow_untyped_defs = false
check_untyped_defs = true
disable_error_code = "no-any-return,misc,type-abstract"
exclude = [
    "tests/",
    "notebooks/",
    "temp_cache/",
    "data/",
    "src/cli/",  # CLI is non-critical
    "scripts/",  # Development scripts
    "test_*.py",
    "extract_*.py",
    "dependency_analyzer.py",
    "format_code.py",
    "simplified_*.py",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"] 
python_functions = ["test_*"]
addopts = "-v --tb=short"
asyncio_mode = "auto"

# M4 Pro specific: parallel test execution
addopts = "-v --tb=short -n auto"  # Requires pytest-xdist