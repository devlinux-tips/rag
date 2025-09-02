# Python Code Formatting Setup

This project is configured with automatic Python code formatting to ensure consistent code style.

## Tools Configured

1. **Black** - Code formatter with 100 character line length
2. **isort** - Import statement organizer
3. **flake8** - Code linter with relaxed rules for this project
4. **pre-commit** - Git hooks to automatically format code before commits

## Configuration Files

- `pyproject.toml` - Black and isort configuration
- `.flake8` - Flake8 linting rules
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `.vscode/settings.json` - VS Code editor settings for format-on-save

## Usage

### Manual Formatting

Format all Python files:
```bash
python format_code.py
```

Format a specific file:
```bash
./venv/bin/black src/generation/ollama_client.py
./venv/bin/isort src/generation/ollama_client.py
```

### Automatic Formatting

1. **VS Code**: Files are automatically formatted on save
2. **Git commits**: Pre-commit hooks format code automatically
3. **Manual trigger**: Run `python format_code.py`

## Setup

The formatting tools are already configured. If you need to reinstall:

```bash
# Install tools
pip install black isort pre-commit

# Install pre-commit hooks
pre-commit install
```

## Configuration Details

- **Line length**: 100 characters (more suitable for modern displays)
- **Target Python**: 3.12+
- **Import style**: Black-compatible with isort
- **Linting**: Relaxed flake8 rules suitable for this project

## Benefits

- Consistent code style across the project
- Automatic formatting prevents formatting conflicts in git
- Focus on logic rather than style during development
- Editor integration for seamless development experience
