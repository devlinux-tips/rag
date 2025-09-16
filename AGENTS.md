# Repository Guidelines

## Project Structure & Module Organization
- Core service: `services/rag-service` (Python 3.12). Source under `src/` with domains: `utils/`, `preprocessing/`, `retrieval/`, `vectordb/`, `generation/`, `pipeline/`, `models/`, and CLI in `src/cli/`.
- Tests: `services/rag-service/tests` (fixtures in `tests/fixtures`).
- Config and data: `services/rag-service/config/*.toml`, `services/rag-service/data/`.
- Tooling: `Makefile`, `.pre-commit-config.yaml`, `services/rag-service/pyproject.toml`, formatting script `services/rag-service/format_code.py`.
- Docs and plans: `docs/`, `planning/`, prototypes in `prototypes/`.

## Build, Test, and Development Commands
- Install deps: `python -m pip install -r services/rag-service/requirements.txt`
- Format + lint: `cd services/rag-service && python format_code.py` (pre-commit is check-only)
- Run tests: `cd services/rag-service && pytest -v` or `python test_runner.py --cov -v`
- CLI entry point: `cd services/rag-service && python rag.py --help` (use `rag.py`, not module paths)
- Monorepo helpers: `make setup | test | build | clean | rag-dev` (some targets assume placeholder services).

## Coding Style & Naming Conventions
- Python 3.12, 4-space indent, 120-char lines.
- Ruff is the canonical linter/formatter; Black optional; MyPy enabled for core `src/*` (CLI excluded).
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Keep changes small and testable within domain folders (e.g., `src/retrieval/`).

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio` (auto mode). Patterns: files `test_*.py`, classes `Test*`, functions `test_*`.
- Place unit tests next to feature domain (under `services/rag-service/tests`). Use fixtures from `tests/fixtures`.
- Aim for meaningful coverage on new/changed code; run `pytest -v` locally before PR.

## Commit & Pull Request Guidelines
- Prefer Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`) with an imperative subject ≤ 72 chars.
- PRs include: clear description, impacted paths, linked issues, sample commands/output, and doc updates.
- Run `format_code.py` and ensure tests pass locally before review; pre-commit hooks must be green.

## Security & Configuration Tips
- Do not commit secrets, models, or large datasets.
- Config is TOML under `services/rag-service/config/*.toml`. GPU/LLM/DB setup: see `docs/pytorch_cuda_setup.md` and `docs/surrealdb-setup.md`.

## Architecture & Quality Principles (from AI_INSTRUCTIONS.md, CLAUDE.md)
- Fail-fast: validate config at startup; access with `config["key"]` after validation. No `.get()` defaults, no silent fallbacks, no hard-coded defaults.
- Consistency > backward compatibility: apply one pattern repo-wide; avoid half-refactors.
- Multilingual: language equality; avoid language-specific comments or logic. Always pass explicit `--language`/`language`.
- Dependency injection, pure functions, protocol-based interfaces; keep business logic separate from I/O.

## Agent-Specific Instructions
- Ask before adding new config values or hard-coded constants. System should fail if required config is missing.
- Use `services/rag-service/format_code.py` before committing; pre-commit hooks are check-only and mirror its scope (core `src/*` subpackages).
- Only touch CLI (`rag.py`, `src/cli/*`) when explicitly required; main entry is `python rag.py`.
