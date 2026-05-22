# Career Intelligence System

A lightweight, modular, AI-powered career intelligence system designed for local performance (optimized for MacBook Air M5 16GB RAM) and developer maintainability. This project uses the ultra-fast `uv` package manager for virtual environment and dependency management.

---

## 🛠️ Production-Grade Local Setup

### Prerequisites
- Python 3.12
- [uv Package Manager](https://github.com/astral-sh/uv) (Recommended) or standard Python `venv`.

Install `uv` (if not already installed):
```bash
# macOS
curl -LsSf https://astral-sh.uv.io/install.sh | sh
```

### 1. Project Initialization
```bash
# Clone the repository and navigate to the project root
cd career_intelligence

# Copy environment template and fill in secrets (e.g., API keys)
cp .env.example .env
```

### 2. Bootstrapping with `uv`
Run the start script, which automatically resolves dependencies and starts Streamlit in a sandboxed, hot-reloading virtualenv:
```bash
./run.sh
```

Alternatively, manage virtual environments manually:
```bash
# Create local virtual environment
uv venv

# Activate virtualenv
source .venv/bin/activate

# Install dependencies (including developer tools)
uv pip install -e . --all-extras
```

### 3. Pre-Commit Hooks Setup
Configure pre-commit hooks to automate formatting and linting checks before committing changes:
```bash
uv run pre-commit install
```

---

## 🧼 Code Quality & Tooling Configurations

This codebase uses centralized configurations inside `pyproject.toml` to enforce high-quality coding practices:

### Formatter (Black & Ruff)
Format all python files to standard PEP-8 style (100-character line limits):
```bash
# Using Black
uv run black src/ tests/

# Or Ruff Formatting
uv run ruff format
```

### Linter (Ruff)
Run static analysis checks and auto-fix formatting or import sorting violations:
```bash
uv run ruff check src/ tests/ --fix
```

### Static Type Checker (mypy)
Check type safety across source code packages (configured to `strict` mode):
```bash
uv run mypy src/
```

### Test Suite (pytest)
Verify execution integrity and mocks:
```bash
uv run pytest
```

---

## 📂 Codebase Modules & Architecture

*   `src/app/`: The Streamlit multi-page frontend view layer (only interfaces with core modules via public APIs).
*   `src/core/`: Central cross-cutting infrastructure (LiteLLM configuration, SQLite connection pool with WAL mode, Pydantic settings).
*   `src/modules/`: High-cohesion vertical feature modules (`resume`, `interview`, `career_path`).
*   `prompts/`: File-based Jinja2 markdown templates for LLM prompt version control.
*   `tests/`: Isolated pytest test suites with automated AI mocking fixtures.
