# Career Intelligence Testing Architecture Guideline

This directory contains the automated test suites for the AI-powered Career Intelligence application. We use `pytest` as our testing framework, extended by `pytest-asyncio` for async integration.

## 📁 Folder Structure

```
tests/
├── README.md                 # This testing guide
├── conftest.py               # Main pytest runner bootstrap and registration
├── fixtures/                 # Domain-specific reusable test fixtures
│   ├── __init__.py
│   ├── ai.py                 # LLM and service mocks
│   ├── cache.py              # SQLite cache backend configurations
│   └── db.py                 # SQLite transaction rollback fixtures
├── utils/                    # Common helper utilities
│   ├── __init__.py
│   └── helpers.py            # JSON generators and verification models
├── unit/                     # Isolated unit tests
│   ├── test_ai_mocks.py      # LLM mocks and response routing validations
│   ├── test_cache.py         # Key-value caching layer tests
│   └── ...
└── integration/              # Component integration tests
    ├── test_ai_caching.py    # Integrates LLM Client, prompts, and cache manager
    └── ...
```

## 🛠️ How to Run Tests

All tests should be run through the local python virtual environment wrapper.

### 1. Run the Entire Test Suite
```bash
.venv/bin/pytest -v
```

### 2. Run Specific Subdirectories
* Run unit tests:
  ```bash
  .venv/bin/pytest tests/unit/
  ```
* Run integration tests:
  ```bash
  .venv/bin/pytest tests/integration/
  ```

### 3. Run a Specific File
```bash
.venv/bin/pytest tests/core/test_prompts.py
```

### 4. Run Filtered Tests by Name Expression
```bash
.venv/bin/pytest -k "test_ttl"
```

## 💡 Best Practices & Guidelines

### 1. Database Isolation (Transactional Rollbacks)
Always use the `db_session` fixture. It runs each test case in a nested transaction (`connection.begin()`), executing rollbacks immediately at teardown. This avoids costly database creation operations, enabling tests to execute in milliseconds.

### 2. Async Testing
For async coroutines, decorate your test functions with `@pytest.mark.asyncio`:
```python
@pytest.mark.asyncio
async def test_async_service(mock_ai_client):
    res = await mock_ai_client.generate("hello")
    assert res == "Mock response"
```

### 3. Isolated Mocking (Boundary Rule)
Do not make external HTTP/API requests in tests. Use `MockLLMClient` or monkeypatch external modules (e.g. `litellm`) to simulate network conditions. Mock at the interface boundary of the service rather than deeply nesting mocks inside internal implementations.

### 4. Code Compliance
Ensure all new tests and modified files comply with strict formatting, style rules, and type annotations before pushing:
```bash
.venv/bin/ruff check src tests
.venv/bin/black --check src tests
.venv/bin/mypy src
```
