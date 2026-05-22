import sys
from pathlib import Path

# Append project root to sys.path to enable absolute imports during testing
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest_plugins = [
    "tests.fixtures.db",
    "tests.fixtures.ai",
    "tests.fixtures.cache",
    "tests.fixtures.scraping_extraction",
]
