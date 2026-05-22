#!/usr/bin/env bash
# Helper script to bootstrap, run migrations, and launch Streamlit frontend using uv.

set -e

# Setup .env if it does not exist
if [ ! -f .env ]; then
  echo "Copying .env.example to .env..."
  cp .env.example .env
fi

# Ensure data directory exists
mkdir -p data/uploads

# Detect virtual environment or uv
if command -v uv &> /dev/null; then
  echo "🚀 Running with uv..."
  # uv run automatically handles venv creation and dependency sync
  uv run streamlit run src/app/main.py
elif [ -d ".venv" ]; then
  echo "Running with existing .venv..."
  source .venv/bin/activate
  streamlit run src/app/main.py
elif [ -d "venv" ]; then
  echo "Running with existing venv..."
  source venv/bin/activate
  streamlit run src/app/main.py
else
  echo "No virtualenv or uv detected. Attempting to install virtualenv..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e .
  streamlit run src/app/main.py
fi
