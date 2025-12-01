#!/usr/bin/env bash
# Simple setup script for collaborators. Creates venv and installs deps.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"

if [ -d ".venv" ]; then
  echo ".venv already exists — activate it with: source .venv/bin/activate"
  exit 0
fi

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install requirements; note: for GPU users, replace `torch` with a CUDA wheel
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  echo "requirements.txt not found; installing minimal deps"
  pip install python-dotenv transformers huggingface-hub
fi

echo "Setup complete. Activate environment with: source .venv/bin/activate"
echo "Run tests: PYTHONPATH=. python tests/test_node_extractor.py"
