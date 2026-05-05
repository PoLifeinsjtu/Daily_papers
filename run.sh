#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

REQUIRED_PACKAGES=(
  requests
  PyMuPDF
  pydantic
  langgraph
  langchain-openai
)

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install "${REQUIRED_PACKAGES[@]}"

if [ -z "${SILICONFLOW_API_KEY:-}" ]; then
  echo "Error: SILICONFLOW_API_KEY is required."
  echo "Set it before running, for example:"
  echo "  export SILICONFLOW_API_KEY='your_siliconflow_key'"
  exit 1
fi

if [ -z "${S2_API_KEY:-}" ]; then
  echo "Warning: S2_API_KEY is not set. Semantic Scholar will run without an API key and may be rate limited."
fi

python main_graph.py "$@"
