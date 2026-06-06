#!/usr/bin/env bash
# photo-curator setup
# Run once on any machine: bash setup.sh
# Detects NVIDIA GPU automatically and installs the right torch variant.

set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

echo ""
echo "=== photo-curator setup ==="
echo ""

# ── Python check ────────────────────────────────────────────────────────────
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ and re-run."
    exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10+ required (found $PY_VERSION)."
    exit 1
fi

echo "Python $PY_VERSION found."

# ── Virtual environment ──────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists — skipping creation."
else
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── GPU detection ────────────────────────────────────────────────────────────
USE_GPU=false
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    if [ -n "$GPU_NAME" ]; then
        USE_GPU=true
        echo "NVIDIA GPU detected: $GPU_NAME"
    fi
fi

# ── Dependencies ─────────────────────────────────────────────────────────────
if [ "$USE_GPU" = true ]; then
    echo "Installing GPU requirements (CUDA 12.1)..."
    pip install -r requirements-gpu.txt --quiet
    echo "GPU install complete."
else
    echo "No NVIDIA GPU detected — installing CPU requirements."
    echo "(On your desktop, re-run this script to switch to GPU automatically.)"
    pip install -r requirements.txt --quiet
    echo "CPU install complete."
fi

# ── .env setup ───────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "Created .env from .env.example."
    echo ""
    read -r -p "Paste your ANTHROPIC_API_KEY now (or press Enter to set it manually later): " API_KEY
    if [ -n "$API_KEY" ]; then
        sed -i "s|your_key_here|$API_KEY|" .env
        echo "API key saved to .env"
    else
        echo "Skipped. Open .env and set ANTHROPIC_API_KEY before running Stage 2."
    fi
else
    echo ".env already exists — not overwriting."
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "To run:"
echo "  source $VENV_DIR/bin/activate"
echo "  python main.py --input /path/to/photos --location \"Scotland\""
echo ""
echo "To test Stage 1 only (no API key needed):"
echo "  python main.py --input /path/to/photos --location \"Scotland\" --stage1-only"
echo ""
