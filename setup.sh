#!/usr/bin/env bash
# photo-curator setup
# Run once on any machine: bash setup.sh
# Single requirements.txt works everywhere — modern PyTorch wheels detect
# CUDA at runtime, so no separate CPU/GPU install path is needed.

set -euo pipefail

# Always operate relative to this script's location, regardless of cwd
cd "$(dirname "${BASH_SOURCE[0]}")"

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

# ── GPU detection (informational only — torch picks it up automatically) ────
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    if [ -n "$GPU_NAME" ]; then
        echo "NVIDIA GPU detected: $GPU_NAME — torch will use it automatically."
    fi
else
    echo "No NVIDIA GPU detected — will run on CPU (slower but fully functional)."
fi

# ── Dependencies ─────────────────────────────────────────────────────────────
echo "Installing dependencies (this can take a few minutes)..."
pip install -r requirements.txt --quiet
echo "Install complete."

# ── Ollama check (default Stage 2 backend — local, free) ────────────────────
echo ""
if command -v ollama &>/dev/null; then
    echo "Ollama found."
    if ollama list 2>/dev/null | grep -q "llava"; then
        echo "llava model already pulled."
    else
        echo "Pulling default vision model (llava, ~4.7GB)..."
        ollama pull llava || echo "Could not pull automatically — run 'ollama pull llava' manually later."
    fi
else
    echo "Ollama not found — it's the default (free, local) Stage 2 backend."
    echo "Install it from https://ollama.com, then run:"
    echo "  ollama pull llava"
    echo ""
    echo "(You can skip this if you only plan to use --use-claude or --stage1-only.)"
fi

# ── .env setup (only needed for --use-claude) ────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "Created .env from .env.example."
    echo "(Only needed if you plan to use --use-claude — Ollama needs no API key.)"
    echo ""
    read -r -p "Paste your ANTHROPIC_API_KEY now (or press Enter to skip): " API_KEY
    if [ -n "$API_KEY" ]; then
        sed -i "s|your_key_here|$API_KEY|" .env
        echo "API key saved to .env"
    else
        echo "Skipped. Set ANTHROPIC_API_KEY in .env later if you want to use --use-claude."
    fi
else
    echo ".env already exists — not overwriting."
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "To run (uses local Ollama by default — no API cost):"
echo "  source $VENV_DIR/bin/activate"
echo "  ollama serve &      # if not already running"
echo "  python main.py --input /path/to/photos --location \"Scotland\""
echo ""
echo "To use the Claude API instead (higher quality, costs API credits):"
echo "  python main.py --input /path/to/photos --location \"Scotland\" --use-claude"
echo ""
echo "To test Stage 1 only (no vision model needed):"
echo "  python main.py --input /path/to/photos --location \"Scotland\" --stage1-only"
echo ""
