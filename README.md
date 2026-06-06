# photo-curator

AI-powered photo selection pipeline. Point it at a folder of thousands of photos and get back the best N images that represent a location — balanced for composition quality, technical quality, and content variety.

## How it works

**Stage 1 (local, free):** CLIP embeddings score every image for aesthetic quality and cluster them by content type. 6,000 photos → ~150 shortlisted candidates in a few minutes.

**Stage 2 (Claude API, ~$1–3):** The shortlist goes to Claude vision, which makes the final selection with an eye for variety and location typicality. Returns ranked picks with reasoning.

Automatically uses your NVIDIA GPU if one is available; falls back to CPU otherwise.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

# CPU (laptop):
pip install -r requirements.txt

# GPU (NVIDIA desktop, CUDA 12.1):
pip install -r requirements-gpu.txt

cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Usage

```bash
# Basic — pick top 10 from a folder
python main.py --input /path/to/photos --location "Scotland"

# Custom count and shortlist size
python main.py --input /path/to/photos --location "Japan" --count 20 --shortlist 300

# Copy selected files to an output directory
python main.py --input /path/to/photos --location "Scotland" --copy-to ./selected
```

Results are written to `results/selection.json` with paths, scores, content clusters, and Claude's reasoning for each pick.

## Supported formats

JPEG, PNG, WEBP, HEIC
