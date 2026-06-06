# photo-curator

AI-powered photo selection pipeline. Point it at a folder of thousands of photos and get back the best N images that represent a location — balanced for composition quality, technical quality, and content variety.

## How it works

**Stage 1 (local, free):** CLIP embeddings score every image for aesthetic quality and cluster them by content type. 6,000 photos → ~150 shortlisted candidates in a few minutes.

**Stage 2 (local by default, free):** The shortlist goes to a local vision model via [Ollama](https://ollama.com) (`llava`), which makes the final selection with an eye for variety and location typicality. Returns ranked picks with reasoning. Optionally route this stage through the Claude API instead with `--use-claude` (requires separate API credits — see below).

Automatically uses your NVIDIA GPU if one is available; falls back to CPU otherwise.

## Setup

```bash
git clone https://github.com/c3gh/photo-curator
cd photo-curator
bash setup.sh
```

`setup.sh` creates a virtualenv, installs dependencies, checks for Ollama, and pulls the default vision model (`llava`, ~4.7GB). One requirements file works on both a CPU laptop and a GPU desktop — modern PyTorch wheels detect CUDA at runtime (`torch.cuda.is_available()`), so there's nothing to switch manually. Re-run it on a different machine and it just works.

Manual setup:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install Ollama from https://ollama.com, then:
ollama pull llava
```

## Usage

```bash
# Basic — pick top 10 from a folder, scored locally via Ollama (no API cost)
python main.py --input /path/to/photos --location "Scotland"

# Custom count and shortlist size
python main.py --input /path/to/photos --location "Japan" --count 20 --shortlist 300

# Copy selected files to an output directory
python main.py --input /path/to/photos --location "Scotland" --copy-to ./selected

# Stage 1 only — quality-ranked shortlist, no vision model needed
python main.py --input /path/to/photos --location "Scotland" --stage1-only
```

Results are written to `results/selection.json` with paths, scores, content clusters, and reasoning for each pick.

## Using the Claude API instead of Ollama

By default, Stage 2 runs entirely locally via Ollama — no cost, no account, works offline. If you want a higher-quality pass, you can route Stage 2 through the Claude API:

```bash
python main.py --input /path/to/photos --location "Scotland" --use-claude
python main.py --input /path/to/photos --location "Scotland" --use-claude --model claude-opus-4-8
```

**Note:** the Claude API is a separate product from a Claude Pro/Max subscription. It requires its own account and pre-purchased credits at [console.anthropic.com](https://console.anthropic.com) (set `ANTHROPIC_API_KEY` in `.env`). Cost is roughly $0.20–0.50 per 6,000-photo run with Sonnet.

## Supported formats

JPEG, PNG, WEBP, HEIC
