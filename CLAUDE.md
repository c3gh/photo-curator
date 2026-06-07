# photo-curator

## Project Context
Two-stage AI photo curation pipeline. Stage 1 runs locally via CLIP (free). Stage 2 defaults to a local Ollama vision model (`llava`, free); the Claude API is available as an opt-in via `--use-claude` for a higher-fidelity pass. See `project_specifications.md` for full architecture.

## Key Decisions
- **CLIP model:** `openai/clip-vit-large-patch14` — best quality/speed tradeoff for aesthetic scoring
- **Aesthetic scoring:** zero-shot CLIP similarity against curated text prompts (no separate model to download)
- **Clustering:** KMeans on CLIP embeddings — k auto-set to ~sqrt(shortlist_size)
- **Stage 2 default backend: Ollama (`llava`)** — Anthropic API requires separate paid credits (not covered by Claude Pro/Max); a local vision model is good enough for bulk curation and keeps the whole pipeline free and offline
- **Stage 2 opt-in backend: Claude API** (`--use-claude`) — for when a higher-quality reasoning pass is worth the small API cost
- **Single requirements.txt** — modern PyTorch wheels detect CUDA at runtime (`torch.cuda.is_available()`); no separate CPU/GPU install paths needed
- **Thumbnail size:** 512px Stage 1, 768px Stage 2 — balances speed vs. visual detail needs
- **Stage 1 result caching** (`results/stage1_cache.json`, keyed on input dir + shortlist size + image count) — Stage 1 can take well over an hour on CPU for large libraries; caching its output means a Stage 2 failure (Ollama down, bad model name, etc.) doesn't force a full rescore. `--refresh-cache` bypasses it.
- **Stage 2 preflight check** — runs before the long Stage 1 pass to verify Ollama is reachable (or `ANTHROPIC_API_KEY` is set for `--use-claude`), so a misconfigured backend fails fast instead of after an hour of scoring
- **Output:** `results/selection.json` with paths, scores, cluster labels, tags, and reasoning per image. With `--copy-to`, final picks are copied to the target dir and the full Stage 1 shortlist is copied into a `shortlist/` subfolder for browsing

## Key File Paths
- `main.py` — CLI entrypoint (`--use-claude`, `--model`, `--stage1-only`, etc.)
- `curator/stage1_filter.py` — CLIP embedding, aesthetic scoring, clustering
- `curator/stage2_select.py` — dual-backend vision scoring (Ollama default, Claude optional)
- `curator/pipeline.py` — orchestrates stage 1 → stage 2
- `curator/utils.py` — image loading, resizing, batching helpers
- `setup.sh` — cross-machine installer (creates venv, installs deps, sets up Ollama + `.env`)

## Dev Setup
```bash
bash setup.sh
```
Or manually:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install Ollama from https://ollama.com, then:
ollama pull llava

# Only needed for --use-claude:
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env
```

## Running
```bash
# Default — local Ollama backend, no API cost
python main.py --input /path/to/photos --location "Scotland" --count 10

# Opt into Claude API for Stage 2
python main.py --input /path/to/photos --location "Scotland" --use-claude

# Force a full Stage 1 rescore, ignoring the cached shortlist
python main.py --input /path/to/photos --location "Scotland" --refresh-cache
```
