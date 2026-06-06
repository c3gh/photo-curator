# photo-curator

## Project Context
Two-stage AI photo curation pipeline. Stage 1 runs locally via CLIP (no API cost). Stage 2 uses Claude vision API for final selection. See `project_specifications.md` for full architecture.

## Key Decisions
- **CLIP model:** `openai/clip-vit-large-patch14` — best quality/speed tradeoff for aesthetic scoring
- **Aesthetic scoring:** zero-shot CLIP similarity against curated text prompts (no separate model to download)
- **Clustering:** KMeans on CLIP embeddings — k auto-set to ~sqrt(shortlist_size)
- **GPU handling:** `torch.cuda.is_available()` — no code changes needed between laptop and desktop
- **Thumbnail size:** 512px Stage 1, 768px Stage 2 — balances speed vs. Claude's visual detail needs
- **Output:** `results/selection.json` with paths, scores, cluster labels, and Claude's reasoning per image

## Key File Paths
- `main.py` — CLI entrypoint
- `curator/stage1_filter.py` — CLIP embedding, aesthetic scoring, clustering
- `curator/stage2_select.py` — Claude vision API calls and final ranking
- `curator/pipeline.py` — orchestrates stage 1 → stage 2
- `curator/utils.py` — image loading, resizing, batching helpers

## Dev Setup
```bash
python -m venv .venv
source .venv/bin/activate

# CPU (laptop):
pip install -r requirements.txt

# GPU (desktop with CUDA 12.1):
pip install -r requirements-gpu.txt

cp .env.example .env
# Set ANTHROPIC_API_KEY in .env
```

## Running
```bash
python main.py --input /path/to/photos --location "Scotland" --count 10
```
