# Project Specifications — photo-curator

## Objective
AI-powered pipeline that takes a large folder of photos (up to 6,000+) and returns the top N images that best represent a location, balancing composition quality, technical quality, and content variety. Runs a fast local stage to pre-filter, then a Claude vision stage for nuanced final selection.

## Context
Born out of need to curate travel photography for suffredini.me. Primary use case: select the best 10 photos representing a country/location from a full shoot library. Must work on CPU-only hardware (laptop) and scale to GPU (NVIDIA desktop) without code changes.

## Architecture

### Stage 1 — Local pre-filter (free, runs on device)
- Load all images as thumbnails (512px longest edge) for speed
- Embed all images with CLIP (`openai/clip-vit-large-patch14`)
- Score aesthetics via CLIP zero-shot: cosine similarity against good/bad quality text prompts
- Cluster embeddings with KMeans to identify content categories (landscapes, architecture, people, details, etc.)
- Output: ranked shortlist of ~100–200 candidates, preserving cluster proportions
- Result is cached to `results/stage1_cache.json` (keyed on input dir + shortlist size + image count) — Stage 1 can take well over an hour on CPU for large libraries, so a matching re-run reuses the cached shortlist instead of rescoring from scratch. `--refresh-cache` forces a fresh pass.

### Stage 2 — Vision model final selection (Ollama by default, free; Claude API opt-in via `--use-claude`)
- Resize shortlist to 768px for vision submission
- Send batches of images (4 for Ollama, 20 for Claude) with location context and variety instruction
- Collect per-image scores + reasoning
- Final elimination: pick top N ensuring no cluster is over-represented
- Before Stage 1 runs, a preflight check verifies the Stage 2 backend is reachable (Ollama running + model pulled, or `ANTHROPIC_API_KEY` set) — fails fast rather than after a long Stage 1 pass

## Constraints
- `ANTHROPIC_API_KEY` via env var or `.env` — never hardcoded; only required for `--use-claude`
- No GPU required — auto-detects CUDA with `torch.cuda.is_available()`
- No internet required after first-time model downloads (CLIP weights, Ollama model pull)
- Must handle corrupt/unreadable image files gracefully
- Output is a JSON file + optional file copy (`--copy-to`, which also copies the full Stage 1 shortlist into a `shortlist/` subfolder) — no database

## Out of Scope
- Web UI (CLI only for now)
- RAW file support (JPEG/PNG/WEBP/HEIC only)
- Video files
- Cloud storage input (local paths only)
