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

### Stage 2 — Claude vision final selection (API, ~$1–3 per 6k run)
- Resize shortlist to 768px for API submission
- Send batches of 20 images to Claude with location context and variety instruction
- Collect per-image scores + reasoning
- Final elimination: pick top N ensuring no cluster is over-represented

## Constraints
- `ANTHROPIC_API_KEY` via env var or `.env` — never hardcoded
- No GPU required — auto-detects CUDA with `torch.cuda.is_available()`
- No internet required for Stage 1 after first model download
- Must handle corrupt/unreadable image files gracefully
- Output is a JSON file + optional file copy — no database

## Out of Scope
- Web UI (CLI only for now)
- RAW file support (JPEG/PNG/WEBP/HEIC only)
- Video files
- Cloud storage input (local paths only)
