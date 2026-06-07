"""
Stage 1: local CLIP-based aesthetic scoring and content clustering.

Runs entirely on device — no API calls. Auto-uses CUDA if available.
Reduces a large image folder to a shortlist of high-quality candidates
spread across content clusters (landscapes, architecture, people, etc.).
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
THUMBNAIL_SIZE = 512

# Prompts that define "good" and "bad" photography for zero-shot scoring
POSITIVE_PROMPTS = [
    "a high quality photograph with excellent composition",
    "a sharp, well-exposed, professional travel photograph",
    "a beautiful landscape photograph",
    "an aesthetically pleasing image with great lighting",
]
NEGATIVE_PROMPTS = [
    "a blurry or out of focus photograph",
    "a dark, underexposed, or overexposed photograph",
    "a low quality, noisy, or poorly composed snapshot",
    "a duplicate or near-identical photo",
]


@dataclass
class ScoredImage:
    path: Path
    aesthetic_score: float
    embedding: np.ndarray
    cluster: int = -1


# ── Shortlist cache ──────────────────────────────────────────────────────────
# Stage 1 can take well over an hour on CPU for large libraries. Caching its
# output means a Stage 2 failure (Ollama down, wrong model, network blip)
# doesn't force a full rescore — the shortlist is reused if the inputs match.

CACHE_PATH = Path("results") / "stage1_cache.json"


def _cache_key(input_dir: str, shortlist_size: int, image_count: int) -> dict:
    return {
        "input_dir": str(Path(input_dir).resolve()),
        "shortlist_size": shortlist_size,
        "image_count": image_count,
    }


def _load_cache(key: dict) -> list[ScoredImage] | None:
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("key") != key:
        return None
    return [
        ScoredImage(
            path=Path(item["path"]),
            aesthetic_score=item["aesthetic_score"],
            embedding=np.empty(0),  # not needed past Stage 1 — clustering is already done
            cluster=item["cluster"],
        )
        for item in data.get("shortlist", [])
    ]


def _save_cache(key: dict, shortlist: list[ScoredImage]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "key": key,
        "shortlist": [
            {"path": str(item.path), "aesthetic_score": item.aesthetic_score, "cluster": item.cluster}
            for item in shortlist
        ],
    }
    CACHE_PATH.write_text(json.dumps(data, indent=2))


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Stage 1] Using GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("[Stage 1] No CUDA GPU detected — using CPU (Stage 1 will take longer)")
    return device


def _load_model(device: torch.device):
    print(f"[Stage 1] Loading CLIP model ({CLIP_MODEL_ID})…")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return model, processor


def _projected_embedding(output) -> torch.Tensor:
    """
    transformers >= 5 wraps get_text_features/get_image_features output in
    BaseModelOutputWithPooling with the projected embedding at .pooler_output;
    older versions return the projected tensor directly. Handle both.
    """
    return output.pooler_output if hasattr(output, "pooler_output") else output


def _embed_text_prompts(
    prompts: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = _projected_embedding(model.get_text_features(**inputs))
    return normalize(features.cpu().numpy())


def _score_and_embed_images(
    paths: list[Path],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    pos_text: np.ndarray,
    neg_text: np.ndarray,
    batch_size: int = 32,
) -> list[ScoredImage]:
    results: list[ScoredImage] = []

    for i in tqdm(range(0, len(paths), batch_size), desc="Scoring images"):
        batch_paths = paths[i : i + batch_size]
        images: list[Image.Image] = []
        valid_paths: list[Path] = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.LANCZOS)
                images.append(img)
                valid_paths.append(p)
            except Exception:
                pass  # skip unreadable files silently

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = _projected_embedding(model.get_image_features(**inputs))

        embeddings = normalize(features.cpu().numpy())  # (B, D)

        # Aesthetic score: mean similarity to positive prompts minus mean to negative
        pos_sim = embeddings @ pos_text.T  # (B, P)
        neg_sim = embeddings @ neg_text.T  # (B, N)
        scores = pos_sim.mean(axis=1) - neg_sim.mean(axis=1)  # (B,)

        for path, emb, score in zip(valid_paths, embeddings, scores):
            results.append(ScoredImage(path=path, aesthetic_score=float(score), embedding=emb))

    return results


def _cluster(scored: list[ScoredImage], n_clusters: int) -> list[ScoredImage]:
    embeddings = np.stack([s.embedding for s in scored])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(embeddings)
    for item, label in zip(scored, labels):
        item.cluster = int(label)
    return scored


def _select_shortlist(
    scored: list[ScoredImage],
    shortlist_size: int,
    n_clusters: int,
) -> list[ScoredImage]:
    """Pick shortlist_size images, taking the top scorers from each cluster proportionally."""
    by_cluster: dict[int, list[ScoredImage]] = {}
    for item in scored:
        by_cluster.setdefault(item.cluster, []).append(item)

    # Sort each cluster by score descending
    for cluster_id in by_cluster:
        by_cluster[cluster_id].sort(key=lambda x: x.aesthetic_score, reverse=True)

    quota_per_cluster = max(1, shortlist_size // n_clusters)
    shortlist: list[ScoredImage] = []

    # First pass: fill quota per cluster
    for cluster_items in by_cluster.values():
        shortlist.extend(cluster_items[:quota_per_cluster])

    # Second pass: fill remaining slots with highest scorers overall
    remaining = shortlist_size - len(shortlist)
    if remaining > 0:
        already_selected = {id(s) for s in shortlist}
        extras = [s for s in scored if id(s) not in already_selected]
        extras.sort(key=lambda x: x.aesthetic_score, reverse=True)
        shortlist.extend(extras[:remaining])

    return shortlist


def run(
    paths: list[Path],
    shortlist_size: int = 150,
    input_dir: str | None = None,
    use_cache: bool = True,
) -> list[ScoredImage]:
    cache_key = None
    if input_dir is not None:
        cache_key = _cache_key(input_dir, shortlist_size, len(paths))
        if use_cache:
            cached = _load_cache(cache_key)
            if cached is not None:
                print(f"[Stage 1] Reusing cached shortlist ({len(cached)} images) — skipping CLIP scoring.")
                print(f"[Stage 1] (Delete {CACHE_PATH} or pass --refresh-cache to rescore from scratch.)")
                return cached

    device = _get_device()
    model, processor = _load_model(device)

    pos_text = _embed_text_prompts(POSITIVE_PROMPTS, model, processor, device)
    neg_text = _embed_text_prompts(NEGATIVE_PROMPTS, model, processor, device)

    print(f"[Stage 1] Scoring {len(paths)} images…")
    scored = _score_and_embed_images(paths, model, processor, device, pos_text, neg_text)

    if not scored:
        raise RuntimeError("No readable images found in the input directory.")

    actual_shortlist = min(shortlist_size, len(scored))
    n_clusters = max(4, min(20, int(math.sqrt(actual_shortlist))))
    print(f"[Stage 1] Clustering into {n_clusters} content groups…")
    scored = _cluster(scored, n_clusters)

    shortlist = _select_shortlist(scored, actual_shortlist, n_clusters)
    print(f"[Stage 1] Shortlisted {len(shortlist)} candidates from {len(scored)} readable images.")

    if cache_key is not None:
        _save_cache(cache_key, shortlist)
        print(f"[Stage 1] Cached shortlist to {CACHE_PATH} for reuse if Stage 2 needs a retry.")

    return shortlist
