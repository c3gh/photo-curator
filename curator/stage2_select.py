"""
Stage 2: vision model final selection.

Default backend: Ollama (local, free, offline after model download).
Optional backend: Claude API (--use-claude flag, requires ANTHROPIC_API_KEY).

Both backends use the same prompt structure and return the same RankedImage output.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from curator.stage1_filter import ScoredImage
from curator.utils import batch, image_to_base64

load_dotenv()

OLLAMA_DEFAULT_MODEL = "llava"
CLAUDE_DEFAULT_MODEL = "claude-sonnet-4-6"

OLLAMA_BATCH_SIZE = 4   # local models have smaller context windows
CLAUDE_BATCH_SIZE = 20

VISION_THUMBNAIL_SIZE = 768


@dataclass
class RankedImage:
    path: Path
    stage1_score: float
    stage2_score: float
    cluster: int
    reasoning: str
    rank: int = 0
    tags: list[str] = field(default_factory=list)


# ── Shared prompt ────────────────────────────────────────────────────────────

def _build_prompt(location: str, n_images: int) -> str:
    return (
        f"You are an expert travel photographer selecting the best images to represent {location}.\n\n"
        f"I will show you {n_images} candidate photograph(s). For each image, provide:\n"
        f"1. A score from 1–10 (10 = exceptional) considering: composition, technical quality "
        f"(sharpness, exposure, colour), and how well it typifies {location}\n"
        f"2. 2–4 content tags (e.g. landscape, architecture, street, portrait, detail, wildlife, coast)\n"
        f"3. One sentence of reasoning\n\n"
        f"Respond ONLY with a JSON array, one object per image in the order shown:\n"
        f'[{{"index": 0, "score": 8, "tags": ["landscape", "coast"], "reasoning": "..."}}]\n\n'
        f"Be decisive — reserve 9–10 for genuinely outstanding shots."
    )


def _parse_json_response(text: str, offset: int) -> list[dict]:
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group())
        for item in items:
            item["index"] = item.get("index", 0) + offset
        return items
    except json.JSONDecodeError:
        return []


# ── Ollama backend ───────────────────────────────────────────────────────────

def _score_batch_ollama(
    items: list[ScoredImage],
    offset: int,
    location: str,
    model: str,
) -> list[dict]:
    try:
        import ollama
    except ImportError:
        raise ImportError("ollama package not installed. Run: pip install ollama==0.2.1")

    images_b64 = [image_to_base64(item.path, VISION_THUMBNAIL_SIZE) for item in items]
    valid_b64 = [b for b in images_b64 if b is not None]

    if not valid_b64:
        return []

    prompt = _build_prompt(location, len(valid_b64))

    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": valid_b64,
            }],
        )
    except Exception as e:
        err = str(e)
        if "connection" in err.lower() or "refused" in err.lower():
            raise RuntimeError(
                "Cannot reach Ollama. Start it with: ollama serve\n"
                f"Then ensure the model is pulled: ollama pull {model}"
            )
        if "model" in err.lower() and "not found" in err.lower():
            raise RuntimeError(f"Model not found. Pull it with: ollama pull {model}")
        raise

    return _parse_json_response(response["message"]["content"], offset)


def _run_ollama(
    shortlist: list[ScoredImage],
    location: str,
    model: str,
) -> list[dict]:
    print(f"[Stage 2] Using Ollama ({model}) — local, no API cost.")
    print(f"[Stage 2] Scoring {len(shortlist)} candidates in batches of {OLLAMA_BATCH_SIZE}…")
    print(f"[Stage 2] Tip: ensure Ollama is running (ollama serve) and model is pulled (ollama pull {model})")

    all_results: list[dict] = []
    batches = list(batch(shortlist, OLLAMA_BATCH_SIZE))

    for i, batch_items in enumerate(tqdm(batches, desc="Ollama batches")):
        offset = i * OLLAMA_BATCH_SIZE
        results = _score_batch_ollama(batch_items, offset, location, model)
        all_results.extend(results)

    return all_results


# ── Claude backend ───────────────────────────────────────────────────────────

def _score_batch_claude(
    items: list[ScoredImage],
    offset: int,
    location: str,
    model: str,
    client,
) -> list[dict]:
    import anthropic

    images_b64 = [image_to_base64(item.path, VISION_THUMBNAIL_SIZE) for item in items]
    prompt = _build_prompt(location, len(items))

    content = [{"type": "text", "text": prompt}]
    for i, b64 in enumerate(images_b64):
        if b64 is None:
            continue
        content.append({"type": "text", "text": f"\nImage {i}:"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": content}],
    )
    return _parse_json_response(response.content[0].text, offset)


def _run_claude(
    shortlist: list[ScoredImage],
    location: str,
    model: str,
) -> list[dict]:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "Add it to .env or export it. Note: requires separate API credits at console.anthropic.com "
            "(not included in Claude Pro/Max)."
        )

    print(f"[Stage 2] Using Claude API ({model}).")
    print(f"[Stage 2] Scoring {len(shortlist)} candidates in batches of {CLAUDE_BATCH_SIZE}…")

    client = anthropic.Anthropic(api_key=api_key)
    all_results: list[dict] = []
    batches = list(batch(shortlist, CLAUDE_BATCH_SIZE))

    for i, batch_items in enumerate(tqdm(batches, desc="Claude batches")):
        offset = i * CLAUDE_BATCH_SIZE
        results = _score_batch_claude(batch_items, offset, location, model, client)
        all_results.extend(results)

    return all_results


# ── Variety enforcement ──────────────────────────────────────────────────────

def _enforce_variety(
    ranked: list[RankedImage],
    count: int,
) -> list[RankedImage]:
    max_per_cluster = max(2, count // 4)
    selected: list[RankedImage] = []
    cluster_counts: dict[int, int] = {}

    for img in sorted(ranked, key=lambda x: x.stage2_score, reverse=True):
        if len(selected) >= count:
            break
        n = cluster_counts.get(img.cluster, 0)
        if n < max_per_cluster:
            selected.append(img)
            cluster_counts[img.cluster] = n + 1

    if len(selected) < count:
        already = {id(s) for s in selected}
        extras = sorted(
            [r for r in ranked if id(r) not in already],
            key=lambda x: x.stage2_score,
            reverse=True,
        )
        selected.extend(extras[: count - len(selected)])

    for i, img in enumerate(selected, 1):
        img.rank = i
    return selected


# ── Public entry point ───────────────────────────────────────────────────────

def run(
    shortlist: list[ScoredImage],
    location: str,
    count: int = 10,
    use_claude: bool = False,
    model: str | None = None,
) -> list[RankedImage]:
    resolved_model = model or (CLAUDE_DEFAULT_MODEL if use_claude else OLLAMA_DEFAULT_MODEL)

    raw_results = (
        _run_claude(shortlist, location, resolved_model)
        if use_claude
        else _run_ollama(shortlist, location, resolved_model)
    )

    if not raw_results:
        raise RuntimeError("Stage 2 returned no scores. Check that the vision model is working.")

    index_to_item = {i: item for i, item in enumerate(shortlist)}
    ranked: list[RankedImage] = []
    seen: set[int] = set()

    for r in raw_results:
        idx = r.get("index", -1)
        if idx not in index_to_item or idx in seen:
            continue
        seen.add(idx)
        item = index_to_item[idx]
        ranked.append(RankedImage(
            path=item.path,
            stage1_score=item.aesthetic_score,
            stage2_score=float(r.get("score", 5)),
            cluster=item.cluster,
            reasoning=r.get("reasoning", ""),
            tags=r.get("tags", []),
        ))

    selected = _enforce_variety(ranked, count)
    print(f"[Stage 2] Selected {len(selected)} final images.")
    return selected
