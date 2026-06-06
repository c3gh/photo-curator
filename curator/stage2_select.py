"""
Stage 2: Claude vision final selection.

Sends shortlisted images to Claude in batches, collecting per-image scores
and reasoning. Final pass picks top N with cluster-diversity enforcement.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from curator.stage1_filter import ScoredImage
from curator.utils import batch, image_to_base64

load_dotenv()

VISION_THUMBNAIL_SIZE = 768
BATCH_SIZE = 20  # images per Claude call — stays well within context limits
CLAUDE_MODEL = "claude-opus-4-8"


@dataclass
class RankedImage:
    path: Path
    stage1_score: float
    stage2_score: float
    cluster: int
    reasoning: str
    rank: int = 0
    tags: list[str] = field(default_factory=list)


def _build_batch_prompt(location: str, count: int, total_shortlist: int) -> str:
    return (
        f"You are an expert travel photographer and photo editor selecting the best images "
        f"to represent {location}.\n\n"
        f"I will show you a batch of candidate photographs. For each image, provide:\n"
        f"1. A score from 1–10 (10 = exceptional) considering: composition, technical quality "
        f"(sharpness, exposure, colour), and how well it typifies {location}\n"
        f"2. 2–4 content tags (e.g. landscape, architecture, street, portrait, detail, wildlife, coast)\n"
        f"3. One sentence of reasoning\n\n"
        f"Respond ONLY with a JSON array, one object per image in the order shown:\n"
        f'[{{"index": 0, "score": 8, "tags": ["landscape", "coast"], "reasoning": "..."}}]\n\n'
        f"Be decisive — reserve 9–10 for genuinely outstanding shots."
    )


def _parse_response(text: str, offset: int) -> list[dict]:
    import json
    import re

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group())
        for item in items:
            item["index"] = item.get("index", 0) + offset
        return items
    except json.JSONDecodeError:
        return []


def _call_claude(
    client: anthropic.Anthropic,
    images_b64: list[str | None],
    prompt: str,
    offset: int,
) -> list[dict]:
    content = [{"type": "text", "text": prompt}]

    for i, b64 in enumerate(images_b64):
        if b64 is None:
            continue
        content.append({
            "type": "text",
            "text": f"\nImage {i}:",
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        })

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": content}],
    )
    return _parse_response(response.content[0].text, offset)


def _enforce_variety(
    ranked: list[RankedImage],
    count: int,
    max_per_cluster: int | None,
) -> list[RankedImage]:
    if max_per_cluster is None:
        max_per_cluster = max(2, count // 4)

    selected: list[RankedImage] = []
    cluster_counts: dict[int, int] = {}

    # First pass: take top scorers respecting cluster cap
    for img in sorted(ranked, key=lambda x: x.stage2_score, reverse=True):
        if len(selected) >= count:
            break
        n = cluster_counts.get(img.cluster, 0)
        if n < max_per_cluster:
            selected.append(img)
            cluster_counts[img.cluster] = n + 1

    # Second pass: fill remaining slots if strict cap left us short
    if len(selected) < count:
        already = {id(s) for s in selected}
        extras = [r for r in ranked if id(r) not in already]
        extras.sort(key=lambda x: x.stage2_score, reverse=True)
        selected.extend(extras[: count - len(selected)])

    for i, img in enumerate(selected, 1):
        img.rank = i
    return selected


def run(
    shortlist: list[ScoredImage],
    location: str,
    count: int = 10,
) -> list[RankedImage]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Add it to .env or export it.")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = _build_batch_prompt(location, count, len(shortlist))

    all_results: list[dict] = []
    print(f"[Stage 2] Sending {len(shortlist)} candidates to Claude in batches of {BATCH_SIZE}…")

    for offset, batch_items in enumerate(
        tqdm(list(batch(shortlist, BATCH_SIZE)), desc="Claude batches"),
        start=0,
    ):
        real_offset = offset * BATCH_SIZE
        images_b64 = [
            image_to_base64(item.path, VISION_THUMBNAIL_SIZE) for item in batch_items
        ]
        results = _call_claude(client, images_b64, prompt, real_offset)
        all_results.extend(results)

    # Map results back to ScoredImage objects
    index_to_item = {i: item for i, item in enumerate(shortlist)}
    ranked: list[RankedImage] = []
    seen_indices: set[int] = set()

    for r in all_results:
        idx = r.get("index", -1)
        if idx not in index_to_item or idx in seen_indices:
            continue
        seen_indices.add(idx)
        item = index_to_item[idx]
        ranked.append(RankedImage(
            path=item.path,
            stage1_score=item.aesthetic_score,
            stage2_score=float(r.get("score", 5)),
            cluster=item.cluster,
            reasoning=r.get("reasoning", ""),
            tags=r.get("tags", []),
        ))

    if not ranked:
        raise RuntimeError("Claude returned no usable scores. Check API key and connectivity.")

    selected = _enforce_variety(ranked, count, max_per_cluster=None)
    print(f"[Stage 2] Selected {len(selected)} final images.")
    return selected
