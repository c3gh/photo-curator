from dataclasses import dataclass
from pathlib import Path

from curator import stage1_filter, stage2_select
from curator.utils import collect_images


@dataclass
class PipelineResult:
    shortlist: list[stage1_filter.ScoredImage]
    selected: list[stage2_select.RankedImage]


def run(
    input_dir: str,
    location: str,
    count: int = 10,
    shortlist_size: int = 150,
    use_claude: bool = False,
    model: str | None = None,
    refresh_cache: bool = False,
) -> PipelineResult:
    backend = "Claude API" if use_claude else "Ollama (local)"
    print(f"\n=== photo-curator | {location} | top {count} | Stage 2: {backend} ===\n")

    paths = collect_images(input_dir)
    if not paths:
        raise RuntimeError(f"No supported images found in: {input_dir}")
    print(f"Found {len(paths)} images in {input_dir}")

    # Fail fast — check Stage 2's backend BEFORE the long Stage 1 CLIP pass,
    # not after (Stage 1 alone can take well over an hour on CPU for large libraries).
    resolved_model = stage2_select.resolve_model(use_claude, model)
    stage2_select.preflight_check(use_claude, resolved_model)

    shortlist = stage1_filter.run(
        paths,
        shortlist_size=shortlist_size,
        input_dir=input_dir,
        use_cache=not refresh_cache,
    )
    selected = stage2_select.run(
        shortlist,
        location=location,
        count=count,
        use_claude=use_claude,
        model=model,
    )

    return PipelineResult(shortlist=shortlist, selected=selected)
