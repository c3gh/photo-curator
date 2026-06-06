from pathlib import Path

from curator import stage1_filter, stage2_select
from curator.utils import collect_images


def run(
    input_dir: str,
    location: str,
    count: int = 10,
    shortlist_size: int = 150,
    use_claude: bool = False,
    model: str | None = None,
) -> list[stage2_select.RankedImage]:
    backend = "Claude API" if use_claude else "Ollama (local)"
    print(f"\n=== photo-curator | {location} | top {count} | Stage 2: {backend} ===\n")

    paths = collect_images(input_dir)
    if not paths:
        raise RuntimeError(f"No supported images found in: {input_dir}")
    print(f"Found {len(paths)} images in {input_dir}")

    shortlist = stage1_filter.run(paths, shortlist_size=shortlist_size)
    selected = stage2_select.run(
        shortlist,
        location=location,
        count=count,
        use_claude=use_claude,
        model=model,
    )

    return selected
