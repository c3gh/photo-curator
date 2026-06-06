from pathlib import Path

from curator import stage1_filter, stage2_select
from curator.utils import collect_images


def run(
    input_dir: str,
    location: str,
    count: int = 10,
    shortlist_size: int = 150,
) -> list[stage2_select.RankedImage]:
    print(f"\n=== photo-curator | {location} | top {count} ===\n")

    paths = collect_images(input_dir)
    if not paths:
        raise RuntimeError(f"No supported images found in: {input_dir}")
    print(f"Found {len(paths)} images in {input_dir}")

    shortlist = stage1_filter.run(paths, shortlist_size=shortlist_size)
    selected = stage2_select.run(shortlist, location=location, count=count)

    return selected
