#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

from curator import pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select the best N photos from a large folder using CLIP + Ollama vision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input ~/Photos/Scotland --location "Scotland"
  python main.py --input ~/Photos/Japan --location "Japan" --count 20 --shortlist 300
  python main.py --input ~/Photos/Scotland --location "Scotland" --copy-to ./selected

  # Use Claude API instead of Ollama (requires ANTHROPIC_API_KEY + paid credits):
  python main.py --input ~/Photos/Scotland --location "Scotland" --use-claude
  python main.py --input ~/Photos/Scotland --location "Scotland" --use-claude --model claude-opus-4-8

  # Override Ollama model:
  python main.py --input ~/Photos/Scotland --location "Scotland" --model llava:13b

  # Stage 1 only (no vision model needed):
  python main.py --input ~/Photos/Scotland --location "Scotland" --stage1-only

  # Force a full Stage 1 rescore, ignoring the cached shortlist:
  python main.py --input ~/Photos/Scotland --location "Scotland" --refresh-cache
        """,
    )
    p.add_argument("--input", required=True, help="Path to the folder of photos")
    p.add_argument("--location", required=True, help="Location name (e.g. 'Scotland')")
    p.add_argument("--count", type=int, default=10, help="Number of final selections (default: 10)")
    p.add_argument(
        "--shortlist",
        type=int,
        default=150,
        help="Stage 1 shortlist size before vision review (default: 150)",
    )
    p.add_argument(
        "--copy-to",
        metavar="DIR",
        help="If set, copy selected files to this directory",
    )
    p.add_argument(
        "--output",
        default="results/selection.json",
        help="Path for the JSON results file (default: results/selection.json)",
    )
    p.add_argument(
        "--use-claude",
        action="store_true",
        help=(
            "Use the Claude API for Stage 2 instead of Ollama. "
            "Requires ANTHROPIC_API_KEY and separate API credits (not included in Claude Pro/Max)."
        ),
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "Override the vision model. "
            "Ollama default: llava. Claude default: claude-sonnet-4-6. "
            "Examples: --model llava:13b  or  --model claude-opus-4-8"
        ),
    )
    p.add_argument(
        "--stage1-only",
        action="store_true",
        help="Run Stage 1 only (no vision model) — useful for a quick quality-ranked shortlist",
    )
    p.add_argument(
        "--refresh-cache",
        action="store_true",
        help=(
            "Ignore the cached Stage 1 shortlist and rescore from scratch. "
            "By default, a matching cached shortlist (results/stage1_cache.json) is reused — "
            "this avoids redoing a long CPU-bound CLIP pass if Stage 2 fails."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stage1_only:
        from curator import stage1_filter
        from curator.utils import collect_images

        paths = collect_images(args.input)
        if not paths:
            print(f"No supported images found in: {args.input}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(paths)} images")
        shortlist = stage1_filter.run(
            paths,
            shortlist_size=args.shortlist,
            input_dir=args.input,
            use_cache=not args.refresh_cache,
        )
        results = [
            {
                "rank": i + 1,
                "path": str(item.path),
                "aesthetic_score": round(item.aesthetic_score, 4),
                "cluster": item.cluster,
            }
            for i, item in enumerate(
                sorted(shortlist, key=lambda x: x.aesthetic_score, reverse=True)
            )
        ]
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nStage 1 shortlist written to {output_path}")
        return

    result = pipeline.run(
        input_dir=args.input,
        location=args.location,
        count=args.count,
        shortlist_size=args.shortlist,
        use_claude=args.use_claude,
        model=args.model,
        refresh_cache=args.refresh_cache,
    )

    results = [
        {
            "rank": img.rank,
            "path": str(img.path),
            "filename": img.path.name,
            "stage1_score": round(img.stage1_score, 4),
            "stage2_score": img.stage2_score,
            "cluster": img.cluster,
            "tags": img.tags,
            "reasoning": img.reasoning,
        }
        for img in sorted(result.selected, key=lambda x: x.rank)
    ]

    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {output_path}")

    if args.copy_to:
        dest = Path(args.copy_to)
        dest.mkdir(parents=True, exist_ok=True)
        for item in results:
            src = Path(item["path"])
            target = dest / f"{item['rank']:02d}_{src.name}"
            shutil.copy2(src, target)
        print(f"Selected files copied to {dest}")

        # Also copy the full Stage 1 shortlist into a subfolder so you can browse
        # the candidates that almost made the cut, not just the final picks.
        shortlist_dest = dest / "shortlist"
        shortlist_dest.mkdir(parents=True, exist_ok=True)
        ranked_shortlist = sorted(result.shortlist, key=lambda x: x.aesthetic_score, reverse=True)
        for i, item in enumerate(ranked_shortlist, 1):
            target = shortlist_dest / f"{i:03d}_{item.path.name}"
            shutil.copy2(item.path, target)
        print(f"Stage 1 shortlist ({len(ranked_shortlist)} images) copied to {shortlist_dest}")

    print("\n--- Final Selection ---")
    for item in results:
        print(f"  #{item['rank']:2d} [{item['stage2_score']}/10] {item['filename']}")
        print(f"       Tags: {', '.join(item['tags'])}")
        print(f"       {item['reasoning']}")


if __name__ == "__main__":
    main()
