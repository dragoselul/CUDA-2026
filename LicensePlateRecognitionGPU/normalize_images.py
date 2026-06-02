#!/usr/bin/env python3
"""
normalize_images.py
Finds the most common image size in a batch of .ppm files and
letterboxes all images to that size (black padding, no stretching).

Usage:
    python normalize_images.py [directory] [--output-dir DIR] [--dry-run]
"""

import argparse
from collections import Counter
from pathlib import Path
from PIL import Image


def collect_ppms(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.ppm"))


def read_sizes(paths: list[Path]) -> dict[Path, tuple[int, int]]:
    sizes = {}
    for p in paths:
        with Image.open(p) as img:
            sizes[p] = img.size  # (W, H)
    return sizes


def most_common_size(sizes: dict[Path, tuple[int, int]]) -> tuple[int, int]:
    (target_w, target_h), _ = Counter(sizes.values()).most_common(1)[0]
    return target_w, target_h


def letterbox(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    scale   = min(target_w / src_w, target_h / src_h)
    new_w   = int(src_w * scale)
    new_h   = int(src_h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas  = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Letterbox .ppm images to the most common size in the batch."
    )
    parser.add_argument(
        "directory", nargs="?", default=".",
        help="Directory containing .ppm files (default: current dir)"
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Where to write results (default: overwrite in-place)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would happen without writing anything"
    )
    args = parser.parse_args()

    src_dir = Path(args.directory)
    paths   = collect_ppms(src_dir)

    if not paths:
        print(f"No .ppm files found in {src_dir}")
        return

    sizes    = read_sizes(paths)
    target_w, target_h = most_common_size(sizes)
    total    = len(paths)
    already  = sum(1 for s in sizes.values() if s == (target_w, target_h))

    print(f"Found       : {total} image(s)")
    print(f"Target size : {target_w}x{target_h}  (most common, {already}/{total} already match)")
    print(f"To letterbox: {total - already}")

    if args.dry_run:
        print()
        for p, (w, h) in sizes.items():
            tag = "ok" if (w, h) == (target_w, target_h) else f"{w}x{h} -> letterbox"
            print(f"  {p.name}: {tag}")
        return

    out_dir = Path(args.output_dir) if args.output_dir else src_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for p, (w, h) in sizes.items():
        out_path = out_dir / p.name
        if (w, h) == (target_w, target_h):
            if out_dir != src_dir:
                import shutil
                shutil.copy2(p, out_path)
            continue

        with Image.open(p) as img:
            result = letterbox(img, target_w, target_h)
        result.save(out_path, format="PPM")
        print(f"  {p.name}: {w}x{h} -> {target_w}x{target_h}")

    print("Done.")


if __name__ == "__main__":
    main()
