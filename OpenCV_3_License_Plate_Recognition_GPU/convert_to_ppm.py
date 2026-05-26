#!/usr/bin/env python3
"""
convert_to_ppm.py
One-time offline conversion: PNG/JPG images -> binary PPM (P6).
Run from the project directory:
    python3 convert_to_ppm.py          # converts all *.png in current dir
    python3 convert_to_ppm.py <dir>    # converts all images in <dir>
    python3 convert_to_ppm.py a.png b.jpg   # specific files

Requires only Pillow (pip install Pillow).
"""

import sys
import os
import glob
from PIL import Image

EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


def convert(src_path: str) -> None:
    dst_path = os.path.splitext(src_path)[0] + '.ppm'
    img = Image.open(src_path).convert('RGB')   # drop alpha, unify to RGB
    img.save(dst_path)                           # Pillow writes binary P6 PPM
    print(f"  {src_path} -> {dst_path}  ({img.width}x{img.height})")


def collect_paths(args):
    paths = []
    for arg in args:
        if os.path.isdir(arg):
            for ext in EXTENSIONS:
                paths.extend(glob.glob(os.path.join(arg, '*' + ext)))
                paths.extend(glob.glob(os.path.join(arg, '*' + ext.upper())))
        elif os.path.isfile(arg):
            paths.append(arg)
        else:
            paths.extend(glob.glob(arg))
    return sorted(set(paths))


if __name__ == '__main__':
    targets = sys.argv[1:] if len(sys.argv) > 1 else ['*.png']
    paths = collect_paths(targets)

    if not paths:
        print("No images found. Usage: python3 convert_to_ppm.py [path|glob ...]")
        sys.exit(1)

    print(f"Converting {len(paths)} image(s):")
    for p in paths:
        try:
            convert(p)
        except Exception as e:
            print(f"  ERROR {p}: {e}")
