#!/usr/bin/env python3
"""
convert_knn_data.py
One-time offline conversion: classifications.xml + images.xml -> knn_data.bin

Binary format of knn_data.bin:
  [uint32]        num_samples  (e.g. 180)
  [uint32]        num_features (e.g. 600 = 20*30)
  [uint32]        reserved     (= 0)
  [int32 x N]     labels       (ASCII values: '0'=48 .. 'Z'=90)
  [float32 x N*F] features     (row-major, pixel values 0.0 - 255.0)

Usage:
  python3 convert_knn_data.py
  python3 convert_knn_data.py classifications.xml images.xml knn_data.bin

Requires only Python stdlib (xml.etree.ElementTree, struct).
"""

import sys
import struct
import xml.etree.ElementTree as ET


def parse_opencv_matrix_xml(path: str, node_name: str):
    tree = ET.parse(path)
    root = tree.getroot()
    node = root.find(node_name)
    if node is None:
        raise RuntimeError(f"Node <{node_name}> not found in {path}")
    rows  = int(node.find('rows').text.strip())
    cols  = int(node.find('cols').text.strip())
    dt    = node.find('dt').text.strip()
    data_text = node.find('data').text.strip()
    values = [float(v) for v in data_text.split()]
    if len(values) != rows * cols:
        raise RuntimeError(
            f"Expected {rows}x{cols}={rows*cols} values, got {len(values)} in {path}"
        )
    return rows, cols, dt, values


def main():
    cls_file = sys.argv[1] if len(sys.argv) > 1 else 'classifications.xml'
    img_file = sys.argv[2] if len(sys.argv) > 2 else 'images.xml'
    out_file = sys.argv[3] if len(sys.argv) > 3 else 'knn_data.bin'

    print(f"Reading labels from {cls_file} ...")
    n_cls, _, dt_cls, labels = parse_opencv_matrix_xml(cls_file, 'classifications')
    print(f"  {n_cls} samples, dtype={dt_cls}")

    print(f"Reading features from {img_file} ...")
    n_img, n_feat, dt_img, features = parse_opencv_matrix_xml(img_file, 'images')
    print(f"  {n_img} samples, {n_feat} features each, dtype={dt_img}")

    if n_cls != n_img:
        raise RuntimeError(f"Sample count mismatch: {n_cls} labels vs {n_img} image rows")

    labels_int = [int(round(v)) for v in labels]

    with open(out_file, 'wb') as f:
        # Header: 3 x uint32
        f.write(struct.pack('<III', n_cls, n_feat, 0))
        # Labels: N x int32
        f.write(struct.pack(f'<{n_cls}i', *labels_int))
        # Features: N*F x float32
        f.write(struct.pack(f'<{n_cls * n_feat}f', *features))

    total_bytes = 12 + n_cls * 4 + n_cls * n_feat * 4
    print(f"Written {out_file}: {total_bytes} bytes "
          f"({n_cls} samples x {n_feat} features)")


if __name__ == '__main__':
    main()
