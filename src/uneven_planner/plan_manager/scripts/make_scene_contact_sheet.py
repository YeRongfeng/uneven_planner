#!/usr/bin/env python3
"""Build a compact contact sheet from terrain-scene preview images."""

import argparse
import math
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()
    if args.columns <= 0:
        raise ValueError("columns must be positive")

    rows = int(math.ceil(len(args.inputs) / args.columns))
    figure, axes = plt.subplots(
        rows, args.columns, figsize=(5.6 * args.columns, 4.6 * rows),
        squeeze=False)
    for axis, path in zip(axes.ravel(), args.inputs):
        axis.imshow(mpimg.imread(path))
        parent = os.path.basename(os.path.dirname(path))
        scene = os.path.basename(path).replace("_preview.png", "")
        axis.set_title(f"{parent} / {scene}", fontsize=10)
        axis.axis("off")
    for axis in axes.ravel()[len(args.inputs):]:
        axis.axis("off")
    figure.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    figure.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
