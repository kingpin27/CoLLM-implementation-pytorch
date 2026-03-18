#!/usr/bin/env python3
"""Shuffle annotation lines in-place or into a new file."""

from __future__ import annotations

import argparse
import random
import pathlib


def shuffle_lines(input_path: pathlib.Path, output_path: pathlib.Path, seed: int | None = None) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        lines = [line for line in f if line.rstrip("\n") != ""]

    if seed is not None:
        random.seed(seed)
    random.shuffle(lines)

    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input",
        type=pathlib.Path,
        default=pathlib.Path("../MTCIR/mtcir_expanded.jsonl"),
        help="Input annotation file (JSONL)."
    )
    parser.add_argument(
        "-o", "--output",
        type=pathlib.Path,
        default=pathlib.Path("../MTCIR/mtcir_expanded_shuffled.jsonl"),
        help="Output shuffled annotation file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic shuffling."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shuffle_lines(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
