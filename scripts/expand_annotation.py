#!/usr/bin/env python3
"""Expand MTCIR annotations so each sample has exactly one modification.

Input format (one JSON object per line):
{"id": "001145879f", "image": "...jpg", "target_image": "...jpg",
 "modifications": ["Add a Marvel badge.", "Change figure to Iron Man.", ...]}

Each element of the `modifications` list becomes a separate output line while
keeping the original row fields. The output `modifications` value is always
presented as a single-item list for compatibility with existing consumers.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable, List


def _normalize_modifications(raw_modifications: object) -> List[str]:
    """Return a list of non-empty modification strings."""
    if isinstance(raw_modifications, list):
        values = raw_modifications
    elif isinstance(raw_modifications, str):
        try:
            loaded = json.loads(raw_modifications)
            if isinstance(loaded, list):
                values = loaded
            else:
                values = [str(loaded)]
        except json.JSONDecodeError:
            values = [raw_modifications]
    else:
        values = [str(raw_modifications)]

    cleaned = []
    for item in values:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def expand_annotations(in_path: pathlib.Path, out_path: pathlib.Path, id_sep: str = "::") -> None:
    with in_path.open("r", encoding="utf-8") as inp_f, out_path.open(
        "w", encoding="utf-8"
    ) as out_f:
        for line in inp_f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            sample_id = str(example.get("id", "")).strip()
            modifications = _normalize_modifications(example.get("modifications", []))

            if not modifications:
                # If there are no usable modifications, write a defensive fallback
                # that keeps schema stable.
                out_record = dict(example)
                out_record["modifications"] = []
                out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                continue

            for i, modification in enumerate(modifications):
                expanded = dict(example)
                expanded["modifications"] = [modification]
                if sample_id:
                    expanded["id"] = f"{sample_id}{id_sep}{i}"
                out_f.write(json.dumps(expanded, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "-i",
        type=pathlib.Path,
        default=pathlib.Path("../MTCIR/mtcir.jsonl"),
        help="Path to the original annotation JSONL file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        default=pathlib.Path("../MTCIR/mtcir_expanded.jsonl"),
        help="Path where expanded annotations will be written.",
    )
    parser.add_argument(
        "--id-separator",
        type=str,
        default="::",
        help="Separator between original id and the modification index in expanded rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expand_annotations(args.input, args.output, id_sep=args.id_separator)


if __name__ == "__main__":
    main()
