#!/usr/bin/env python3
"""CLI for tagging .docx files with zero-shot classification."""
import argparse
import sys
from pathlib import Path

from tagger.pipeline import tag_file, tag_directory


def main():
    parser = argparse.ArgumentParser(description="Tag .docx files for recommendation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Single .docx file to tag")
    group.add_argument("--dir", help="Directory of .docx files to tag")

    parser.add_argument("--output", help="Output JSON path (for --file)")
    parser.add_argument("--output-dir", help="Output directory (for --dir)")
    parser.add_argument(
        "--model",
        default="facebook/bart-large-mnli",
        help="HuggingFace model for zero-shot classification",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device index (-1 = CPU, 0 = first GPU)",
    )

    args = parser.parse_args()

    if args.file:
        if not args.output:
            args.output = str(Path(args.file).with_suffix(".json"))
        print(f"Tagging {args.file} -> {args.output}")
        tag_file(args.file, args.output, model_name=args.model, device=args.device)
        print("Done.")
    else:
        if not args.output_dir:
            args.output_dir = str(Path(args.dir) / "tagged")
        print(f"Tagging directory {args.dir} -> {args.output_dir}")
        results = tag_directory(
            args.dir, args.output_dir, model_name=args.model, device=args.device
        )
        print(f"Done. Tagged {len(results)} file(s).")


if __name__ == "__main__":
    main()
