"""Entry point for Light-Go command line interface."""
from __future__ import annotations

import argparse

from core.engine import Engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Light-Go")
    parser.add_argument("--mode", choices=["train", "api", "gtp", "single"], default="train")
    parser.add_argument("--data", help="Input data directory")
    parser.add_argument("--output", help="Output model directory")
    args = parser.parse_args()

    if args.mode == "train":
        if not args.data or not args.output:
            parser.error("--data and --output are required for train mode")
        engine = Engine(args.output)
        name = engine.train(args.data)
        print(f"Saved strategy {name} to {args.output}")
    else:
        print(f"Mode '{args.mode}' is not implemented in this simplified version")


if __name__ == "__main__":
    main()
