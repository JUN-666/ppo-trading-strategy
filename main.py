
"""Command-line entry point for PPO HFT experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO high-frequency trading research runner")
    parser.add_argument("mode", choices=["train", "evaluate", "describe"], help="Execution mode")
    parser.add_argument("--data", type=Path, default=DEFAULT_CONFIG.data_path, help="Private feature store path")
    parser.add_argument("--model", type=Path, default=DEFAULT_CONFIG.model_path, help="Model checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "describe":
        print(DEFAULT_CONFIG)
        return
    if args.mode == "train":
        from scripts.train import main as train_main

        train_main()  # existing script owns data/model configuration
        return
    if args.mode == "evaluate":
        from scripts.evaluate import main as evaluate_main

        evaluate_main()  # existing script owns data/model configuration
        return


if __name__ == "__main__":
    main()
