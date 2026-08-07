"""Command-line entry point for model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import KAGGLE_FILENAME, generate_customers, load_training_data
from .model import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare churn models.")
    parser.add_argument("--data", type=Path, help="Optional CSV dataset")
    parser.add_argument(
        "--rows", type=int, default=5_000, help="Synthetic rows when no CSV is given"
    )
    parser.add_argument("--model", type=Path, default=Path("models/churn_model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("models/metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_data = Path("data") / KAGGLE_FILENAME
    if args.data:
        data = load_training_data(args.data)
    elif default_data.exists():
        data = load_training_data(default_data)
    else:
        data = generate_customers(args.rows)
    _, report = train_and_evaluate(data, args.model, args.metrics)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
