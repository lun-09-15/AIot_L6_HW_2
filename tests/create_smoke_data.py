"""Create a compact, reproducible FloodProbability CSV for CI validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow this helper to be run from the repository root in GitHub Actions.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import FEATURES, ID_COLUMN, TARGET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=300)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    values = rng.integers(0, 11, size=(args.rows, len(FEATURES)))
    frame = pd.DataFrame(values, columns=FEATURES)
    frame.insert(0, ID_COLUMN, np.arange(args.rows))

    signal = frame[FEATURES].to_numpy().mean(axis=1) / 11
    noise = rng.normal(0, 0.003, size=args.rows)
    frame[TARGET] = np.clip(signal + noise, 0, 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
