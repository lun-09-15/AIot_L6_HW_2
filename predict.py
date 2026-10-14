"""Predict FloodProbability for a compatible CSV."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import FEATURES


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=root / "outputs" / "flood_probability_pipeline.joblib")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "new_predictions.csv")
    options = parser.parse_args()
    frame = pd.read_csv(options.input)
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    model = joblib.load(options.model)  # Load trusted model artifacts only.
    result = frame.copy()
    result["predicted_FloodProbability"] = model.predict(frame.loc[:, FEATURES])
    options.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(options.output, index=False)
    print(f"Saved: {options.output.resolve()}")


if __name__ == "__main__":
    main()
