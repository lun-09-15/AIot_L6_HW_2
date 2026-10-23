"""Validate the train/predict products created by the GitHub Actions smoke run."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    options = parser.parse_args()

    metrics = json.loads((options.outputs / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["raw_rows"] == 300
    assert metrics["training_rows"] == 240
    assert metrics["test_rows"] == 60
    assert metrics["source_feature_count"] == 20
    assert 10 <= metrics["selected_feature_count"] <= 20

    test_rmse = metrics["metrics"]["test"]["RMSE"]
    baseline_rmse = metrics["metrics"]["baseline_test"]["RMSE"]
    assert math.isfinite(test_rmse) and math.isfinite(baseline_rmse)
    assert test_rmse < baseline_rmse, "Regression should beat the mean baseline on smoke data"

    predictions = pd.read_csv(options.predictions)
    assert len(predictions) == 300
    assert "predicted_FloodProbability" in predictions.columns
    assert predictions["predicted_FloodProbability"].map(math.isfinite).all()


if __name__ == "__main__":
    main()
