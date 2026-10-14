"""HW2 multiple linear regression: Kaggle FloodProbability, CRISP-DM workflow."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, ID_COLUMN, TARGET

ROOT = Path(__file__).resolve().parent
SEED = 42
SELECTION_SAMPLE_ROWS = 100_000


def build_model(k):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_regression, k=k)),
        ("regression", LinearRegression()),
    ])


def metric_values(actual, predicted):
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(root_mean_squared_error(actual, predicted)),
            "R2": float(r2_score(actual, predicted))}


def check_data(frame):
    required = set(FEATURES + [ID_COLUMN, TARGET])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    numeric = frame.loc[:, FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("The 20 features and target must be finite numeric values with no missing data.")
    if not numeric[TARGET].between(0, 1).all():
        raise ValueError("FloodProbability must be in [0, 1].")
    return numeric.loc[:, FEATURES], numeric[TARGET]


def save_feature_selection_plot(results, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, column, title, ylabel in [
        (axes[0], "RMSE", "CV RMSE by selected-feature count", "RMSE"),
        (axes[1], "R2", "CV R² by selected-feature count", "R²"),
    ]:
        axis.plot(results.k, results[column], marker="o")
        axis.set(title=title, xlabel="Selected features", ylabel=ylabel)
        axis.set_xticks(results.k)
        axis.grid(alpha=0.25)
    fig.suptitle("Training-only feature selection", weight="bold")
    fig.tight_layout()
    fig.savefig(output / "feature_selection.png", dpi=180)
    plt.close(fig)


def save_evaluation_plots(y, prediction, lower, upper, output):
    rng = np.random.default_rng(SEED)
    shown = rng.choice(len(y), size=min(10_000, len(y)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    limits = [min(y.min(), prediction.min()), max(y.max(), prediction.max())]
    axes[0].scatter(y.iloc[shown], prediction[shown], s=4, alpha=0.18)
    axes[0].plot(limits, limits, "r--", label="Perfect prediction")
    axes[0].set(title="Held-out actual vs. predicted", xlabel="Actual FloodProbability", ylabel="Predicted FloodProbability")
    axes[0].legend()
    axes[1].scatter(prediction[shown], y.iloc[shown].to_numpy() - prediction[shown], s=4, alpha=0.18)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set(title="Held-out residuals", xlabel="Predicted FloodProbability", ylabel="Actual − predicted")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=180)
    plt.close(fig)

    order = np.linspace(0, len(prediction) - 1, min(150, len(prediction)), dtype=int)
    order = order[np.argsort(prediction[order])]
    fig, axis = plt.subplots(figsize=(11, 5.8))
    x = np.arange(1, len(order) + 1)
    axis.errorbar(x, prediction[order], yerr=[prediction[order] - lower[order], upper[order] - prediction[order]],
                  fmt="o", markersize=3, capsize=2, label="Prediction ± 95% interval")
    axis.scatter(x, y.iloc[order], color="#d62728", marker="x", s=30, label="Actual held-out value", zorder=3)
    axis.set(title="Held-out predictions with 95% prediction intervals",
             xlabel="150 test rows sorted by prediction", ylabel="FloodProbability", ylim=(-0.05, 1.05))
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "prediction_intervals.png", dpi=180)
    plt.close(fig)


def main(csv_path, output):
    output.mkdir(parents=True, exist_ok=True)
    # 1. Business understanding: predict a continuous flood-risk probability.
    # Academic success: improve held-out MAE/RMSE over a training-mean baseline.
    raw = pd.read_csv(csv_path)
    X, y = check_data(raw)
    # 2. Data understanding: `id` is only an identifier and is excluded.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
    X_train.describe().T.to_csv(output / "training_feature_summary.csv")
    X_train.corrwith(y_train).sort_values().rename("correlation_with_target").to_csv(output / "training_target_correlations.csv")

    # 3–4. Preparation/modeling: choose count using training data only.
    selection_X = X_train.sample(n=min(SELECTION_SAMPLE_ROWS, len(X_train)), random_state=SEED)
    selection_y = y_train.loc[selection_X.index]
    cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = []
    for k in range(10, 21):
        scores = cross_validate(build_model(k), selection_X, selection_y, cv=cv,
                                scoring={"RMSE": "neg_root_mean_squared_error", "R2": "r2"}, n_jobs=-1)
        rows.append({"k": k, "RMSE": -scores["test_RMSE"].mean(), "R2": scores["test_R2"].mean()})
    selection = pd.DataFrame(rows)
    selection.to_csv(output / "feature_selection_cv.csv", index=False)
    save_feature_selection_plot(selection, output)
    best_k = int(selection.loc[selection.RMSE.idxmin(), "k"])
    model, baseline = build_model(best_k), DummyRegressor(strategy="mean")
    model.fit(X_train, y_train)
    baseline.fit(X_train, y_train)
    prediction, baseline_prediction = model.predict(X_test), baseline.predict(X_test)
    selected_features = np.array(FEATURES)[model.named_steps["select"].get_support()].tolist()

    # 5. Evaluation: only now evaluate the untouched 20% holdout.
    results = {"train": metric_values(y_train, model.predict(X_train)),
               "test": metric_values(y_test, prediction),
               "baseline_test": metric_values(y_test, baseline_prediction)}
    transformed_train = model.named_steps["select"].transform(model.named_steps["scale"].transform(model.named_steps["impute"].transform(X_train)))
    transformed_test = model.named_steps["select"].transform(model.named_steps["scale"].transform(model.named_steps["impute"].transform(X_test)))
    design_train = np.column_stack([np.ones(len(X_train)), transformed_train])
    design_test = np.column_stack([np.ones(len(X_test)), transformed_test])
    residual_df = len(X_train) - np.linalg.matrix_rank(design_train)
    mse = np.sum((y_train.to_numpy() - model.predict(X_train)) ** 2) / residual_df
    leverage = np.einsum("ij,jk,ik->i", design_test, np.linalg.pinv(design_train.T @ design_train), design_test)
    half_width = t.ppf(0.975, residual_df) * np.sqrt(mse * (1 + leverage))
    lower, upper = prediction - half_width, prediction + half_width
    test_results = pd.DataFrame({"source_row_index": X_test.index, "actual": y_test.to_numpy(), "predicted": prediction,
                                 "residual": y_test.to_numpy() - prediction, "prediction_interval_lower": lower,
                                 "prediction_interval_upper": upper, "inside_95pct_interval": (y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper)})
    test_results.to_csv(output / "test_predictions_and_intervals.csv", index=False)
    test_results.assign(absolute_residual=lambda data: data.residual.abs()).nlargest(10, "absolute_residual").to_csv(output / "top_10_test_outliers.csv", index=False)
    save_evaluation_plots(y_test.reset_index(drop=True), prediction, lower, upper, output)

    # 6. Deployment: save the evaluated training-only pipeline and reload-check it.
    model_path = output / "flood_probability_pipeline.joblib"
    joblib.dump(model, model_path)
    np.testing.assert_allclose(joblib.load(model_path).predict(X_test), prediction)
    metadata = {"source": "https://www.kaggle.com/competitions/playground-series-s4e5",
                "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(), "raw_rows": len(raw),
                "source_feature_count": 20, "selected_feature_count": best_k, "selected_features": selected_features,
                "training_rows": len(X_train), "test_rows": len(X_test), "selection_sample_rows": len(selection_X),
                "selection_cv_folds": 3, "metrics": results,
                "prediction_interval_coverage": float(test_results.inside_95pct_interval.mean()),
                "prediction_interval_residual_df": int(residual_df)}
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = f"""# HW2 — FloodProbability Multiple Linear Regression

## 1. Business Understanding
Predict the continuous `FloodProbability` from 20 environmental, infrastructure and planning indicators. This is an academic regression analysis, not a flood-warning system. Success means lower held-out MAE and RMSE than a mean-prediction baseline.

## 2. Data Understanding
Source: {metadata['source']}

The training data contains {len(raw):,} rows, 20 numeric source features, row identifier `id`, and target `FloodProbability` (0–1). `id` is excluded. No missing/non-finite values were found. The fixed 80/20 split has {len(X_train):,}/{len(X_test):,} rows. See `training_feature_summary.csv` and `training_target_correlations.csv`.

## 3. Data Preparation
Median imputation and standardization are inside the pipeline. There are no categorical variables. The test split is isolated before selection/fitting. No row-sum, mean, or other target-proxy aggregate feature is constructed.

## 4. Modeling and Feature Selection
`SelectKBest(f_regression)` compares 10–20 features using 3-fold CV on a fixed {len(selection_X):,}-row sample from the training split. The lowest mean CV RMSE selects **{best_k} features**. The final estimator is ordinary least squares (`LinearRegression`) fitted on all training rows. Selected features: {', '.join(selected_features)}. See `feature_selection.png` and `feature_selection_cv.csv`.

## 5. Evaluation
| Model / split | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear regression / training | {results['train']['MAE']:.5f} | {results['train']['RMSE']:.5f} | {results['train']['R2']:.5f} |
| Linear regression / test | {results['test']['MAE']:.5f} | {results['test']['RMSE']:.5f} | {results['test']['R2']:.5f} |
| Mean baseline / test | {results['baseline_test']['MAE']:.5f} | {results['baseline_test']['RMSE']:.5f} | {results['baseline_test']['R2']:.5f} |

`evaluation.png` contains prediction/residual plots. `prediction_intervals.png` contains 150 held-out predictions with 95% OLS prediction intervals. Observed interval coverage: {metadata['prediction_interval_coverage']:.2%}. These intervals rely on linear-model assumptions and are descriptive.

## 6. Deployment
`flood_probability_pipeline.joblib` bundles imputation, scaling, selection and OLS; it was reloaded and verified. Use `predict.py` for new CSV data with the 20 named source features. Before real deployment, validate using representative locations/times and monitor input drift.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"selected_feature_count": best_k, "test": results["test"],
                      "interval_coverage": metadata["prediction_interval_coverage"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    options = parser.parse_args()
    main(options.csv, options.output)
