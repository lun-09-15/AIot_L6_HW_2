# HW2 — FloodProbability Multiple Linear Regression

Kaggle source: [Regression with a Flood Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e5). The data has **20 numeric source features** and continuous target `FloodProbability`.

The large Kaggle files are deliberately excluded from Git. Download `train.csv` and
`test.csv` from Kaggle, then place them in `data/` before training locally.

```powershell
python -m pip install -r requirements.txt
python train.py
python predict.py --input data/test.csv
```

`train.py` follows all six CRISP-DM phases. It creates an 80/20 holdout split, compares 10–20 `SelectKBest(f_regression)` feature counts through training-only CV, fits OLS using the selected count, evaluates MAE/RMSE/R² against a baseline, and creates a 95% prediction-interval plot.

Generated files are in `outputs/`: `report.md`, `feature_selection.png`, `evaluation.png`, `prediction_intervals.png`, `feature_selection_cv.csv`, `test_predictions_and_intervals.csv`, `top_10_test_outliers.csv`, and `flood_probability_pipeline.joblib`.

## GitHub Actions

The repository workflow runs syntax checks plus a complete train-and-predict smoke
test with generated data whenever code is pushed or submitted in a pull request.
It does not upload or download the large Kaggle CSV files.
