# HW2 — Multiple Linear Regression Web App

Kaggle source: [Regression with a Flood Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e5). The data has **20 numeric source features** and continuous target `FloodProbability`.

The complete source data is included as
`data/flood_prediction_dataset.zip` (containing `train.csv` and `test.csv`).
Extract it into `data/` before running the command-line analysis. Individual CSV
files remain excluded from Git so the repository stores one compact source archive.

## Streamlit web deployment

The main submission is a GitHub repository and Streamlit web app. `app.py` uses
the same Python scientific-computing environment as the command-line analysis.
It has a dropdown for two regression problems and meets the bonus requirement for
loading an arbitrary CSV URL or uploaded CSV file:

- **Flood Probability**: automatically downloads the repository's complete source-data ZIP; it has 20 source features.
- **50 Startups**: a ready-to-run second regression problem.
- **Custom CSV URL / upload**: choose any numeric target column in the interface.

Run locally:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

To deploy with Streamlit Community Cloud, create a new app from this GitHub
repository, select branch `main`, and set the main file path to `app.py`. The
deployed app runs the sklearn preprocessing, feature selection, linear regression,
model evaluation, and prediction-interval calculations directly in Python.

## Command-line analysis

```powershell
python train.py
python predict.py --input data/test.csv
```

`train.py` follows all six CRISP-DM phases. It creates an 80/20 holdout split, compares 10–20 `SelectKBest(f_regression)` feature counts through training-only CV, fits OLS using the selected count, evaluates MAE/RMSE/R² against a baseline, and creates a 95% prediction-interval plot.

Generated files are in `outputs/`: `report.md`, `feature_selection.png`, `evaluation.png`, `prediction_intervals.png`, `feature_selection_cv.csv`, `test_predictions_and_intervals.csv`, `top_10_test_outliers.csv`, and `flood_probability_pipeline.joblib`.

## GitHub Actions

The repository workflow runs syntax checks plus a complete train-and-predict smoke
test with generated data whenever code is pushed or submitted in a pull request.
It does not upload or download the large Kaggle CSV files.
