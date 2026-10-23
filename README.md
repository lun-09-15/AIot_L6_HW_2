# HW2｜多元線性回歸分析網站

[![Validate HW2](https://github.com/lun-09-15/AIot_L6_HW_2/actions/workflows/hw2-validate.yml/badge.svg)](https://github.com/lun-09-15/AIot_L6_HW_2/actions/workflows/hw2-validate.yml)

**線上網站：** [Multiple Linear Regression Lab](https://lun-09-15-aiotl6hw2.streamlit.app/)

本專案依 CRISP-DM 流程，以 Kaggle 洪水資料為主要題目，使用 `sklearn.LinearRegression` 預測連續目標 `FloodProbability`。網站可切換三個回歸題目，也支援自訂 CSV 網址與上傳；提供資料探索、特徵選擇、模型評估、95% **預測區間**、異常值分析及單筆預測。Flood 資料有 **20 個原始輸入特徵**，符合本次作業「超過 13 個特徵」的要求。

## 網站畫面

![Flood 資料已載入，顯示資料來源、筆數、欄位數及可分頁的資料預覽](docs/images/dashboard-overview.png)

首頁會自動載入 GitHub 中的 Flood 資料 ZIP，顯示 Kaggle 來源、1,117,957 筆資料、22 欄與資料品質摘要。下方表格每頁顯示 5,000 筆，可捲動及切換頁碼。

![模型評估中的實際值與預測值，以及 95% 預測區間](docs/images/dashboard-prediction.png)

「預測表現」分頁左側比較實際值與預測值，右側畫出預測值及 95% 預測區間；頁面上方另提供 MAE、RMSE、R² 和區間涵蓋率。右圖只顯示依預測值排序後的前 200 筆，不能用它估計整體涵蓋率。

Kaggle source: [Regression with a Flood Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e5). The data has **20 numeric source features** and continuous target `FloodProbability`.

Additional Kaggle sources:

- [Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who): 21 original predictors, target `Life expectancy`.
- [Student Performance Data Set](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set): 32 original predictors in `student-por.csv`, target `G3`.

The source data is included as `data/flood_prediction_dataset.zip` (containing
`train.csv` and `test.csv`), `data/life_expectancy_who.zip`, and
`data/student_performance.zip`. Extract the Flood archive into `data/` before
running the command-line analysis. Individual CSV files remain excluded from Git.
The Kaggle `test.csv` has no `FloodProbability` labels, so holdout evaluation
uses a split of the labeled `train.csv`, not the Kaggle test file.

## Streamlit web deployment

Live app: [AIoT HW2 regression dashboard](https://lun-09-15-aiotl6hw2.streamlit.app/).

The main submission is a GitHub repository and Streamlit web app. `app.py` uses
the same Python scientific-computing environment as the command-line analysis.
It has a dropdown for three regression problems and meets the bonus requirement for
loading an arbitrary CSV URL or uploaded CSV file:

- **Flood Probability**: automatically downloads the repository's complete source-data ZIP; it has 20 source features.
- **Life Expectancy (WHO)**: automatically loads 21 source features.
- **Student Performance**: automatically loads 32 source features.
- **Custom CSV URL / upload**: choose any numeric target column in the interface.

Custom CSV inputs are limited to 50 MB. The app checks for empty data, missing
numeric targets, too few usable rows, and excessive categorical expansion, and
shows an explanation when a dataset cannot be modeled.

Run locally with Python 3.12:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

To deploy with Streamlit Community Cloud, create a new app from this GitHub
repository, select branch `main`, and set the main file path to `app.py`. The
deployed app runs the sklearn preprocessing, feature selection, linear regression,
model evaluation, and prediction-interval calculations directly in Python.
The data table shows 5,000 scrollable rows per page, so all records can be
inspected without sending over a million rows to the browser at once. The web
model uses a reproducible 100,000-row sample for large datasets and updates
automatically when the target, feature count, or test split changes.
The dashboard also shows target distribution, missing-value and correlation
charts, column quality, feature selection scores, standardized coefficients,
residual diagnostics, and the top 10 largest prediction errors. Large charts
use bounded samples while their data-table pagination covers all rows.
It compares the best single-feature model with the selected multiple-feature
model on the same holdout set. A separate 3-fold training-only cross-validation
curve compares 1–40 encoded feature counts (using up to 3,000 training rows),
without using the holdout test set to recommend a feature count. The deployment
section accepts one editable observation and immediately shows its prediction
and 95% prediction interval; unfamiliar categories and out-of-range numeric
values produce a caution message.
The checked-in `.streamlit/config.toml` fixes a light, high-contrast theme so
the hosted app keeps readable tables and controls across browser theme settings.

## Command-line analysis

```powershell
Expand-Archive -LiteralPath data/flood_prediction_dataset.zip -DestinationPath data -Force
python train.py
python predict.py --input data/test.csv
```

`train.py` follows all six CRISP-DM phases. It creates an 80/20 holdout split, compares 10–20 `SelectKBest(f_regression)` feature counts through training-only CV, fits OLS using the selected count, evaluates MAE/RMSE/R² against a baseline, and creates a 95% prediction-interval plot.

Generated files are in `outputs/`: `report.md`, `feature_selection.png`, `evaluation.png`, `prediction_intervals.png`, `feature_selection_cv.csv`, `test_predictions_and_intervals.csv`, `top_10_test_outliers.csv`, and `flood_probability_pipeline.joblib`.

## GitHub Actions

The repository workflow runs syntax checks plus a complete train-and-predict smoke
test with generated data whenever code is pushed or submitted in a pull request.
It also validates that the three included data ZIPs contain the expected CSV
files and target columns. It does not train on the full Kaggle dataset in CI;
the checked-in `outputs/` results come from the full local experiment.
