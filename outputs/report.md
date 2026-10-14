# HW2 — FloodProbability Multiple Linear Regression

## 1. Business Understanding
Predict the continuous `FloodProbability` from 20 environmental, infrastructure and planning indicators. This is an academic regression analysis, not a flood-warning system. Success means lower held-out MAE and RMSE than a mean-prediction baseline.

## 2. Data Understanding
Source: https://www.kaggle.com/competitions/playground-series-s4e5

The training data contains 1,117,957 rows, 20 numeric source features, row identifier `id`, and target `FloodProbability` (0–1). `id` is excluded. No missing/non-finite values were found. The fixed 80/20 split has 894,365/223,592 rows. See `training_feature_summary.csv` and `training_target_correlations.csv`.

## 3. Data Preparation
Median imputation and standardization are inside the pipeline. There are no categorical variables. The test split is isolated before selection/fitting. No row-sum, mean, or other target-proxy aggregate feature is constructed.

## 4. Modeling and Feature Selection
`SelectKBest(f_regression)` compares 10–20 features using 3-fold CV on a fixed 100,000-row sample from the training split. The lowest mean CV RMSE selects **20 features**. The final estimator is ordinary least squares (`LinearRegression`) fitted on all training rows. Selected features: MonsoonIntensity, TopographyDrainage, RiverManagement, Deforestation, Urbanization, ClimateChange, DamsQuality, Siltation, AgriculturalPractices, Encroachments, IneffectiveDisasterPreparedness, DrainageSystems, CoastalVulnerability, Landslides, Watersheds, DeterioratingInfrastructure, PopulationScore, WetlandLoss, InadequatePlanning, PoliticalFactors. See `feature_selection.png` and `feature_selection_cv.csv`.

## 5. Evaluation
| Model / split | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear regression / training | 0.01582 | 0.02010 | 0.84497 |
| Linear regression / test | 0.01579 | 0.02008 | 0.84488 |
| Mean baseline / test | 0.04089 | 0.05098 | -0.00000 |

`evaluation.png` contains prediction/residual plots. `prediction_intervals.png` contains 150 held-out predictions with 95% OLS prediction intervals. Observed interval coverage: 94.61%. These intervals rely on linear-model assumptions and are descriptive.

## 6. Deployment
`flood_probability_pipeline.joblib` bundles imputation, scaling, selection and OLS; it was reloaded and verified. Use `predict.py` for new CSV data with the 20 named source features. Before real deployment, validate using representative locations/times and monitor input drift.
