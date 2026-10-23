"""Interactive web app for HW2 multiple linear regression."""

from __future__ import annotations

from io import BytesIO
from hashlib import sha256
from typing import Tuple
from urllib.parse import urlparse
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FLOOD_TARGET = "FloodProbability"
DATA_BASE_URL = "https://raw.githubusercontent.com/lun-09-15/AIot_L6_HW_2/main/data/"
DATASETS = {
    "Flood Probability（Kaggle，20 特徵）": ("flood_prediction_dataset.zip", "train.csv", FLOOD_TARGET),
    "Life Expectancy（WHO，21 特徵）": ("life_expectancy_who.zip", "Life Expectancy Data.csv", "Life expectancy"),
    "Student Performance（葡萄牙語，32 特徵）": ("student_performance.zip", "student-por.csv", "G3"),
}
SOURCES = {
    "Flood Probability（Kaggle，20 特徵）": ("https://www.kaggle.com/competitions/playground-series-s4e5", "洪水發生機率 · 0–1 連續數值"),
    "Life Expectancy（WHO，21 特徵）": ("https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who", "各國預期壽命 · 年"),
    "Student Performance（葡萄牙語，32 特徵）": ("https://www.kaggle.com/datasets/larsen0966/student-performance-data-set", "學生期末成績 · 0–20 分"),
}

BLUE, TEAL, ORANGE = "#3478d4", "#19a78e", "#f3a344"
RESULT_CHART_SIZE = (7, 4.4)
MAX_CUSTOM_BYTES = 50 * 1024 * 1024
MAX_ENCODED_FEATURES = 500


def finish_chart(fig, axis):
    fig.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")
    axis.grid(axis="y", color="#e9eef5", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def data_profile(frame: pd.DataFrame, target_name: str):
    sample = frame.sample(n=min(len(frame), 10_000), random_state=42)
    numeric = sample.select_dtypes(include=np.number)
    target_values = pd.to_numeric(sample[target_name], errors="coerce")
    target_values = target_values[np.isfinite(target_values)]
    fig_target, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(target_values, bins=35, color=BLUE, alpha=0.85, edgecolor="white")
    ax.set(title="Target distribution", xlabel=target_name, ylabel="Count")

    missing = frame.isna().sum().sort_values(ascending=False).head(12)
    fig_missing, ax_missing = plt.subplots(figsize=(7, 3.6))
    ax_missing.barh(missing.index[::-1], missing.values[::-1], color=ORANGE)
    ax_missing.set(title="Missing values by column", xlabel="Count")

    fig_corr, ax_corr = plt.subplots(figsize=(7, 3.6))
    if target_name in numeric.columns:
        corr = numeric.corr(numeric_only=True)[target_name].drop(target_name).dropna()
        corr = corr.reindex(corr.abs().sort_values(ascending=False).head(10).index).iloc[::-1]
        ax_corr.barh(corr.index, corr.values, color=[TEAL if x >= 0 else ORANGE for x in corr])
        ax_corr.axvline(0, color="#6f7f90", linewidth=0.8)
    ax_corr.set(title="Top numeric correlations with target", xlabel="Pearson r")
    return [finish_chart(fig_target, fig_target.axes[0]),
            finish_chart(fig_missing, ax_missing), finish_chart(fig_corr, ax_corr)]


def diagnostic_charts(summary: dict, result: pd.DataFrame):
    scores = summary["feature_scores"].head(15).iloc[::-1]
    fig_scores, ax_scores = plt.subplots(figsize=RESULT_CHART_SIZE)
    ax_scores.barh(scores["feature"], scores["score"], color=BLUE)
    ax_scores.set(title="Top feature selection scores", xlabel="F score")

    coefficients = summary["coefficients"].sort_values("standardized").tail(20)
    fig_coeff, ax_coeff = plt.subplots(figsize=RESULT_CHART_SIZE)
    ax_coeff.barh(coefficients["feature"], coefficients["standardized"],
                  color=[TEAL if value >= 0 else ORANGE for value in coefficients["standardized"]])
    ax_coeff.axvline(0, color="#6f7f90", linewidth=0.8)
    ax_coeff.set(title="Standardized model coefficients", xlabel="Coefficient")

    sample = result.sample(n=min(5_000, len(result)), random_state=42)
    fig_residual, ax_residual = plt.subplots(figsize=RESULT_CHART_SIZE)
    ax_residual.scatter(sample["predicted"], sample["residual"], s=9, alpha=0.35, color=BLUE)
    ax_residual.axhline(0, color=ORANGE, linewidth=1.4)
    ax_residual.set(title="Residuals vs prediction", xlabel="Prediction", ylabel="Residual")

    fig_distribution, ax_distribution = plt.subplots(figsize=RESULT_CHART_SIZE)
    ax_distribution.hist(sample["residual"], bins=35, color=TEAL, edgecolor="white")
    ax_distribution.set(title="Residual distribution", xlabel="Residual", ylabel="Count")
    return (finish_chart(fig_scores, ax_scores), finish_chart(fig_coeff, ax_coeff),
            finish_chart(fig_residual, ax_residual), finish_chart(fig_distribution, ax_distribution))


@st.cache_data(show_spinner=False)
def read_csv_url(url: str) -> pd.DataFrame:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("請輸入 http 或 https 的 CSV 網址。")
    with urlopen(url, timeout=20) as response:
        content = response.read(MAX_CUSTOM_BYTES + 1)
    if len(content) > MAX_CUSTOM_BYTES:
        raise ValueError("自訂 CSV 超過 50 MB，請使用較小的資料集。")
    return pd.read_csv(BytesIO(content))


def read_train_from_zip(content: bytes) -> pd.DataFrame:
    with ZipFile(BytesIO(content)) as archive:
        train_files = [name for name in archive.namelist() if name.lower().endswith("train.csv")]
        if not train_files:
            raise ValueError("ZIP 中找不到 train.csv。")
        with archive.open(train_files[0]) as train_file:
            return pd.read_csv(train_file)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    if not content:
        raise ValueError("上傳的檔案是空的。")
    if len(content) > MAX_CUSTOM_BYTES:
        raise ValueError("上傳檔案超過 50 MB，請使用較小的資料集。")
    if uploaded_file.name.lower().endswith(".zip"):
        return read_train_from_zip(content)
    return pd.read_csv(BytesIO(content))


@st.cache_data(show_spinner=False)
def read_dataset_from_github(archive_name: str, csv_name: str) -> pd.DataFrame:
    with urlopen(DATA_BASE_URL + archive_name, timeout=90) as response:
        content = response.read()
    with ZipFile(BytesIO(content)) as archive:
        with archive.open(csv_name) as csv_file:
            frame = pd.read_csv(csv_file)
    frame.columns = frame.columns.str.strip()
    return frame


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric)
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def to_dense(values) -> np.ndarray:
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("CSV 沒有資料列。")
    if frame.columns.duplicated().any():
        raise ValueError("CSV 含重複欄名，請先修正。")
    if len(frame.columns) < 2:
        raise ValueError("至少需要一個特徵欄位與一個目標欄位。")
    if not len(frame.select_dtypes(include=np.number).columns):
        raise ValueError("沒有數值欄位可作為回歸目標。")
    return frame


def selectable_feature_limit(frame: pd.DataFrame, target: str) -> int:
    """Estimate the encoded feature count before the model has been fitted."""
    features = frame.drop(columns=[target])
    features = features.drop(columns=[column for column in features if column.lower() in {"id", "index", "unnamed: 0"}])
    numeric_count = len(features.select_dtypes(include=np.number).columns)
    categorical = features.select_dtypes(exclude=np.number)
    category_counts = categorical.nunique(dropna=True)
    encoded_count = numeric_count + int(category_counts[(category_counts > 1) & (category_counts <= 200)].sum())
    return min(40, max(1, encoded_count))


def run_regression(
    data: pd.DataFrame, target: str, requested_k: int, test_size: float
) -> Tuple[dict, pd.DataFrame, plt.Figure, plt.Figure, dict]:
    clean = data.dropna(subset=[target]).copy()
    y = pd.to_numeric(clean[target], errors="coerce")
    valid_target = y.notna() & np.isfinite(y)
    clean = clean.loc[valid_target].copy()
    y = y.loc[valid_target]
    X = clean.drop(columns=[target])
    source_feature_count = X.shape[1]

    id_columns = [column for column in X.columns if column.lower() in {"id", "index", "unnamed: 0"}]
    X = X.drop(columns=id_columns)
    all_missing_columns = X.columns[X.isna().all()].tolist()
    X = X.dropna(axis=1, how="all")
    for column in X.select_dtypes(include=np.number).columns:
        X[column] = X[column].replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    high_cardinality = [column for column in X.select_dtypes(exclude=np.number).columns if X[column].nunique(dropna=True) > 200]
    X = X.drop(columns=high_cardinality)
    constant_columns = X.columns[X.nunique(dropna=True) <= 1].tolist()
    X = X.drop(columns=constant_columns)
    if X.shape[1] == 0:
        raise ValueError("沒有可用的預測特徵。")
    if len(X) < 30:
        raise ValueError("至少需要 30 筆含目標值的資料，才能進行訓練與測試切分。")
    if y.nunique() < 2:
        raise ValueError("目標欄位只有一種有效數值，無法建立回歸模型。")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("訓練集或測試集的目標值只有一種，請增加資料或調整測試比例。")
    preprocessor = make_preprocessor(X_train)
    train_transformed = preprocessor.fit_transform(X_train)
    if train_transformed.shape[1] > MAX_ENCODED_FEATURES:
        raise ValueError(f"類別編碼後超過 {MAX_ENCODED_FEATURES} 個欄位，請先減少高基數類別欄。")
    test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    if train_transformed.shape[1] == 0:
        raise ValueError("前處理後沒有可用的特徵。")
    k = min(requested_k, train_transformed.shape[1], len(X_train) - 6)
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = to_dense(selector.fit_transform(train_transformed, y_train))
    X_test_selected = to_dense(selector.transform(test_transformed))
    selected_names = feature_names[selector.get_support()].tolist()

    model = LinearRegression()
    model.fit(X_train_selected, y_train)
    prediction = model.predict(X_test_selected)
    residuals = y_test.to_numpy() - prediction

    single_index = int(np.argmax(np.nan_to_num(selector.scores_, nan=-np.inf)))
    single_train = to_dense(train_transformed[:, [single_index]])
    single_test = to_dense(test_transformed[:, [single_index]])
    single_model = LinearRegression().fit(single_train, y_train)
    single_prediction = single_model.predict(single_test)

    mae = mean_absolute_error(y_test, prediction)
    rmse = mean_squared_error(y_test, prediction) ** 0.5
    r2 = r2_score(y_test, prediction)

    # 95% prediction interval for a future observation: yhat ± t * sqrt(MSE * (1 + h)).
    design = np.column_stack([np.ones(len(X_train_selected)), X_train_selected])
    test_design = np.column_stack([np.ones(len(X_test_selected)), X_test_selected])
    degrees_of_freedom = max(len(design) - design.shape[1], 1)
    train_residuals = y_train.to_numpy() - model.predict(X_train_selected)
    mse = float(np.sum(train_residuals**2) / degrees_of_freedom)
    covariance = np.linalg.pinv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", test_design, covariance, test_design)
    critical_t = stats.t.ppf(0.975, degrees_of_freedom)
    interval_radius = critical_t * np.sqrt(mse * (1 + leverage))

    result = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": prediction,
            "residual": residuals,
            "lower_95_prediction": prediction - interval_radius,
            "upper_95_prediction": prediction + interval_radius,
        },
        index=y_test.index,
    ).sort_values("predicted")
    result["absolute_error"] = result["residual"].abs()
    feature_scores = pd.DataFrame({"feature": feature_names, "score": selector.scores_}).replace([np.inf, -np.inf], np.nan).dropna().sort_values("score", ascending=False)
    coefficients = pd.DataFrame({"feature": selected_names, "coefficient": model.coef_})
    coefficients["standardized"] = coefficients["coefficient"] * X_train_selected.std(axis=0) / max(float(y_train.std()), 1e-12)
    comparison = pd.DataFrame([
        {"模型": "單一特徵", "特徵數": 1,
         "MAE": mean_absolute_error(y_test, single_prediction),
         "RMSE": mean_squared_error(y_test, single_prediction) ** 0.5,
         "R²": r2_score(y_test, single_prediction)},
        {"模型": "多元特徵", "特徵數": k, "MAE": mae, "RMSE": rmse, "R²": r2},
    ])
    numeric_columns = X_train.select_dtypes(include=np.number).columns.tolist()
    numeric_bounds = {column: (X_train[column].min(), X_train[column].max()) for column in numeric_columns}
    categorical_columns = [column for column in X_train.columns if column not in numeric_columns]
    known_categories = {column: set(X_train[column].dropna().astype(str)) for column in categorical_columns}
    model_bundle = {
        "preprocessor": preprocessor, "selector": selector, "model": model,
        "covariance": covariance, "mse": mse, "critical_t": critical_t,
        "input_columns": X_train.columns.tolist(), "numeric_columns": numeric_columns,
        "numeric_bounds": numeric_bounds, "known_categories": known_categories,
        "default_row": X_train.iloc[[0]].reset_index(drop=True),
        "X_train": X_train, "y_train": y_train,
    }

    fig_scatter, axis = plt.subplots(figsize=RESULT_CHART_SIZE)
    plot_result = result.sample(n=min(5_000, len(result)), random_state=42)
    axis.scatter(plot_result["actual"], plot_result["predicted"], s=12, alpha=0.35, color=BLUE)
    limits = [min(result["actual"].min(), result["predicted"].min()), max(result["actual"].max(), result["predicted"].max())]
    axis.plot(limits, limits, "r--", label="Perfect prediction")
    axis.set(xlabel="Actual target", ylabel="Predicted target", title="Actual vs. Predicted")
    axis.legend()
    fig_scatter.tight_layout()

    fig_interval, axis = plt.subplots(figsize=RESULT_CHART_SIZE)
    shown = result.head(min(200, len(result)))
    x_axis = np.arange(len(shown))
    axis.fill_between(
        x_axis,
        shown["lower_95_prediction"],
        shown["upper_95_prediction"],
        color="#90caf9",
        alpha=0.5,
        label="95% prediction interval",
    )
    axis.plot(x_axis, shown["predicted"], color="#1565c0", label="Prediction")
    axis.scatter(x_axis, shown["actual"], color="#d32f2f", s=15, label="Actual")
    axis.set(xlabel="Test observations sorted by prediction", ylabel=target, title="Prediction interval")
    axis.legend()
    fig_interval.tight_layout()

    return (
        {
            "rows": len(clean),
            "source_features": source_feature_count,
            "raw_features": X.shape[1],
            "dropped_features": [*id_columns, *all_missing_columns, *high_cardinality, *constant_columns],
            "encoded_features": train_transformed.shape[1],
            "selected_features": selected_names,
            "feature_scores": feature_scores,
            "coefficients": coefficients,
            "single_feature": feature_names[single_index],
            "comparison": comparison,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "coverage": float(
                ((result.actual >= result.lower_95_prediction) & (result.actual <= result.upper_95_prediction)).mean()
            ),
        },
        result,
        fig_scatter,
        fig_interval,
        model_bundle,
    )


def feature_count_cv(bundle: dict, max_rows: int = 3_000) -> pd.DataFrame:
    """Compare feature counts using training rows only; refit preprocessing in each fold."""
    X = bundle["X_train"]
    y = bundle["y_train"]
    if len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=17)
        y = y.loc[X.index]

    fold_data = []
    splitter = KFold(n_splits=3, shuffle=True, random_state=17)
    for train_indices, valid_indices in splitter.split(X):
        fold_train, fold_valid = X.iloc[train_indices], X.iloc[valid_indices]
        y_train, y_valid = y.iloc[train_indices], y.iloc[valid_indices]
        if y_train.nunique() < 2 or y_valid.nunique() < 2:
            raise ValueError("交叉驗證的某個區段目標值沒有變化，請使用更多資料。")
        preprocessor = make_preprocessor(fold_train)
        train_values = preprocessor.fit_transform(fold_train)
        valid_values = preprocessor.transform(fold_valid)
        scores, _ = f_regression(train_values, y_train)
        ranking = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
        fold_data.append((train_values, valid_values, y_train, y_valid, ranking))

    max_k = min(40, *(min(train_values.shape[1], len(y_train) - 6)
                      for train_values, _, y_train, _, _ in fold_data))
    if max_k < 1:
        raise ValueError("交叉驗證沒有足夠的樣本或特徵。")
    rows = []
    for k in range(1, max_k + 1):
        fold_rmse, fold_r2 = [], []
        for train_values, valid_values, y_train, y_valid, ranking in fold_data:
            train_selected = to_dense(train_values[:, ranking[:k]])
            valid_selected = to_dense(valid_values[:, ranking[:k]])
            prediction = LinearRegression().fit(train_selected, y_train).predict(valid_selected)
            fold_rmse.append(mean_squared_error(y_valid, prediction) ** 0.5)
            fold_r2.append(r2_score(y_valid, prediction))
        rows.append({"特徵數": k, "CV RMSE": np.mean(fold_rmse), "CV R²": np.mean(fold_r2)})
    return pd.DataFrame(rows)


def feature_count_charts(curve: pd.DataFrame):
    best = curve.loc[curve["CV RMSE"].idxmin()]
    figures = []
    for metric, color, title in [
        ("CV RMSE", BLUE, "RMSE by number of features"),
        ("CV R²", TEAL, "R² by number of features"),
    ]:
        fig, axis = plt.subplots(figsize=RESULT_CHART_SIZE)
        axis.plot(curve["特徵數"], curve[metric], marker="o", markersize=3, color=color)
        axis.axvline(best["特徵數"], color=ORANGE, linestyle="--", label="Lowest CV RMSE")
        axis.set(title=title, xlabel="Selected encoded features", ylabel=metric)
        axis.legend(loc="best")
        figures.append(finish_chart(fig, axis))
    return figures


def predict_one(bundle: dict, edited_row: pd.DataFrame) -> tuple[float, float, float, list[str]]:
    if len(edited_row) != 1:
        raise ValueError("請保留一筆輸入資料。")
    if set(edited_row.columns) != set(bundle["input_columns"]):
        raise ValueError("輸入欄位已變動，請重新載入範例資料。")
    row = edited_row[bundle["input_columns"]].copy()
    notices = []
    for column in bundle["numeric_columns"]:
        value = row.iloc[0][column]
        if pd.isna(value) or value == "":
            row[column] = np.nan
            continue
        numeric_value = pd.to_numeric(value, errors="coerce")
        if not np.isfinite(numeric_value):
            raise ValueError(f"「{column}」必須是有限的數字。")
        row[column] = float(numeric_value)
        lower, upper = bundle["numeric_bounds"][column]
        if numeric_value < lower or numeric_value > upper:
            notices.append(f"{column} 超出訓練資料範圍")
    for column, known in bundle["known_categories"].items():
        value = row.iloc[0][column]
        if pd.notna(value) and str(value) not in known:
            notices.append(f"{column} 是未見過的類別")

    transformed = bundle["preprocessor"].transform(row)
    selected = to_dense(bundle["selector"].transform(transformed))
    prediction = float(bundle["model"].predict(selected)[0])
    design = np.r_[1.0, selected[0]]
    leverage = max(float(design @ bundle["covariance"] @ design), 0.0)
    radius = float(bundle["critical_t"] * np.sqrt(bundle["mse"] * (1 + leverage)))
    return prediction, prediction - radius, prediction + radius, notices


st.set_page_config(page_title="HW2 Multiple Linear Regression", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+TC:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans','Noto Sans TC',sans-serif;}
.stApp {background:#f4f7fb;color:#18324b;}
.block-container {max-width:1320px;padding-top:1.6rem;padding-bottom:4rem;}
.hero {padding:2.1rem 2.4rem;border-radius:22px;background:linear-gradient(110deg,#143657 0%,#1b5b88 58%,#147f8b 100%);box-shadow:0 14px 34px #19365a21;margin-bottom:1.8rem;}
.hero .eyebrow {font-size:.8rem;letter-spacing:.16em;text-transform:uppercase;color:#b8f0ed;font-weight:700;}
.hero h1 {color:#fff !important;margin:.45rem 0;font-size:2.45rem;line-height:1.18;}
.hero p {color:#e9f6fb !important;margin:0;font-size:1.02rem;}
[data-testid="stMetric"] {background:#fff;border:1px solid #dce6f0;border-radius:14px;padding:1rem 1.15rem;box-shadow:0 5px 18px #1539580c;}
[data-testid="stMetricLabel"] {color:#51677c;}
[data-testid="stMetricValue"] {color:#173f65;}
h2,h3 {color:#183f63;letter-spacing:-.02em;}
.section-note {color:#536b80;font-size:.92rem;margin-top:-.55rem;margin-bottom:1rem;}
div[data-testid="stTabs"] [role="tabpanel"] {min-height:520px;}
div[data-testid="stDownloadButton"] button {background:#1768a8;color:#fff;border:1px solid #1768a8;font-weight:600;}
div[data-testid="stDownloadButton"] button:hover {background:#105483;color:#fff;border-color:#105483;}
@media (max-width:760px) {.hero {padding:1.45rem 1.3rem;}.hero h1 {font-size:1.9rem;}}
</style>""", unsafe_allow_html=True)
st.markdown("""<div class="hero"><div class="eyebrow">AIoT · HW2 · CRISP-DM</div>
<h1>Multiple Linear Regression Lab</h1>
<p>探索資料、挑選特徵、評估模型，並檢視 95% 預測區間。</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("資料集與問題")
    source = st.selectbox(
        "選擇問題／資料來源",
        [*DATASETS, "自訂 CSV 網址", "上傳 CSV 檔案"],
    )

data = None
default_target = None
if source in DATASETS:
    try:
        archive_name, csv_name, default_target = DATASETS[source]
        with st.spinner("正在從 GitHub 載入資料集…"):
            data = read_dataset_from_github(archive_name, csv_name)
        st.success(f"已載入 {source.split('（')[0]} 資料集。")
    except Exception as exc:
        st.error(f"無法從 GitHub 載入資料集：{exc}")
elif source == "自訂 CSV 網址":
    url = st.text_input("CSV 直接網址", placeholder="https://example.com/data.csv")
    if url:
        try:
            data = read_csv_url(url)
            st.success("已讀取自訂資料集。")
        except Exception as exc:
            st.error(f"無法讀取網址：{exc}")
else:
    uploaded = st.file_uploader("上傳 CSV", type="csv", key="custom")
    if uploaded:
        try:
            data = read_uploaded_csv(uploaded)
            st.success("已載入上傳的 CSV 資料集。")
        except Exception as exc:
            st.error(f"無法讀取 CSV：{exc}")

if data is None:
    st.stop()
try:
    data = validate_dataset(data)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if source in SOURCES:
    source_url, topic = SOURCES[source]
    st.caption(f"{topic}　·　[查看 Kaggle 資料來源]({source_url})")

dataset_identity = source
if source == "自訂 CSV 網址":
    dataset_identity = (source, url)
elif source == "上傳 CSV 檔案" and uploaded:
    dataset_identity = (source, sha256(uploaded.getvalue()).hexdigest())

st.subheader("01 · Data Understanding｜資料理解")
st.markdown('<div class="section-note">先確認資料規模與品質，再瀏覽所有資料列。</div>', unsafe_allow_html=True)
data_metrics = st.columns(4)
data_metrics[0].metric("資料筆數", f"{len(data):,}")
data_metrics[1].metric("欄位數", len(data.columns))
data_metrics[2].metric("缺失值總數", f"{int(data.isna().sum().sum()):,}")
data_metrics[3].metric("數值欄位數", len(data.select_dtypes(include=np.number).columns))
page_size = 5_000
page_count = (len(data) + page_size - 1) // page_size
page = st.number_input("資料頁碼（可瀏覽全部資料）", min_value=1, max_value=page_count, value=1)
first_row = (page - 1) * page_size
st.caption(f"顯示第 {first_row + 1:,}–{min(first_row + page_size, len(data)):,} 筆，共 {len(data):,} 筆；表格內可捲動。")
st.dataframe(data.iloc[first_row:first_row + page_size], height=400, use_container_width=True)
with st.expander("查看所有欄位與資料品質"):
    quality_sample = data.sample(n=min(len(data), 10_000), random_state=42)
    quality = pd.DataFrame({
        "類型": data.dtypes.astype(str),
        "缺失筆數": data.isna().sum(),
        "缺失率": (data.isna().mean() * 100).round(2).astype(str) + "%",
        "不同值數（抽樣）": quality_sample.nunique(dropna=True),
    })
    st.dataframe(quality, use_container_width=True, height=360)

numeric_targets = data.select_dtypes(include=np.number).columns.tolist()
if not numeric_targets:
    st.error("資料集沒有數值目標欄位，無法做線性回歸。")
    st.stop()
target_index = numeric_targets.index(default_target) if default_target in numeric_targets else 0
if st.session_state.get("chosen_target") not in numeric_targets:
    st.session_state["chosen_target"] = numeric_targets[target_index]
target = st.selectbox("選擇預測目標欄位", numeric_targets, index=target_index, key="chosen_target")

st.divider()
st.subheader("02 · Data Profile｜探索分析")
st.markdown('<div class="section-note">分布與相關性使用至多 10,000 筆固定抽樣；缺失值以全資料計算。</div>', unsafe_allow_html=True)
profile_key = (dataset_identity, target)
if st.session_state.get("profile_key") != profile_key:
    st.session_state["profile_charts"] = data_profile(data, target)
    st.session_state["profile_key"] = profile_key
profile_cols = st.columns(3)
for column, figure in zip(profile_cols, st.session_state["profile_charts"]):
    column.pyplot(figure, use_container_width=True)

st.divider()
st.subheader("03 · Modeling｜建模設定")
st.markdown('<div class="section-note">調整設定後，模型與評估結果會自動更新。</div>', unsafe_allow_html=True)
control_1, control_2 = st.columns(2)
with control_1:
    feature_limit = selectable_feature_limit(data, target)
    requested_k = st.slider("保留特徵數（編碼後）", 1, feature_limit, min(20, feature_limit))
with control_2:
    test_percentage = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)

analysis_key = (dataset_identity, target, requested_k, test_percentage)
if st.session_state.get("analysis_key") != analysis_key:
    try:
        with st.spinner("正在進行資料前處理、特徵選擇與模型評估…"):
            model_data = data.sample(n=100_000, random_state=42) if len(data) > 100_000 else data
            summary, result, scatter_figure, interval_figure, bundle = run_regression(
                model_data, target, requested_k, test_percentage
            )
        st.session_state["analysis"] = (summary, result, scatter_figure, interval_figure, target, bundle)
        st.session_state["diagnostics"] = diagnostic_charts(summary, result)
        st.session_state["analysis_key"] = analysis_key
    except Exception as exc:
        st.session_state.pop("analysis", None)
        st.session_state.pop("analysis_key", None)
        st.session_state.pop("diagnostics", None)
        st.error(f"無法完成分析：{exc}")

if "analysis" in st.session_state:
    summary, result, scatter_figure, interval_figure, target, bundle = st.session_state["analysis"]
    cv_key = (dataset_identity, target, test_percentage)
    if st.session_state.get("cv_key") != cv_key:
        try:
            with st.spinner("正在比較不同特徵數的交叉驗證結果…"):
                curve = feature_count_cv(bundle)
                st.session_state["cv_curve"] = curve
                st.session_state["cv_charts"] = feature_count_charts(curve)
                st.session_state.pop("cv_error", None)
        except Exception as exc:
            st.session_state.pop("cv_curve", None)
            st.session_state.pop("cv_charts", None)
            st.session_state["cv_error"] = str(exc)
        st.session_state["cv_key"] = cv_key
    st.divider()
    st.subheader("04 · Evaluation｜模型評估")
    metric_columns = st.columns(4)
    metric_columns[0].metric("MAE", f"{summary['mae']:.4f}")
    metric_columns[1].metric("RMSE", f"{summary['rmse']:.4f}")
    metric_columns[2].metric("R²", f"{summary['r2']:.4f}")
    metric_columns[3].metric("95% Interval Coverage", f"{summary['coverage']:.1%}")

    st.markdown("**單一特徵與多元特徵比較**")
    st.dataframe(summary["comparison"].round(4), hide_index=True, use_container_width=True)
    st.caption(f"單一模型使用訓練資料選出的「{summary['single_feature']}」；兩個模型使用相同的測試資料。")

    st.write(
        f"使用 {summary['rows']:,} 筆有效資料（訓練 {summary['train_rows']:,}／測試 {summary['test_rows']:,}）、"
        f"{summary['source_features']} 個來源特徵、{summary['raw_features']} 個可用特徵；"
        f"類別編碼後共有 {summary['encoded_features']} 個欄位，"
        f"SelectKBest 選出 {len(summary['selected_features'])} 個。"
    )
    if summary["dropped_features"]:
        st.info("已排除 ID、全空、常數或種類過多的欄位：" + "、".join(summary["dropped_features"]))
    if len(summary["selected_features"]) < requested_k:
        st.caption("依可用特徵數與訓練樣本數，自動調整了實際選取的特徵數。")
    if len(data) > 100_000:
        st.caption("網頁模型使用固定亂數種子的 100,000 筆樣本，以便調整設定時快速更新；資料預覽仍可瀏覽全部資料。")
    with st.expander("查看完整特徵排名、係數與入選欄位"):
        st.write("入選特徵：", summary["selected_features"])
        table_1, table_2 = st.columns(2)
        table_1.dataframe(summary["feature_scores"], use_container_width=True, height=360)
        table_2.dataframe(summary["coefficients"], use_container_width=True, height=360)
    tab_prediction, tab_features, tab_count, tab_residuals = st.tabs(
        ["📈 預測表現", "🧭 特徵解讀", "🔢 特徵數比較", "🧪 殘差診斷"]
    )
    with tab_prediction:
        chart_1, chart_2 = st.columns(2)
        chart_1.pyplot(scatter_figure, clear_figure=False, use_container_width=True)
        chart_2.pyplot(interval_figure, clear_figure=False, use_container_width=True)
        st.caption("藍色區域是 95% 預測區間，覆蓋率為測試資料實際值落入區間的比例。")
    with tab_features:
        chart_1, chart_2 = st.columns(2)
        chart_1.pyplot(st.session_state["diagnostics"][0], clear_figure=False, use_container_width=True)
        chart_2.pyplot(st.session_state["diagnostics"][1], clear_figure=False, use_container_width=True)
        st.caption("F 分數用於選擇特徵；標準化係數顯示選出特徵的方向與相對影響。")
    with tab_count:
        if "cv_curve" in st.session_state:
            chart_1, chart_2 = st.columns(2)
            chart_1.pyplot(st.session_state["cv_charts"][0], clear_figure=False, use_container_width=True)
            chart_2.pyplot(st.session_state["cv_charts"][1], clear_figure=False, use_container_width=True)
            best_k = int(st.session_state["cv_curve"].loc[st.session_state["cv_curve"]["CV RMSE"].idxmin(), "特徵數"])
            st.caption(f"訓練集內三折交叉驗證，至多抽樣 3,000 筆；最低平均 RMSE 出現在 {best_k} 個特徵。測試集未參與此選擇。")
            with st.expander("查看每個特徵數的完整驗證結果"):
                st.dataframe(st.session_state["cv_curve"].round(4), hide_index=True, use_container_width=True)
        else:
            st.warning("無法計算特徵數比較：" + st.session_state.get("cv_error", "未知錯誤"))
    with tab_residuals:
        chart_1, chart_2 = st.columns(2)
        chart_1.pyplot(st.session_state["diagnostics"][2], clear_figure=False, use_container_width=True)
        chart_2.pyplot(st.session_state["diagnostics"][3], clear_figure=False, use_container_width=True)
        st.caption("理想情況下，殘差應在零附近分布，且不隨預測值出現明顯模式。")

    st.divider()
    st.subheader("05 · Error Analysis｜誤差最大的 10 筆")
    st.markdown('<div class="section-note">依實際值與預測值的差距排序，可用來檢查模型在哪些資料上預測不準。</div>', unsafe_allow_html=True)
    outliers = result.nlargest(10, "absolute_error").rename(columns={
        "actual": "實際值", "predicted": "預測值", "residual": "殘差",
        "lower_95_prediction": "95% 預測下界", "upper_95_prediction": "95% 預測上界",
        "absolute_error": "絕對誤差",
    }).rename_axis("原始列號").round(2)
    st.dataframe(outliers, use_container_width=True, height=390)
    st.download_button(
        "下載預測結果 CSV",
        data=result.to_csv(index_label="original_row").encode("utf-8-sig"),
        file_name="multiple_linear_regression_predictions.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("06 · Deployment｜輸入一筆資料試預測")
    st.markdown('<div class="section-note">點選表格中的儲存格修改數值；完成編輯後會自動重新計算。表格可水平捲動查看全部欄位。</div>', unsafe_allow_html=True)
    editor_key = "prediction_editor_" + sha256(repr(analysis_key).encode()).hexdigest()[:12]
    edited_row = st.data_editor(bundle["default_row"], hide_index=True, num_rows="fixed",
                                use_container_width=True, key=editor_key)
    try:
        value, lower, upper, notices = predict_one(bundle, edited_row)
        prediction_metrics = st.columns(3)
        prediction_metrics[0].metric("預測值", f"{value:,.3f}")
        prediction_metrics[1].metric("95% 預測下界", f"{lower:,.3f}")
        prediction_metrics[2].metric("95% 預測上界", f"{upper:,.3f}")
        if notices:
            st.warning("此筆資料與訓練資料不同：" + "、".join(notices) + "；預測可能較不可靠。")
        st.caption("預測區間為線性回歸假設下的估計，並非每筆個別預測都保證落在區間內。")
    except Exception as exc:
        st.error(f"無法預測這筆資料：{exc}")

with st.expander("CRISP-DM workflow"):
    st.markdown(
        "1. **Business Understanding**：預測連續型數值目標。\n"
        "2. **Data Understanding**：顯示列數、欄數、缺失值與資料預覽。\n"
        "3. **Data Preparation**：移除 ID、補值、類別 One-Hot Encoding、訓練／測試切分。\n"
        "4. **Modeling**：SelectKBest 特徵選擇後訓練 `LinearRegression`，並用訓練集交叉驗證比較特徵數。\n"
        "5. **Evaluation**：比較單一與多元模型的 MAE、RMSE、R²，檢視預測圖、95% 預測區間與離群值。\n"
        "6. **Deployment**：網頁可載入預設或自訂 CSV，並輸入一筆新資料取得預測值與區間。"
    )
