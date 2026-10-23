"""Interactive web app for HW2 multiple linear regression."""

from __future__ import annotations

from io import BytesIO
from hashlib import sha256
from typing import Tuple
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


STARTUPS_URL = (
    "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/"
    "master/datasets/50_Startups.csv"
)
FLOOD_TARGET = "FloodProbability"
FLOOD_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/lun-09-15/AIot_L6_HW_2/"
    "main/data/flood_prediction_dataset.zip"
)


@st.cache_data(show_spinner=False)
def read_csv_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def read_train_from_zip(content: bytes) -> pd.DataFrame:
    with ZipFile(BytesIO(content)) as archive:
        train_files = [name for name in archive.namelist() if name.lower().endswith("train.csv")]
        if not train_files:
            raise ValueError("ZIP 中找不到 train.csv。")
        with archive.open(train_files[0]) as train_file:
            return pd.read_csv(train_file)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".zip"):
        return read_train_from_zip(content)
    return pd.read_csv(BytesIO(content))


@st.cache_data(show_spinner=False)
def read_flood_from_github() -> pd.DataFrame:
    with urlopen(FLOOD_ARCHIVE_URL, timeout=90) as response:
        return read_train_from_zip(response.read())


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


def run_regression(
    data: pd.DataFrame, target: str, requested_k: int, test_size: float
) -> Tuple[dict, pd.DataFrame, plt.Figure, plt.Figure]:
    clean = data.dropna(subset=[target]).copy()
    y = pd.to_numeric(clean[target], errors="coerce")
    clean = clean.loc[y.notna()].copy()
    y = y.loc[y.notna()]
    X = clean.drop(columns=[target])

    id_columns = [column for column in X.columns if column.lower() in {"id", "index", "unnamed: 0"}]
    X = X.drop(columns=id_columns)
    if X.shape[1] == 0:
        raise ValueError("沒有可用的預測特徵。")
    if len(X) < 30:
        raise ValueError("至少需要 30 筆含目標值的資料，才能進行訓練與測試切分。")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    preprocessor = make_preprocessor(X_train)
    X_train_processed = to_dense(preprocessor.fit_transform(X_train))
    X_test_processed = to_dense(preprocessor.transform(X_test))
    feature_names = preprocessor.get_feature_names_out()

    k = min(requested_k, X_train_processed.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train_processed, y_train)
    X_test_selected = selector.transform(X_test_processed)
    selected_names = feature_names[selector.get_support()].tolist()

    model = LinearRegression()
    model.fit(X_train_selected, y_train)
    prediction = model.predict(X_test_selected)
    residuals = y_test.to_numpy() - prediction

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

    fig_scatter, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(result["actual"], result["predicted"], alpha=0.65, color="#1769aa")
    limits = [min(result["actual"].min(), result["predicted"].min()), max(result["actual"].max(), result["predicted"].max())]
    axis.plot(limits, limits, "r--", label="Perfect prediction")
    axis.set(xlabel="Actual target", ylabel="Predicted target", title="Actual vs. Predicted")
    axis.legend()
    fig_scatter.tight_layout()

    fig_interval, axis = plt.subplots(figsize=(9, 5))
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
            "raw_features": X.shape[1],
            "encoded_features": X_train_processed.shape[1],
            "selected_features": selected_names,
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
    )


st.set_page_config(page_title="HW2 Multiple Linear Regression", layout="wide")
st.title("HW2｜Multiple Linear Regression Web App")
st.caption("CRISP-DM · Feature Selection · Model Evaluation · 95% Prediction Interval")

with st.sidebar:
    st.header("Dataset")
    source = st.selectbox(
        "選擇問題／資料來源",
        [
            "Flood Probability（Kaggle，20 特徵）",
            "50 Startups（範例）",
            "自訂 CSV 網址",
            "上傳 CSV 檔案",
        ],
    )

data = None
default_target = None
if source.startswith("Flood"):
    try:
        with st.spinner("正在從 GitHub 載入 Flood 資料集…"):
            data = read_flood_from_github()
        default_target = FLOOD_TARGET
        st.success("已載入 Flood Probability 資料集。")
    except Exception as exc:
        st.error(f"無法從 GitHub 載入 Flood 資料集：{exc}")
elif source.startswith("50 Startups"):
    try:
        data, default_target = read_csv_url(STARTUPS_URL), "Profit"
        st.success("已載入 50 Startups 範例資料集。")
    except Exception as exc:
        st.error(f"無法讀取範例資料集：{exc}")
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
        data = read_uploaded_csv(uploaded)

if data is None:
    st.stop()

st.subheader("Data Understanding")
left, right = st.columns([2, 1])
with left:
    page_size = 5_000
    page_count = (len(data) + page_size - 1) // page_size
    page = st.number_input("資料頁碼（可瀏覽全部資料）", min_value=1, max_value=page_count, value=1)
    first_row = (page - 1) * page_size
    st.caption(f"顯示第 {first_row + 1:,}–{min(first_row + page_size, len(data)):,} 筆，共 {len(data):,} 筆；表格內可捲動。")
    st.dataframe(data.iloc[first_row:first_row + page_size], height=400, use_container_width=True)
with right:
    st.metric("Rows", f"{len(data):,}")
    st.metric("Columns", len(data.columns))
    st.write("Missing values", int(data.isna().sum().sum()))

numeric_targets = data.select_dtypes(include=np.number).columns.tolist()
if not numeric_targets:
    st.error("資料集沒有數值目標欄位，無法做線性回歸。")
    st.stop()
target_index = numeric_targets.index(default_target) if default_target in numeric_targets else 0

st.subheader("Modeling and Evaluation")
control_1, control_2, control_3 = st.columns(3)
with control_1:
    target = st.selectbox("預測目標欄位", numeric_targets, index=target_index)
with control_2:
    requested_k = st.slider("保留特徵數（編碼後）", 1, min(40, max(1, data.shape[1] - 1)), min(20, max(1, data.shape[1] - 1)))
with control_3:
    test_percentage = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)

dataset_key = source
if source == "自訂 CSV 網址":
    dataset_key = (source, url)
elif source == "上傳 CSV 檔案" and uploaded:
    dataset_key = (source, sha256(uploaded.getvalue()).hexdigest())
analysis_key = (dataset_key, target, requested_k, test_percentage)
if st.session_state.get("analysis_key") != analysis_key:
    try:
        with st.spinner("正在進行資料前處理、特徵選擇與模型評估…"):
            model_data = data.sample(n=100_000, random_state=42) if len(data) > 100_000 else data
            summary, result, scatter_figure, interval_figure = run_regression(
                model_data, target, requested_k, test_percentage
            )
        st.session_state["analysis"] = (summary, result, scatter_figure, interval_figure, target)
        st.session_state["analysis_key"] = analysis_key
    except Exception as exc:
        st.error(f"無法完成分析：{exc}")

if "analysis" in st.session_state:
    summary, result, scatter_figure, interval_figure, target = st.session_state["analysis"]
    st.subheader("Evaluation")
    metric_columns = st.columns(4)
    metric_columns[0].metric("MAE", f"{summary['mae']:.4f}")
    metric_columns[1].metric("RMSE", f"{summary['rmse']:.4f}")
    metric_columns[2].metric("R²", f"{summary['r2']:.4f}")
    metric_columns[3].metric("95% Interval Coverage", f"{summary['coverage']:.1%}")

    st.write(
        f"使用 {summary['rows']:,} 筆資料、{summary['raw_features']} 個原始特徵；"
        f"類別編碼後共有 {summary['encoded_features']} 個欄位，SelectKBest 選出 {len(summary['selected_features'])} 個。"
    )
    if len(data) > 100_000:
        st.caption("網頁模型使用固定亂數種子的 100,000 筆樣本，以便調整設定時快速更新；資料預覽仍可瀏覽全部資料。")
    st.expander("Selected features").write(summary["selected_features"])
    chart_1, chart_2 = st.columns(2)
    chart_1.pyplot(scatter_figure, clear_figure=False)
    chart_2.pyplot(interval_figure, clear_figure=False)

    st.subheader("Top 10 outliers")
    outliers = result.nlargest(10, "absolute_error")
    st.dataframe(outliers, use_container_width=True)
    st.download_button(
        "下載預測結果 CSV",
        data=result.to_csv(index_label="original_row").encode("utf-8-sig"),
        file_name="multiple_linear_regression_predictions.csv",
        mime="text/csv",
    )

with st.expander("CRISP-DM workflow"):
    st.markdown(
        "1. **Business Understanding**：預測連續型數值目標。\n"
        "2. **Data Understanding**：顯示列數、欄數、缺失值與資料預覽。\n"
        "3. **Data Preparation**：移除 ID、補值、類別 One-Hot Encoding、訓練／測試切分。\n"
        "4. **Modeling**：SelectKBest 特徵選擇後訓練 `LinearRegression`。\n"
        "5. **Evaluation**：MAE、RMSE、R²、實際與預測散佈圖、95% 預測區間與離群值。\n"
        "6. **Deployment**：網頁可直接載入預設、URL 或上傳的 CSV 資料集。"
    )
