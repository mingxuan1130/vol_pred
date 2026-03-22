from __future__ import annotations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl

from analysis.evals import evaluate


RAW_IMPORTANCE_COLUMNS = (
    "feature",
    "importance_gain",
    "importance_split",
    "block_name",
    "pred_file",
)
MEAN_IMPORTANCE_COLUMNS = (
    "feature",
    "importance_gain_mean",
    "importance_split_mean",
)
PREDICTION_REQUIRED_COLUMNS = ("y_true", "y_pred")
METRIC_PLOT_COLUMNS = ("block_name", "rmse")
IMPORTANCE_VALUE_COLUMNS = ("importance_gain", "importance_split")
__all__ = [
    "load_block_parquets",
    "load_importance_blocks",
    "analyze_feature_importance",
    "analyze_predictions",
    "analyze_metrics",
    "detect_importance_columns",
    "plot_importance_heatmap",
    "plot_feature_importance_lines",
]


# =============================
# Helper functions 
# =============================
def _ensure_columns(df: pl.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}. got columns={df.columns}"
        )


def _load_parquet_dir(directory: Path) -> pl.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found in {directory}")

    frames = [pl.read_parquet(path) for path in files]
    out = pl.concat(frames, how="vertical_relaxed")
    print(f"[INFO] loaded {len(files)} parquet files from {directory}, rows={out.height}")
    return out


def load_parquet_frame(path: str | Path) -> pl.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"path not found: {resolved}")

    if resolved.is_dir():
        return _load_parquet_dir(resolved)

    if resolved.suffix != ".parquet":
        raise ValueError(f"expected a parquet file or directory, got {resolved}")

    df = pl.read_parquet(resolved)
    print(f"[INFO] loaded parquet file: {resolved}, rows={df.height}")
    return df


def load_block_parquets(block_dir: str) -> pl.DataFrame:
    return load_parquet_frame(block_dir)


def load_importance_blocks(path: str) -> pl.DataFrame:
    imp_df = load_parquet_frame(path)
    _ensure_columns(imp_df, RAW_IMPORTANCE_COLUMNS, "importance dataframe")
    return imp_df


def summarize_feature_importance(imp_df: pl.DataFrame) -> pl.DataFrame:
    _ensure_columns(imp_df, RAW_IMPORTANCE_COLUMNS, "importance dataframe")
    return (
        imp_df.group_by("feature")
        .agg(
            [
                pl.col("importance_gain").mean().alias("importance_gain_mean"),
                pl.col("importance_split").mean().alias("importance_split_mean"),
            ]
        )
        .sort("importance_gain_mean", descending=True)
    )


def _resolve_importance_mean_df(imp_df: pl.DataFrame) -> pl.DataFrame:
    if set(MEAN_IMPORTANCE_COLUMNS).issubset(imp_df.columns):
        return imp_df.select(MEAN_IMPORTANCE_COLUMNS).sort(
            "importance_gain_mean",
            descending=True,
        )

    _ensure_columns(imp_df, RAW_IMPORTANCE_COLUMNS, "importance dataframe")
    return summarize_feature_importance(imp_df)

# =============================
# Feature importance analysis
# =============================

def get_top_feature_importance(
    imp_mean_df: pl.DataFrame,
    value_col: str,
    top_k: int = 20,
) -> pl.DataFrame:
    _ensure_columns(imp_mean_df, MEAN_IMPORTANCE_COLUMNS, "importance mean dataframe")
    if value_col not in ("importance_gain_mean", "importance_split_mean"):
        raise ValueError(
            "value_col must be one of "
            "('importance_gain_mean', 'importance_split_mean'), "
            f"got {value_col}"
        )

    return imp_mean_df.sort(value_col, descending=True).head(top_k)


def print_feature_importance_summary(imp_mean_df: pl.DataFrame, top_k: int = 20) -> None:
    top_gain = get_top_feature_importance(
        imp_mean_df=imp_mean_df,
        value_col="importance_gain_mean",
        top_k=top_k,
    )
    print("\n===== Top features by mean gain =====")
    print(top_gain)

    top_split = get_top_feature_importance(
        imp_mean_df=imp_mean_df,
        value_col="importance_split_mean",
        top_k=top_k,
    )
    print("\n===== Top features by mean split =====")
    print(top_split)


def plot_feature_importance_bar(
    imp_mean_df: pl.DataFrame,
    value_col: str,
    top_k: int = 20,
    figsize: tuple[int, int] = (10, 7),
) -> None:
    top_df = get_top_feature_importance(
        imp_mean_df=imp_mean_df,
        value_col=value_col,
        top_k=top_k,
    )
    plot_df = top_df.to_pandas().iloc[::-1]

    plt.figure(figsize=figsize)
    plt.barh(plot_df["feature"], plot_df[value_col])
    plt.xlabel(
        "Mean Gain Importance"
        if value_col == "importance_gain_mean"
        else "Mean Split Importance"
    )
    plt.ylabel("Feature")
    plt.title(
        f"Top {top_k} Features by Mean "
        f"{'Gain' if value_col == 'importance_gain_mean' else 'Split'}"
    )
    plt.tight_layout()
    plt.show()


def analyze_feature_importance(
    imp_df: pl.DataFrame,
    top_k: int = 20,
    plot: bool = True,
) -> pl.DataFrame | None:
    if imp_df is None or imp_df.is_empty():
        print("[WARN] no importance data")
        return None

    imp_mean = _resolve_importance_mean_df(imp_df)
    print_feature_importance_summary(imp_mean, top_k=top_k)

    if plot:
        plot_feature_importance_bar(
            imp_mean_df=imp_mean,
            value_col="importance_gain_mean",
            top_k=top_k,
        )
        plot_feature_importance_bar(
            imp_mean_df=imp_mean,
            value_col="importance_split_mean",
            top_k=top_k,
        )

    return imp_mean


# =============================
# Prediction analysis
# =============================
def prepare_prediction_analysis_frame(pred_df: pl.DataFrame) -> pl.DataFrame:
    _ensure_columns(pred_df, PREDICTION_REQUIRED_COLUMNS, "prediction dataframe")
    return pred_df.with_columns(
        [
            (pl.col("y_pred") - pl.col("y_true")).alias("error"),
            (pl.col("y_pred") - pl.col("y_true")).abs().alias("abs_error"),
        ]
    )


def compute_prediction_metrics(pred_df: pl.DataFrame) -> dict[str, float]:
    prepared_df = prepare_prediction_analysis_frame(pred_df)
    y_true = prepared_df["y_true"].to_numpy()
    y_pred = prepared_df["y_pred"].to_numpy()
    metrics = evaluate(y_true=y_true, y_pred=y_pred)
    metrics["rows"] = float(prepared_df.height)
    return metrics


def get_worst_predictions(pred_df: pl.DataFrame, top_k: int = 20) -> pl.DataFrame:
    prepared_df = prepare_prediction_analysis_frame(pred_df)
    select_cols = [
        column
        for column in ("ts", "symbol", "y_true", "y_pred", "error", "abs_error", "block_name")
        if column in prepared_df.columns
    ]
    return prepared_df.sort("abs_error", descending=True).select(select_cols).head(top_k)


def compute_per_symbol_prediction_metrics(pred_df: pl.DataFrame) -> pl.DataFrame | None:
    """
    按照每个 symbol 计算预测指标
    """
    prepared_df = prepare_prediction_analysis_frame(pred_df)
    if "symbol" not in prepared_df.columns:
        return None

    return (
        prepared_df.group_by("symbol")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("abs_error").mean().alias("mae"),
                ((pl.col("error") ** 2).mean().sqrt()).alias("rmse"),
            ]
        )
        .sort("rmse", descending=True)
    )


def plot_prediction_scatter(
    pred_df: pl.DataFrame,
    max_points: int = 5000,
    figsize: tuple[int, int] = (7, 7),
) -> None:
    """
    y_true vs y_pred 散点图
    """
    prepared_df = prepare_prediction_analysis_frame(pred_df)
    pdf = prepared_df.select(["y_true", "y_pred"]).to_pandas()
    if len(pdf) > max_points:
        pdf = pdf.sample(max_points, random_state=42)

    plt.figure(figsize=figsize)
    plt.scatter(pdf["y_true"], pdf["y_pred"], alpha=0.35, s=10)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("Prediction Scatter")
    plt.tight_layout()
    plt.show()


def analyze_predictions(
    pred_df: pl.DataFrame,
    top_k: int = 20,
    plot: bool = True,
) -> pl.DataFrame | None:
    """
    整合预测分析
    """
    if pred_df is None or pred_df.is_empty():
        print("[WARN] no prediction data")
        return None

    prepared_df = prepare_prediction_analysis_frame(pred_df)
    per_symbol_df = compute_per_symbol_prediction_metrics(prepared_df)

    if plot:
        plot_prediction_scatter(prepared_df)

    return prepared_df


def analyze_metrics(metric_df: pl.DataFrame) -> None:
    if metric_df is None or metric_df.is_empty():
        print("[WARN] no metric data")
        return

    print("\n===== Metric dataframe =====")
    print(metric_df)

    if set(METRIC_PLOT_COLUMNS).issubset(metric_df.columns):
        df_plot = metric_df.sort("block_name").to_pandas()
        plt.figure(figsize=(12, 5))
        plt.plot(df_plot["block_name"], df_plot["rmse"])
        plt.xticks(rotation=90)
        plt.xlabel("block_name")
        plt.ylabel("rmse")
        plt.title("RMSE by Block")
        plt.tight_layout()
        plt.show()


def detect_importance_columns(df: pl.DataFrame) -> tuple[str, str, str]:
    _ensure_columns(df, RAW_IMPORTANCE_COLUMNS, "importance dataframe")
    return "feature", "importance_gain", "importance_split"


def _select_top_features(imp_df: pl.DataFrame, value_col: str, top_k: int) -> list[str]:
    _ensure_columns(imp_df, RAW_IMPORTANCE_COLUMNS, "importance dataframe")
    if value_col not in IMPORTANCE_VALUE_COLUMNS:
        raise ValueError(
            f"value_col must be one of {IMPORTANCE_VALUE_COLUMNS}, got {value_col}"
        )

    return (
        imp_df.group_by("feature")
        .agg(pl.col(value_col).mean().alias("value_mean"))
        .sort("value_mean", descending=True)
        .head(top_k)
        .get_column("feature")
        .to_list()
    )

# =============================
# 外部接口
# =============================

def compute_importance_heatmap_matrix(
    imp_df: pl.DataFrame,
    value_col: str,
    top_k: int = 20,
) -> Any:
    top_features = _select_top_features(imp_df=imp_df, value_col=value_col, top_k=top_k)
    df_top = imp_df.filter(pl.col("feature").is_in(top_features))

    mat = (
        df_top.select(["feature", "block_name", value_col])
        .to_pandas()
        .pivot(index="feature", columns="block_name", values=value_col)
        .fillna(0.0)
    )

    feature_order = (
        df_top.group_by("feature")
        .agg(pl.col(value_col).mean().alias("value_mean"))
        .sort("value_mean", descending=True)
        .get_column("feature")
        .to_list()
    )
    mat = mat.loc[feature_order]

    return mat


def plot_importance_heatmap_matrix(
    mat: Any,
    value_col: str,
    normalize_by_feature: bool = True,
    figsize: tuple[int, int] = (16, 8),
) -> None:
    if normalize_by_feature:
        mat_plot = mat.astype(float).copy()
        for idx in mat_plot.index:
            row = mat_plot.loc[idx]
            row_min = row.min()
            row_max = row.max()
            if row_max > row_min:
                mat_plot.loc[idx] = (row - row_min) / (row_max - row_min)
            else:
                mat_plot.loc[idx] = 0.0
        title_suffix = " (row-normalized 0-1)"
    else:
        mat_plot = mat
        title_suffix = ""

    plt.figure(figsize=figsize)
    plt.imshow(mat_plot.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label=value_col)
    plt.yticks(range(len(mat_plot.index)), mat_plot.index)

    xticks = list(range(len(mat_plot.columns)))
    if len(xticks) > 20:
        step = max(1, len(xticks) // 20)
        xticks_show = xticks[::step]
    else:
        xticks_show = xticks

    plt.xticks(xticks_show, [mat_plot.columns[i] for i in xticks_show], rotation=90)
    plt.xlabel("Rolling Block")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance Heatmap: {value_col}{title_suffix}")
    plt.tight_layout()
    plt.show()


def plot_importance_heatmap(
    imp_df: pl.DataFrame,
    value_col: str,
    top_k: int = 20,
    normalize_by_feature: bool = True,
    figsize: tuple[int, int] = (16, 8),
):
    mat = compute_importance_heatmap_matrix(
        imp_df=imp_df,
        value_col=value_col,
        top_k=top_k,
    )
    plot_importance_heatmap_matrix(
        mat=mat,
        value_col=value_col,
        normalize_by_feature=normalize_by_feature,
        figsize=figsize,
    )
    return mat


def compute_feature_importance_line_frame(
    imp_df: pl.DataFrame,
    value_col: str,
    top_k: int = 8,
) -> Any:
    top_features = _select_top_features(imp_df=imp_df, value_col=value_col, top_k=top_k)
    pdf = (
        imp_df.filter(pl.col("feature").is_in(top_features))
        .select(["feature", "block_name", value_col])
        .to_pandas()
    )

    pivot_df = (
        pdf.pivot(index="block_name", columns="feature", values=value_col)
        .fillna(0.0)
        .sort_index()
    )

    return pivot_df


def plot_feature_importance_line_frame(
    pivot_df: Any,
    value_col: str,
    top_k: int = 8,
    figsize: tuple[int, int] = (16, 6),
) -> None:
    plt.figure(figsize=figsize)
    for column in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[column], label=column)

    if len(pivot_df.index) > 20:
        step = max(1, len(pivot_df.index) // 20)
        xticks_show = range(0, len(pivot_df.index), step)
        plt.xticks(xticks_show, [pivot_df.index[i] for i in xticks_show], rotation=90)
    else:
        plt.xticks(rotation=90)

    plt.xlabel("Rolling Block")
    plt.ylabel(value_col)
    plt.title(f"Top {top_k} Feature Importance Over Time: {value_col}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance_lines(
    imp_df: pl.DataFrame,
    value_col: str,
    top_k: int = 8,
    figsize: tuple[int, int] = (16, 6),
) -> None:
    pivot_df = compute_feature_importance_line_frame(
        imp_df=imp_df,
        value_col=value_col,
        top_k=top_k,
    )
    plot_feature_importance_line_frame(
        pivot_df=pivot_df,
        value_col=value_col,
        top_k=top_k,
        figsize=figsize,
    )
