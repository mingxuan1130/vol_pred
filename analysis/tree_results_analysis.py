from __future__ import annotations

import os
import glob
import math
import polars as pl
import numpy as np
import matplotlib.pyplot as plt


# =========================================
# 工具函数
# =========================================
def load_block_parquets(block_dir: str) -> pl.DataFrame | None:
    """
    读取某个目录下所有 parquet，并纵向拼接。
    如果目录不存在或没有文件，返回 None
    """
    if not os.path.exists(block_dir):
        print(f"[WARN] directory not found: {block_dir}")
        return None

    files = sorted(glob.glob(os.path.join(block_dir, "*.parquet")))
    if not files:
        print(f"[WARN] no parquet files found in: {block_dir}")
        return None

    dfs = []
    for f in files:
        df = pl.read_parquet(f)

        # 从文件名提取 block 名字，方便后面定位
        block_name = os.path.basename(f).replace(".parquet", "")
        df = df.with_columns(pl.lit(block_name).alias("block_name"))

        dfs.append(df)

    out = pl.concat(dfs, how="vertical_relaxed")
    print(f"[INFO] loaded {len(files)} files from {block_dir}, rows={out.height}")
    return out

def safe_print_head(name: str, df: pl.DataFrame | None, n: int = 5):
    print(f"\n===== {name} =====")
    if df is None:
        print("None")
    else:
        print(df.head(n))
        print("shape:", df.shape)
        print("columns:", df.columns)

# =========================================
# Feature importance 
# =========================================
def analyze_feature_importance(imp_df: pl.DataFrame, top_k: int = 20):
    """
    假设 importance parquet 至少包含：
    - feature
    - importance_gain 或 importance
    - importance_split（如果有）
    """
    if imp_df is None or imp_df.is_empty():
        print("[WARN] no importance data")
        return

    cols = imp_df.columns

    gain_col = None
    split_col = None

    # 兼容不同命名
    for c in ["importance_gain", "gain", "importance"]:
        if c in cols:
            gain_col = c
            break

    for c in ["importance_split", "split"]:
        if c in cols:
            split_col = c
            break

    if gain_col is None:
        raise ValueError(f"Cannot find gain-like importance column. got columns={cols}")

    agg_exprs = [
        pl.col(gain_col).mean().alias("gain_mean"),
        pl.col(gain_col).std().alias("gain_std"),
        pl.len().alias("n_blocks"),
    ]

    if split_col is not None:
        agg_exprs.extend([
            pl.col(split_col).mean().alias("split_mean"),
            pl.col(split_col).std().alias("split_std"),
        ])

    imp_mean = (
        imp_df
        .group_by("feature")
        .agg(agg_exprs)
        .sort("gain_mean", descending=True)
    )

    print("\n===== Top features by mean gain =====")
    print(imp_mean.head(top_k))

    # 画 gain top k
    plot_df = imp_mean.head(top_k).to_pandas().iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["feature"], plot_df["gain_mean"])
    plt.xlabel("Mean Gain Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_k} Features by Mean Gain")
    plt.tight_layout()
    plt.show()

    if split_col is not None:
        imp_mean_split = imp_mean.sort("split_mean", descending=True)

        print("\n===== Top features by mean split =====")
        print(imp_mean_split.head(top_k))

        plot_df2 = imp_mean_split.head(top_k).to_pandas().iloc[::-1]

        plt.figure(figsize=(10, 7))
        plt.barh(plot_df2["feature"], plot_df2["split_mean"])
        plt.xlabel("Mean Split Importance")
        plt.ylabel("Feature")
        plt.title(f"Top {top_k} Features by Mean Split")
        plt.tight_layout()
        plt.show()

    return imp_mean

# =========================================
# Prediction 分析
# =========================================
def analyze_predictions(pred_df: pl.DataFrame, top_k: int = 20):
    """
    假设 prediction parquet 至少包含：
    - y_true
    - y_pred
    - symbol（如果有）
    - ts（如果有）
    """
    if pred_df is None or pred_df.is_empty():
        print("[WARN] no prediction data")
        return

    required = {"y_true", "y_pred"}
    if not required.issubset(set(pred_df.columns)):
        raise ValueError(f"prediction df must contain {required}, got {pred_df.columns}")

    pred_df = pred_df.with_columns([
        (pl.col("y_pred") - pl.col("y_true")).alias("error"),
        (pl.col("y_pred") - pl.col("y_true")).abs().alias("abs_error"),
    ])

    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df["y_pred"].to_numpy()

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else math.nan

    print("\n===== Prediction summary =====")
    print(f"rows : {pred_df.height}")
    print(f"rmse : {rmse:.6f}")
    print(f"mae  : {mae:.6f}")
    print(f"corr : {corr:.6f}")

    print("\n===== Worst predictions =====")
    select_cols = [c for c in ["ts", "symbol", "y_true", "y_pred", "error", "abs_error", "block_name"] if c in pred_df.columns]
    print(
        pred_df
        .sort("abs_error", descending=True)
        .select(select_cols)
        .head(top_k)
    )

    # 散点图
    pdf = pred_df.select(["y_true", "y_pred"]).to_pandas()
    if len(pdf) > 5000:
        pdf = pdf.sample(5000, random_state=42)

    plt.figure(figsize=(7, 7))
    plt.scatter(pdf["y_true"], pdf["y_pred"], alpha=0.35, s=10)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("Prediction Scatter")
    plt.tight_layout()
    plt.show()

    # 如果有 symbol，就看每个 symbol 的误差
    if "symbol" in pred_df.columns:
        per_symbol = (
            pred_df
            .group_by("symbol")
            .agg([
                pl.len().alias("n"),
                pl.col("abs_error").mean().alias("mae"),
                ((pl.col("error") ** 2).mean().sqrt()).alias("rmse"),
            ])
            .sort("rmse", descending=True)
        )

        print("\n===== Per-symbol metrics =====")
        print(per_symbol)

    return pred_df

# =========================================
# Metric 分析
# =========================================
def analyze_metrics(metric_df: pl.DataFrame):
    if metric_df is None or metric_df.is_empty():
        print("[WARN] no metric data")
        return

    print("\n===== Metric dataframe =====")
    print(metric_df)

    # 如果有 rmse 列，就画 block 曲线
    if "rmse" in metric_df.columns:
        df_plot = metric_df.sort("block_name").to_pandas()

        plt.figure(figsize=(12, 5))
        plt.plot(df_plot["block_name"], df_plot["rmse"])
        plt.xticks(rotation=90)
        plt.xlabel("block_name")
        plt.ylabel("rmse")
        plt.title("RMSE by Block")
        plt.tight_layout()
        plt.show()

# =========================================
# 读取所有 importance block
# =========================================
def load_importance_blocks(block_dir: str) -> pl.DataFrame:
    files = sorted(glob.glob(os.path.join(block_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files found in {block_dir}")

    dfs = []
    for f in files:
        df = pl.read_parquet(f)
        block_name = os.path.basename(f).replace(".parquet", "")
        df = df.with_columns(pl.lit(block_name).alias("block_name"))
        dfs.append(df)

    out = pl.concat(dfs, how="vertical_relaxed")
    print(f"[INFO] loaded {len(files)} block files, shape={out.shape}")
    print("[INFO] columns:", out.columns)
    return out

# =========================================
# 自动识别 importance 列名
# =========================================
def detect_importance_columns(df: pl.DataFrame):
    cols = df.columns

    feature_col = "feature" if "feature" in cols else None
    if feature_col is None:
        raise ValueError(f"cannot find feature column, got columns={cols}")

    gain_col = None
    for c in ["importance_gain", "gain", "importance"]:
        if c in cols:
            gain_col = c
            break

    split_col = None
    for c in ["importance_split", "split"]:
        if c in cols:
            split_col = c
            break

    if gain_col is None:
        raise ValueError(f"cannot find gain-like column, got columns={cols}")

    return feature_col, gain_col, split_col

# =========================================
# 画 heatmap
# =========================================
def plot_importance_heatmap(
    imp_df: pl.DataFrame,
    value_col: str,
    top_k: int = 20,
    normalize_by_feature: bool = True,
    figsize: tuple[int, int] = (16, 8),
):
    """
    value_col:
        - gain-like column: importance_gain / gain / importance
        - split-like column: importance_split / split

    normalize_by_feature=True:
        每个 feature 自己做 0~1 标准化，更容易看“时间变化模式”
    """
    # 先选 top_k 特征（按整体平均重要性）
    top_features = (
        imp_df
        .group_by("feature")
        .agg(pl.col(value_col).mean().alias("mean_importance"))
        .sort("mean_importance", descending=True)
        .head(top_k)
        .get_column("feature")
        .to_list()
    )

    df_top = imp_df.filter(pl.col("feature").is_in(top_features))

    # 转成 matrix: 行=feature, 列=block_name
    mat = (
        df_top
        .select(["feature", "block_name", value_col])
        .to_pandas()
        .pivot(index="feature", columns="block_name", values=value_col)
        .fillna(0.0)
    )

    # 行顺序按整体平均 importance 排序
    feature_order = (
        df_top
        .group_by("feature")
        .agg(pl.col(value_col).mean().alias("mean_importance"))
        .sort("mean_importance", descending=True)
        .get_column("feature")
        .to_list()
    )
    mat = mat.loc[feature_order]

    # 可选：每个 feature 单独归一化到 0~1
    if normalize_by_feature:
        mat_plot = mat.copy()
        for idx in mat_plot.index:
            row = mat_plot.loc[idx]
            rmin = row.min()
            rmax = row.max()
            if rmax > rmin:
                mat_plot.loc[idx] = (row - rmin) / (rmax - rmin)
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

    # x 轴别全打出来，不然会像天书
    xticks = list(range(len(mat_plot.columns)))
    if len(xticks) > 20:
        step = max(1, len(xticks) // 20)
        xticks_show = xticks[::step]
    else:
        xticks_show = xticks

    plt.xticks(
        xticks_show,
        [mat_plot.columns[i] for i in xticks_show],
        rotation=90
    )

    plt.xlabel("Rolling Block")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance Heatmap: {value_col}{title_suffix}")
    plt.tight_layout()
    plt.show()

    return mat

# =========================================
# 画每个特征的时间曲线
# =========================================
def plot_feature_importance_lines(
    imp_df: pl.DataFrame,
    value_col: str,
    top_k: int = 8,
    figsize: tuple[int, int] = (16, 6),
):
    top_features = (
        imp_df
        .group_by("feature")
        .agg(pl.col(value_col).mean().alias("mean_importance"))
        .sort("mean_importance", descending=True)
        .head(top_k)
        .get_column("feature")
        .to_list()
    )

    pdf = (
        imp_df
        .filter(pl.col("feature").is_in(top_features))
        .select(["feature", "block_name", value_col])
        .to_pandas()
    )

    pivot_df = (
        pdf
        .pivot(index="block_name", columns="feature", values=value_col)
        .fillna(0.0)
        .sort_index()
    )

    plt.figure(figsize=figsize)
    for c in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[c], label=c)

    if len(pivot_df.index) > 20:
        step = max(1, len(pivot_df.index) // 20)
        xticks_show = range(0, len(pivot_df.index), step)
        plt.xticks(
            xticks_show,
            [pivot_df.index[i] for i in xticks_show],
            rotation=90
        )
    else:
        plt.xticks(rotation=90)

    plt.xlabel("Rolling Block")
    plt.ylabel(value_col)
    plt.title(f"Top {top_k} Feature Importance Over Time: {value_col}")
    plt.legend()
    plt.tight_layout()
    plt.show()
