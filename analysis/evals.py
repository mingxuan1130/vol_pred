from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt


def _to_1d_float_array(x: Iterable) -> np.ndarray:
    arr = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def qlike_from_vol(y_true_vol, y_pred_vol, eps: float = 1e-12) -> float:
    """
    QLIKE on variance, but inputs are VOLATILITY (sigma), not variance.
    So we square both sides first:

        ratio = sigma_true^2 / sigma_pred^2
        qlike = mean(ratio - log(ratio) - 1)
    """
    yt = _to_1d_float_array(y_true_vol)
    yp = _to_1d_float_array(y_pred_vol)

    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    valid = np.isfinite(yt) & np.isfinite(yp)
    if valid.sum() == 0:
        raise ValueError("No valid finite values found in y_true/y_pred.")

    yt = yt[valid]
    yp = yp[valid]

    # 预测的是波动率，先平方成方差
    yt_var = np.clip(yt ** 2, eps, None)
    yp_var = np.clip(yp ** 2, eps, None)

    ratio = yt_var / yp_var
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def evaluate(y_true, y_pred, eps: float = 1e-12) -> dict:
    """
    Evaluate predictions from array-like inputs.
    Includes:
        - rmse
        - mae
        - corr
        - qlike  (computed on squared vols)
        - n
    """
    yt = _to_1d_float_array(y_true)
    yp = _to_1d_float_array(y_pred)

    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    valid = np.isfinite(yt) & np.isfinite(yp)
    if valid.sum() == 0:
        raise ValueError("No valid finite values found in y_true/y_pred.")

    yt_valid = yt[valid]
    yp_valid = yp[valid]

    rmse = float(np.sqrt(np.mean((yt_valid - yp_valid) ** 2)))
    mae = float(np.mean(np.abs(yt_valid - yp_valid)))
    corr = (
        float(np.corrcoef(yt_valid, yp_valid)[0, 1])
        if np.std(yt_valid) > 0 and np.std(yp_valid) > 0
        else np.nan
    )
    qlike = qlike_from_vol(yt_valid, yp_valid, eps=eps)

    return {
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "qlike": qlike,
        "n": int(valid.sum()),
    }


def evaluate_models(
    results: pl.DataFrame,
    y_true_col: str = "y_vol_5m",
    model_cols: list[str] | None = None,
) -> pl.DataFrame:
    """
    评估 DataFrame 中的多个模型预测结果，返回一个包含评估指标的 DataFrame
    """
    if y_true_col not in results.columns:
        raise ValueError(f"Missing y_true column: {y_true_col}")
    if model_cols is None or not all(col in results.columns for col in model_cols):
        raise ValueError("model_cols must be provided and exist in results")

    if results.null_count().select(pl.sum_horizontal(pl.all())).item() != 0:
        raise ValueError("The results DataFrame contains null values.")

    y_true = results.get_column(y_true_col).cast(pl.Float64).to_numpy()

    rows = []
    for col in model_cols:
        y_pred = results.get_column(col).cast(pl.Float64).to_numpy()
        metrics = evaluate(y_true=y_true, y_pred=y_pred)
        rows.append({"model": col, **metrics})

    return pl.DataFrame(rows).sort("rmse")


def evaluate_all_results(
    all_results: pl.DataFrame,
    model_type: str,
    date: str,
    *,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    by_symbol: bool = False,
) -> pl.DataFrame:
    """
    评估 rolling 预测结果。
    """
    required_cols = [y_true_col, y_pred_col]
    missing_cols = [c for c in required_cols if c not in all_results.columns]
    if missing_cols:
        raise ValueError(f"all_results 缺少评估所需列: {missing_cols}")

    if by_symbol:
        if "symbol" not in all_results.columns:
            raise ValueError("by_symbol=True 时 all_results 必须包含 symbol 列")

        rows = []
        for sym in all_results.get_column("symbol").unique().sort().to_list():
            g = all_results.filter(pl.col("symbol") == sym)
            metrics = evaluate(
                y_true=g.get_column(y_true_col).cast(pl.Float64).to_numpy(),
                y_pred=g.get_column(y_pred_col).cast(pl.Float64).to_numpy(),
            )
            rows.append({
                "date": date,
                "model_type": model_type,
                "symbol": sym,
                **metrics,
            })
        return pl.DataFrame(rows).sort("rmse")

    metrics = evaluate(
        y_true=all_results.get_column(y_true_col).cast(pl.Float64).to_numpy(),
        y_pred=all_results.get_column(y_pred_col).cast(pl.Float64).to_numpy(),
    )
    return pl.DataFrame([{
        "date": date,
        "model_type": model_type,
        "symbol": "ALL",
        **metrics,
    }])


def evaluate_result_parquets(
    results_dir: str = "results",
    *,
    run_suffix: str = "lgbm",
    parquet_name: str = "run_results.parquet",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> pl.DataFrame:
    """
    扫描 results_dir 下所有以 run_suffix 结尾的子目录中的 parquet，并做逐文件评估。
    """
    root = Path(results_dir)
    pattern = f"*{run_suffix}/**/{parquet_name}"
    parquet_paths = sorted(root.glob(pattern))
    if not parquet_paths:
        raise ValueError(f"未找到匹配文件: {root / pattern}")

    rows = []
    for p in parquet_paths:
        df = pl.read_parquet(str(p))
        if y_true_col not in df.columns or y_pred_col not in df.columns:
            raise ValueError(f"{p} 缺少评估列: {y_true_col}/{y_pred_col}")

        metrics = evaluate(
            y_true=df.get_column(y_true_col).cast(pl.Float64).to_numpy(),
            y_pred=df.get_column(y_pred_col).cast(pl.Float64).to_numpy(),
        )

        run_dir = next((part for part in p.parts if part.endswith(run_suffix)), "")
        symbol = (
            str(df.get_column("symbol")[0])
            if "symbol" in df.columns and df.height > 0
            else p.parent.name
        )

        rows.append({
            "run_dir": run_dir,
            "symbol": symbol,
            **metrics,
        })

    return pl.DataFrame(rows).sort("rmse")


def plot_5min_metrics_from_dict(
    data_dict: dict,
    *,
    floor_freq: str = "5min",
    tick_freq: str = "30min",
    eps: float = 1e-12,
    check_n: int = 100,
):

    ts = pd.to_datetime(pd.Series(data_dict["ts"])).reset_index(drop=True)
    models = data_dict["models"]

    model_names = list(models.keys())
    ref_name = model_names[0]
    ref_true = pd.Series(models[ref_name]["y_true"]).to_numpy()

    for m in model_names[1:]:
        y = pd.Series(models[m]["y_true"]).to_numpy()

        mask = np.isclose(
            ref_true[:check_n],
            y[:check_n],
            rtol=1e-8,
            atol=1e-4,
        )

        match_count = mask.sum()
        mismatch_count = check_n - match_count

        print(f"\n[y_true check] {ref_name} vs {m}")
        print(f"匹配数量: {match_count}")
        print(f"不匹配数量: {mismatch_count}")

        if mismatch_count > 0:
            idx = np.where(~mask)[0]
            print("不匹配位置:", idx[:10])
            print(f"{ref_name}:", ref_true[idx[:10]])
            print(f"{m}:", y[idx[:10]])


    # 构建长表
    dfs = []
    for model_name, model_data in models.items():

        tmp = pd.DataFrame({
            "ts": ts,
            "model": model_name,
            "y_pred": pd.Series(model_data["y_pred"]),
            "y_true": pd.Series(model_data["y_true"]),
        })

        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True).dropna()

    df["5min_grid"] = df["ts"].dt.floor(floor_freq)

    # ------------------------------------------------
    # 向量化误差
    # ------------------------------------------------
    df["err"] = df["y_pred"] - df["y_true"]
    df["sq_err"] = df["err"] ** 2
    df["abs_err"] = df["err"].abs()

    ratio = df["y_true"] / (df["y_pred"] + eps)
    df["qlike"] = ratio - np.log(ratio + eps) - 1

    # corr helper
    df["x"] = df["y_pred"]
    df["y"] = df["y_true"]
    df["x2"] = df["x"] ** 2
    df["y2"] = df["y"] ** 2
    df["xy"] = df["x"] * df["y"]

    # ------------------------------------------------
    # 4. groupby
    # ------------------------------------------------
    agg = (
        df.groupby(["5min_grid", "model"])
        .agg(
            n=("x", "size"),
            sum_x=("x", "sum"),
            sum_y=("y", "sum"),
            sum_x2=("x2", "sum"),
            sum_y2=("y2", "sum"),
            sum_xy=("xy", "sum"),
            mse=("sq_err", "mean"),
            mae=("abs_err", "mean"),
            qlike=("qlike", "mean"),
        )
        .reset_index()
    )

    agg["rmse"] = np.sqrt(agg["mse"])

    num = agg["n"] * agg["sum_xy"] - agg["sum_x"] * agg["sum_y"]
    den_x = agg["n"] * agg["sum_x2"] - agg["sum_x"] ** 2
    den_y = agg["n"] * agg["sum_y2"] - agg["sum_y"] ** 2
    den = np.sqrt(den_x * den_y)

    agg["corr"] = np.where(den > 0, num / den, np.nan)

    metrics = ["rmse", "mae", "corr", "qlike"]

    # ------------------------------------------------
    # 5. pivot
    # ------------------------------------------------
    wide = {
        m: agg.pivot(index="5min_grid", columns="model", values=m)
        for m in metrics
    }

    # ------------------------------------------------
    # 6. plot
    # ------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    tick_times = pd.date_range(
        start=wide["rmse"].index.min(),
        end=wide["rmse"].index.max(),
        freq=tick_freq
    )

    for i, m in enumerate(metrics):

        ax = axes[i]
        df_plot = wide[m]

        for col in df_plot.columns:
            ax.plot(df_plot.index, df_plot[col], label=col)

        ax.set_title(m.upper())
        ax.set_ylabel(m)

        ax.set_xticks(tick_times)
        ax.set_xticklabels(
            [t.strftime("%H:%M") for t in tick_times],
            rotation=45
        )

        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("time")

    plt.tight_layout()
    plt.show()
