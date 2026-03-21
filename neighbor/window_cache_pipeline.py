from __future__ import annotations

import os
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence, Optional

import polars as pl

from data.massive_data_adapter import _normalize_freq_name
from neighbor.build_neighbors import (
    make_symbol_nn_feature_oos,
    make_time_nn_feature_oos,
)


@dataclass
class RollingNeighborConfig:
    ts_col: str = "ts"
    symbol_col: str = "symbol"

    train_window: str = "3h"
    step: str = "5m"
    train_subsample: str = "200ms"
    query_subsample: str = "20ms"

    feature_cols: tuple[str, ...] = (
        "vwap",
        "imbalance",
        "spread",
        "log_return",
        "vol_5m_lag0",
    )
    base_cols: tuple[str, ...] = ("y_vol_5m",)

    n_list: tuple[int, ...] = (5, 10)
    use_time_neighbors: bool = True
    use_symbol_neighbors: bool = True

    metric: str = "minkowski"
    p: float = 2.0
    base_coins: tuple[str, ...] | None = None


def _window_cache_dir(
    out_root: str,
    date: str,
    train_window: str,
    step: str,
    train_subsample: str,
    query_subsample: str,
) -> str:
    name = (
        f"window_cache_"
        f"tw_{_normalize_freq_name(train_window)}_"
        f"step_{_normalize_freq_name(step)}_"
        f"train_{_normalize_freq_name(train_subsample)}_"
        f"query_{_normalize_freq_name(query_subsample)}"
    )
    return os.path.join(out_root, name, date)


def _neighbor_output_dir(
    out_root: str,
    date: str,
    train_window: str,
    step: str,
    train_subsample: str,
    query_subsample: str,
) -> str:
    name = (
        f"neighbor_"
        f"tw_{_normalize_freq_name(train_window)}_"
        f"step_{_normalize_freq_name(step)}_"
        f"train_{_normalize_freq_name(train_subsample)}_"
        f"query_{_normalize_freq_name(query_subsample)}"
    )
    return os.path.join(out_root, name, date)


def _list_cutoff_dirs(window_cache_root: str) -> list[str]:
    if not os.path.exists(window_cache_root):
        raise FileNotFoundError(f"window cache root not found: {window_cache_root}")
    return sorted(
        os.path.join(window_cache_root, d)
        for d in os.listdir(window_cache_root)
        if os.path.isdir(os.path.join(window_cache_root, d))
    )


def run_neighbor_for_one_cutoff(
    cutoff_dir: str,
    output_dir: str,
    config: RollingNeighborConfig,
    *,
    overwrite: bool = True,
    compression: str = "zstd",
    compression_level: int = 3,
) -> str | None:
    """
    把所有 rolling window 的 cutoff 切片并行计算邻居特征，然后把结果写成 parquet。
    """

    ts_col = config.ts_col
    symbol_col = config.symbol_col

    cutoff_name = os.path.basename(cutoff_dir)
    out_path = os.path.join(output_dir, f"{cutoff_name}.parquet")
    if (not overwrite) and os.path.exists(out_path):
        return out_path

    base_query_path = os.path.join(cutoff_dir, "base_query.parquet")
    if not os.path.exists(base_query_path):
        return None

    base_query = pl.read_parquet(base_query_path).sort([ts_col, symbol_col])
    if base_query.is_empty():
        return None

    enriched = base_query
    join_cols = [ts_col, symbol_col]

    for feature_col in config.feature_cols:
        train_path = os.path.join(cutoff_dir, f"{feature_col}_train.parquet")
        query_path = os.path.join(cutoff_dir, f"{feature_col}_query.parquet")
        if (not os.path.exists(train_path)) or (not os.path.exists(query_path)):
            continue

        train_df = pl.read_parquet(train_path).sort([ts_col, symbol_col])
        query_df = pl.read_parquet(query_path).sort([ts_col, symbol_col])

        if train_df.is_empty() or query_df.is_empty():
            continue

        for n in config.n_list:
            if config.use_time_neighbors:
                feat_df = make_time_nn_feature_oos(
                    train_df=train_df,
                    query_df=query_df,
                    feature_col=feature_col,
                    n=n,
                    metric=config.metric,
                    p=config.p,
                    ts_col=ts_col,
                    symbol_col=symbol_col,
                    base_coins=config.base_coins,
                )
                enriched = enriched.join(feat_df, on=join_cols, how="left")

            if config.use_symbol_neighbors:
                feat_df = make_symbol_nn_feature_oos(
                    train_df=train_df,
                    query_df=query_df,
                    feature_col=feature_col,
                    n=n,
                    metric=config.metric,
                    p=config.p,
                    ts_col=ts_col,
                    symbol_col=symbol_col,
                    base_coins=config.base_coins,
                )
                enriched = enriched.join(feat_df, on=join_cols, how="left")

    if enriched.is_empty():
        return None

    os.makedirs(output_dir, exist_ok=True)
    enriched.write_parquet(
        out_path,
        compression=compression,
        compression_level=compression_level,
    )
    return out_path


def load_neighbor_results(
    output_root: str,
    date: str,
    config: RollingNeighborConfig,
) -> pl.DataFrame:

    output_dir = _neighbor_output_dir(
        out_root=output_root,
        date=date,
        train_window=config.train_window,
        step=config.step,
        train_subsample=config.train_subsample,
        query_subsample=config.query_subsample,
    )
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"neighbor output dir not found: {output_dir}")

    files = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".parquet")
    )
    if not files:
        return pl.DataFrame()

    return pl.concat([pl.read_parquet(f) for f in files], how="vertical").sort(
        [config.ts_col, config.symbol_col]
    )
