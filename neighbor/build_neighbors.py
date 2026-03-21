from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

def make_symbol_nn_feature_oos(
    train_df: pl.DataFrame,
    query_df: pl.DataFrame,
    feature_col: str,
    n: int,
    *,
    metric: str = "minkowski",
    p: float = 2,
    metric_params: dict | None = None,
    agg: Callable = np.mean,
    ts_col: str = "ts",
    symbol_col: str = "symbol",
    exclude_self: bool = True,
    base_coins: Sequence[str] | None = None,
) -> pl.DataFrame:
    """
    用 train_df 中每个 symbol 的历史路径定义 symbol 邻居，
    然后把这个邻居关系应用到 query_df 的同一时刻上。

    如果提供 base_coins，则每个 symbol 只在这些 base_coins 中找邻居。
    """

    # 对 train_df 和 query_df 都做 pivot，得到以 ts 为行索引，symbol 为列的矩阵
    train_pivot = _pivot_feature(train_df, feature_col, ts_col=ts_col, symbol_col=symbol_col)
    query_pivot = _pivot_feature(query_df, feature_col, ts_col=ts_col, symbol_col=symbol_col)

    # 防御性检查，确保 train 和 query 的 pivot 结果都包含 同样的 symbol 列
    train_pivot, query_pivot, symbol_cols = _align_query_symbols(train_pivot, query_pivot, ts_col=ts_col)

    # 全部 query symbol 的历史路径（每个 symbol 一行）
    x_query_symbol = train_pivot.select(symbol_cols).to_numpy().T   # (n_query_symbols, n_train_ts)
    x_query_ts = query_pivot.select(symbol_cols).to_numpy()         # (n_query_ts, n_query_symbols)

    n_symbols = x_query_symbol.shape[0]
    if n_symbols == 0 or len(x_query_ts) == 0:
        raise ValueError("empty train/query matrix in make_symbol_nn_feature_oos")

    # 只保留 base_coins 作为候选邻居库
    if base_coins is None:
        raise ValueError("base_coins must be provided for make_symbol_nn_feature_oos to define neighbor universe")
    else:
        base_coin_set = set(base_coins)
        base_symbol_cols = [s for s in symbol_cols if s in base_coin_set]
        if not base_symbol_cols:
            raise ValueError("none of base_coins are present in common train/query symbols")

    # base symbols 的历史路径（邻居候选库）
    x_base_symbol = train_pivot.select(base_symbol_cols).to_numpy().T   # (n_base_symbols, n_train_ts)

    k_fit = min(max(n + (1 if exclude_self else 0), 1), len(base_symbol_cols))

    nn = NearestNeighbors(
        n_neighbors=k_fit,
        metric=metric,
        p=p,
        metric_params=metric_params,
    )
    nn.fit(x_base_symbol)

    # 对“所有 symbol”做 query，但邻居只从 base_coins 中返回
    _, neighbor_idx = nn.kneighbors(x_query_symbol, return_distance=True)  # (n_symbols, k_fit)

    # query 时刻里，只取 base_coins 这些列来聚合
    base_col_idx = [symbol_cols.index(s) for s in base_symbol_cols]

    out = np.zeros((x_query_ts.shape[0], n_symbols), dtype=float)  # (n_query_ts, n_symbols)

    # 对每一个 symbol，用它的邻居 symbol 的当前值做一个聚合特征
    for j, symbol in enumerate(symbol_cols):
        nbrs_local = neighbor_idx[j]

        # 找到所有neighbor
        # 如果 exclude_self=True，并且当前 symbol 本身也在 base_coins 里，要去掉自己
        if exclude_self and symbol in base_symbol_cols:
            self_base_idx = base_symbol_cols.index(symbol)
            nbrs_local = nbrs_local[nbrs_local != self_base_idx]

        nbrs_local = nbrs_local[:n]
        if len(nbrs_local) == 0:
            # 极端情况：base_coins 只有自己一个，且 exclude_self=True（现在币多不太会发生）
            # 这里退化为用自己 (或者用np.nan）
            if symbol in base_symbol_cols:
                self_global_idx = symbol_cols.index(symbol)
                out[:, j] = x_query_ts[:, self_global_idx]
            else:
                out[:, j] = np.nan
            continue

        # 把base_coins的局部neighbor idx 转换成全局idx
        nbrs_global = [base_col_idx[i] for i in nbrs_local]
        # 聚合这些 neighbor 的当前值，得到 feature值
        out[:, j] = agg(x_query_ts[:, nbrs_global], axis=1)

    out_col = f"{feature_col}_symbol_nn{n}_{agg.__name__}"

    # 转化为长表输出
    return _to_long_feature_df(
        values=out,
        ts_values=query_pivot[ts_col].to_list(),
        symbols=symbol_cols,
        out_col=out_col,
        ts_col=ts_col,
        symbol_col=symbol_col,
    )

def make_time_nn_feature_oos(
    train_df: pl.DataFrame,
    query_df: pl.DataFrame,
    feature_col: str,
    n: int,
    *,
    metric: str = "minkowski",
    p: float = 2,
    metric_params: dict | None = None,
    agg: Callable = np.mean,
    ts_col: str = "ts",
    symbol_col: str = "symbol",
    exclude_self: bool = False,
    base_coins: Sequence[str] | None = None,
) -> pl.DataFrame:
    """
    用 train_df 的历史时刻做 fit，
    对 query_df 的时刻做 query，
    生成每个 (ts, symbol) 的 time-neighbor feature。

    如果提供 base_coins，则“时刻相似度”只基于这些币的特征列计算；
    但最终输出仍然对所有 symbol 生成。
    """
    train_pivot = _pivot_feature(train_df, feature_col, ts_col=ts_col, symbol_col=symbol_col)
    query_pivot = _pivot_feature(query_df, feature_col, ts_col=ts_col, symbol_col=symbol_col)

    # 防御性检查，确保 train 和 query 的 pivot 结果都包含同样的 symbol 列
    train_pivot, query_pivot, symbol_cols = _align_query_symbols(
        train_pivot, query_pivot, ts_col=ts_col
    )

    # 用于最终聚合输出的完整矩阵
    x_train_all = train_pivot.select(symbol_cols).to_numpy()   # (n_train_ts, n_symbols)
    x_query_all = query_pivot.select(symbol_cols).to_numpy()   # (n_query_ts, n_symbols)

    if len(x_train_all) == 0 or len(x_query_all) == 0:
        raise ValueError("empty train/query matrix in make_time_nn_feature_oos")

    # 只用 base_coins 来定义“时刻之间的相似度”
    if base_coins is None:
        fit_symbol_cols = symbol_cols
    else:
        base_coin_set = set(base_coins)
        fit_symbol_cols = [s for s in symbol_cols if s in base_coin_set]
        if not fit_symbol_cols:
            raise ValueError("none of base_coins are present in common train/query symbols")

    x_train_fit = train_pivot.select(fit_symbol_cols).to_numpy()   # (n_train_ts, n_fit_symbols)
    x_query_fit = query_pivot.select(fit_symbol_cols).to_numpy()   # (n_query_ts, n_fit_symbols)

    # neighbor 数量
    k_fit = min(max(n + (1 if exclude_self else 0), 1), len(x_train_fit))

    nn = NearestNeighbors(
        n_neighbors=k_fit,
        metric=metric,
        p=p,
        metric_params=metric_params,
    )
    nn.fit(x_train_fit)
    _, neighbor_idx = nn.kneighbors(x_query_fit, return_distance=True)

    # 找到邻居时刻后，仍然对所有 symbol 聚合
    agg_values = _agg_neighbors(
        source=x_train_all,
        neighbor_idx=neighbor_idx,
        n=n,
        agg=agg,
        exclude_self=False,  # query 和 train 不同集时通常不需要排自己
    )

    out_col = f"{feature_col}_time_nn{n}_{agg.__name__}"
    return _to_long_feature_df(
        values=agg_values,
        ts_values=query_pivot[ts_col].to_list(),
        symbols=symbol_cols,
        out_col=out_col,
        ts_col=ts_col,
        symbol_col=symbol_col,
    )

def _pivot_feature(
    df: pl.DataFrame,
    feature_col: str,
    ts_col: str = "ts",
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    if ts_col not in df.columns or symbol_col not in df.columns:
        raise ValueError(f"df must contain [{ts_col!r}, {symbol_col!r}]")
    if feature_col not in df.columns:
        raise ValueError(f"missing feature column: {feature_col}")

    pivot_df = (
        df.pivot(
            values=feature_col,
            index=ts_col,
            on=symbol_col,
            aggregate_function="first",
        )
        .sort(ts_col)
    )

    symbol_cols = _get_symbol_cols_from_pivot(pivot_df, ts_col=ts_col)

    # 暂时只选用均值填空
    pivot_df = pivot_df.with_columns(
        [pl.col(c).fill_null(pl.col(c).mean()) for c in symbol_cols]
    )
    return pivot_df


def _align_query_symbols(
    train_pivot: pl.DataFrame,
    query_pivot: pl.DataFrame,
    ts_col: str = "ts",
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """
    检查 train_pivot 和 query_pivot 的 symbol 列，保留两者共有的 symbol 列
    """
    train_symbols = [c for c in train_pivot.columns if c != ts_col]
    query_symbols = [c for c in query_pivot.columns if c != ts_col]

    common_symbols = [c for c in train_symbols if c in query_symbols]
    if not common_symbols:
        raise ValueError("no common symbols between train and query pivot")
    
    if len(common_symbols) < len(train_symbols) or len(common_symbols) < len(query_symbols):
        print(
            f"warning: train and query pivot have different symbol columns, "
            f"only keeping common symbols ({len(common_symbols)}), "
            f"train_symbols={train_symbols}, query_symbols={query_symbols}"
        )

    train_out = train_pivot.select([ts_col] + common_symbols)
    query_out = query_pivot.select([ts_col] + common_symbols)
    return train_out, query_out, common_symbols


def _to_long_feature_df(
    values: np.ndarray,
    ts_values,
    symbols: list[str],
    out_col: str,
    ts_col: str = "ts",
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    df = pl.DataFrame(
        {
            symbol_col: np.asarray(symbols).repeat(len(ts_values)).tolist(),
            ts_col: np.tile(np.asarray(ts_values), len(symbols)).tolist(),
            out_col: values.T.reshape(-1).tolist(),
        }
    )

    return df.with_columns(
        pl.col(ts_col).cast(pl.Datetime("ns"))
    )


def _agg_neighbors(
    source: np.ndarray,              # (n_source, dim)
    neighbor_idx: np.ndarray,        # (n_query, k)
    n: int,
    agg: Callable = np.mean,
    exclude_self: bool = False,
) -> np.ndarray:
    """
    对 source[neighbor_idx[:, :n]] 做聚合
    返回 shape: (n_query, dim)
    """
    k_total = neighbor_idx.shape[1]
    start = 1 if exclude_self else 0
    n = min(n, k_total)
    if n <= start:
        n = start + 1

    picked = source[neighbor_idx[:, start:n], :]   # (n_query, n_pick, dim)
    return agg(picked, axis=1)

def _get_symbol_cols_from_pivot(pivot_df: pl.DataFrame, ts_col: str = "ts") -> list[str]:
    cols = [c for c in pivot_df.columns if c != ts_col]
    if not cols:
        raise ValueError("pivot result has no symbol columns")
    return cols