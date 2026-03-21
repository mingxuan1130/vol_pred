from __future__ import annotations
from typing import List
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Sequence

import polars as pl

from data.vol_adapter import build_aligned_snapshot_vol

# ==========================================================================
# 单资产全天分块保存
# ==========================================================================
# schema:
# aligned_20ms/
#    2025-10-01/
#       BTCUSDT/data.parquet
#       ETHUSDT/data.parquet

def build_aligned_snapshot_vol_to_parquet(
    symbol: str,
    date: str,
    vol_data_dir: str,
    snapshot_root: str,
    snapshot_feats: List[str],
    vol_feats: List[str],
    vol_labels: List[str],
    out_root: str,
    *,
    subsample: str = "20ms",
    compression: str = "zstd",
    compression_level: int = 3,
    overwrite: bool = True,
) -> str:
    """
    build_aligned_snapshot_vol 是读取数据的完整pipeline
    这里把pipeline的结果直接存起来
    """
    # 确认输出路径
    out_path = _symbol_parquet_path(
        out_root=out_root,
        date=date,
        symbol=symbol,
        subsample=subsample,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if (not overwrite) and os.path.exists(out_path):
        return out_path

    # 通过pipeline构造因子数据
    df_symbol = build_aligned_snapshot_vol(
        symbol=symbol,
        date=date,
        vol_data_dir=vol_data_dir,
        snapshot_root=snapshot_root,
        snapshot_feats=snapshot_feats,
        vol_feats=vol_feats,
        vol_labels=vol_labels,
        subsample=subsample,
    ).with_columns(
        pl.lit(symbol).alias("symbol")
    )

    # 将结果写入 parquet
    df_symbol.write_parquet(
        out_path,
        compression=compression,
        compression_level=compression_level,
    )
    return out_path

def _normalize_freq_name(subsample: str) -> str:
    """
    把类似 '20ms' / '200ms' / '1s' 规范成目录名可直接使用的字符串
    """
    return subsample.strip().lower()


def _symbol_parquet_path(
    out_root: str,
    date: str,
    symbol: str,
    subsample: str = "20ms",
    filename: str = "data.parquet",
) -> str:
    freq_dir = f"aligned_{_normalize_freq_name(subsample)}"
    return os.path.join(out_root, freq_dir, date, symbol, filename)


# ==================
# 读取保存的 aligned parquet
# ==================
def load_aligned_symbol_parquet(
    out_root: str,
    date: str,
    symbol: str,
    *,
    subsample: str = "20ms",
    columns: Optional[List[str]] = None,
    lazy: bool = False,
):
    """
    读取单个 symbol 的 parquet。
    lazy=True 时返回 LazyFrame，否则返回 DataFrame。
    """
    path = _symbol_parquet_path(
        out_root=out_root,
        date=date,
        symbol=symbol,
        subsample=subsample,
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"aligned parquet not found: {path}")

    if lazy:
        lf = pl.scan_parquet(path)
        return lf.select(columns) if columns is not None else lf

    df = pl.read_parquet(path, columns=columns)
    return df


def load_aligned_multi_symbol_from_disk(
    symbols: Sequence[str],
    out_root: str,
    date: str,
    *,
    subsample: str = "20ms",
    columns: Optional[List[str]] = None,
    lazy: bool = False,
):
    """
    从磁盘加载多个 symbol。
    lazy=True 返回 concat 后的 LazyFrame。
    """
    if not symbols:
        raise ValueError("symbols cannot be empty")

    paths = [
        _symbol_parquet_path(out_root=out_root, date=date, symbol=s, subsample=subsample)
        for s in symbols
    ]

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing parquet files: {missing[:5]}")

    if lazy:
        lfs = [pl.scan_parquet(p) for p in paths]
        lf = pl.concat(lfs, how="vertical")
        return lf.select(columns) if columns is not None else lf

    dfs = [pl.read_parquet(p, columns=columns) for p in paths]
    return pl.concat(dfs, how="vertical")

# ==========================================================================
# 把数据按照block 保存
# ==========================================================================
def _block_cache_dir(
    out_root: str,
    date: str,
    block_freq: str = "5m",
    ) -> str:
    return os.path.join(out_root, f"block_{block_freq}", date)


def _block_file_name(ts) -> str:
    return ts.strftime("%Y-%m-%d_%H-%M-%S.parquet")


def load_block_cache(
    out_root: str,
    date: str,
    *,
    block_freq: str = "5m",
    block_starts: Optional[Sequence] = None,
    columns: Optional[Sequence[str]] = None,
    lazy: bool = False,
):
    """
    读取若干个 5m block parquet。
    """
    block_dir = _block_cache_dir(out_root=out_root, date=date, block_freq=block_freq)
    if not os.path.exists(block_dir):
        raise FileNotFoundError(f"block cache dir not found: {block_dir}")

    if block_starts is None:
        files = sorted(
            os.path.join(block_dir, f)
            for f in os.listdir(block_dir)
            if f.endswith(".parquet")
        )
    else:
        files = [os.path.join(block_dir, _block_file_name(ts)) for ts in block_starts]

    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing block cache files: {missing[:5]}")

    if lazy:
        lfs = [pl.scan_parquet(p) for p in files]
        lf = pl.concat(lfs, how="vertical")
        return lf.select(columns) if columns is not None else lf

    dfs = [pl.read_parquet(p, columns=columns) for p in files]
    return pl.concat(dfs, how="vertical") if dfs else pl.DataFrame()

# ==========================================================================
# 把数据按照cutoffs 保存
# ==========================================================================
def _block_cache_dir(out_root: str, date: str, block_freq: str = "5m") -> str:
    return os.path.join(out_root, f"block_{_normalize_freq_name(block_freq)}", date)


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


def _cutoff_dir_name(ts) -> str:
    return ts.strftime("%Y-%m-%d_%H-%M-%S")


def _block_file_name(ts) -> str:
    return ts.strftime("%Y-%m-%d_%H-%M-%S.parquet")


def _offset_ts(ts, by: str):
    return (
        pl.DataFrame({"ts": [ts]})
        .with_columns(pl.col("ts").dt.offset_by(by).alias("out"))
        .select("out")
        .item()
    )


def load_block_cache(
    out_root: str,
    date: str,
    *,
    block_freq: str = "5m",
    block_starts: Optional[Sequence] = None,
    columns: Optional[Sequence[str]] = None,
    lazy: bool = False,
):
    """
    读取某一天所有的 parquet files 并拼起来
    """
    block_dir = _block_cache_dir(out_root=out_root, date=date, block_freq=block_freq)
    if not os.path.exists(block_dir):
        raise FileNotFoundError(f"block cache dir not found: {block_dir}")

    if block_starts is None:
        files = sorted(
            os.path.join(block_dir, f)
            for f in os.listdir(block_dir)
            if f.endswith(".parquet")
        )
    else:
        files = [os.path.join(block_dir, _block_file_name(ts)) for ts in block_starts]

    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing block cache files: {missing[:5]}")

    if lazy:
        lfs = [pl.scan_parquet(p) for p in files]
        lf = pl.concat(lfs, how="vertical")
        return lf.select(columns) if columns is not None else lf

    dfs = [pl.read_parquet(p, columns=columns) for p in files]
    return pl.concat(dfs, how="vertical") if dfs else pl.DataFrame()
