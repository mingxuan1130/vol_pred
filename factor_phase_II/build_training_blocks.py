from __future__ import annotations

"""
输入 neighbor block, 输出 enriched ready block

不关心 rolling, 不关心 npz。
"""


import glob
import os
from typing import Sequence
from typing import TYPE_CHECKING
from datetime import datetime

import polars as pl

from data.massive_data_adapter import load_aligned_symbol_parquet

if TYPE_CHECKING:
    from factor_phase_II.build_rolling_npz import RollingNPZFastConfig


def list_neighbor_files(neighbor_dir: str) -> list[str]:
    """
    列出neighbor block的所有文件, 按文件名排序。每个文件对应一个 block。
    """
    files = sorted(glob.glob(os.path.join(neighbor_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {neighbor_dir}")
    return files

    
def _load_aligned_slice_for_symbols(
    *,
    aligned_root: str,
    date: str,
    symbols: Sequence[str],
    ts_min,
    ts_max,
    subsample: str,
    ts_col: str,
    symbol_col: str,
    base_feature_cols: Sequence[str],
) -> pl.DataFrame:
    """
    一次性把指定 symbols 的基础特征从 aligned 数据里读出来，时间范围为 [ts_min, ts_max]
    这一步只在 ready block 阶段做一次
    """
    parts: list[pl.DataFrame] = []
    need_cols = [ts_col, *base_feature_cols]

    for sym in symbols:
        df_sym = load_aligned_symbol_parquet(
            out_root=aligned_root,
            date=date,
            symbol=sym,
            subsample=subsample,
            columns=need_cols,
            lazy=False,
        )

        df_sym = (
            df_sym
            .filter((pl.col(ts_col) >= ts_min) & (pl.col(ts_col) <= ts_max))
            .with_columns(pl.lit(sym).alias(symbol_col))
            .select([ts_col, symbol_col, *base_feature_cols])
        )

        if not df_sym.is_empty():
            parts.append(df_sym)

    if not parts:
        return pl.DataFrame(
            schema=[
                (ts_col, pl.Datetime("ns")),
                (symbol_col, pl.String),
                *[(c, pl.Float64) for c in base_feature_cols],
            ]
        )

    return pl.concat(parts, how="vertical")


def enrich_neighbor_block_with_base_features(
    neighbor_df: pl.DataFrame,
    cfg: RollingNPZFastConfig,
) -> pl.DataFrame:
    """
    拿之前读取好的 所有symbols 和基础因子数据和新构造的 neighbor block 做 join
    """
    if neighbor_df.is_empty():
        return neighbor_df

    ts_min = neighbor_df.select(pl.col(cfg.ts_col).min()).item()
    ts_max = neighbor_df.select(pl.col(cfg.ts_col).max()).item()
    symbols = neighbor_df.get_column(cfg.symbol_col).unique().sort().to_list()

    base_df = _load_aligned_slice_for_symbols(
        aligned_root=cfg.aligned_root,
        date=cfg.date,
        symbols=symbols,
        ts_min=ts_min,
        ts_max=ts_max,
        subsample=cfg.aligned_subsample,
        ts_col=cfg.ts_col,
        symbol_col=cfg.symbol_col,
        base_feature_cols=cfg.base_feature_cols,
    )

    out = neighbor_df.join(
        base_df,
        on=[cfg.ts_col, cfg.symbol_col],
        how="left",
        coalesce=True,
    )

    return out

def build_ready_blocks(cfg: RollingNPZFastConfig) -> dict:
    """
    先把 neighbor block -> enriched ready block。
    这样之后 rolling 阶段就不再去读 aligned parquet 
    """
    os.makedirs(cfg.ready_block_dir, exist_ok=True)

    neighbor_files = list_neighbor_files(cfg.neighbor_dir)

    meta_dir = os.path.join(cfg.ready_block_dir, "_meta")
    os.makedirs(meta_dir, exist_ok=True)

    save_json(
        {
            "saved_at": datetime.now().isoformat(),
            "date": cfg.date,
            "neighbor_dir": cfg.neighbor_dir,
            "aligned_root": cfg.aligned_root,
            "ready_block_dir": cfg.ready_block_dir,
            "aligned_subsample": cfg.aligned_subsample,
            "ts_col": cfg.ts_col,
            "symbol_col": cfg.symbol_col,
            "target_col": cfg.target_col,
            "base_feature_cols": list(cfg.base_feature_cols),
            "num_neighbor_blocks": len(neighbor_files),
        },
        os.path.join(meta_dir, "ready_block_config.json"),
    )

    built = 0
    skipped = 0

    for path in neighbor_files:
        block_name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(cfg.ready_block_dir, f"{block_name}.parquet")

        if os.path.exists(out_path):
            print(f"[READY][SKIP] {block_name}")
            skipped += 1
            continue

        print(f"[READY][BUILD] {block_name}")

        neighbor_df = pl.read_parquet(path)
        ready_df = enrich_neighbor_block_with_base_features(neighbor_df, cfg)

        # 这里先不 drop_nulls，保留原始 ready block。
        # 后面 rolling 阶段根据 feature_cols 统一清洗。
        ready_df.write_parquet(
            out_path,
            compression=cfg.ready_block_compression,
        )
        built += 1

    return {
        "num_neighbor_blocks": len(neighbor_files),
        "built": built,
        "skipped": skipped,
        "ready_block_dir": cfg.ready_block_dir,
    }
