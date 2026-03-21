"""
用来构造 block 和 train npz
加入 rolling train downsample:
- 训练集分别生成 freq_100ms / freq_200ms / freq_500ms
- 预测集仍保持原始 20ms 频率
- 对指定列做 last/mean/std，其他列做 last
"""

from __future__ import annotations

import glob
import json
import os
from collections import deque
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
import polars as pl

from data.config import *
from data.massive_data_adapter import load_aligned_symbol_parquet


# =========================================================
# Config
# =========================================================
@dataclass
class RollingNPZFastConfig:
    neighbor_dir: str
    aligned_root: str
    date: str

    # ready block 输出目录
    ready_block_dir: str

    # 最终 rolling npz 输出目录（这里建议传一个 base dir）
    # 例如: saved_data/lgbm_rolling_training_data/{date}
    # 代码内部会自动生成:
    #   out_dir/freq_100ms/train
    #   out_dir/freq_100ms/predict
    #   out_dir/freq_100ms/meta
    #   ...
    out_dir: str

    ts_col: str 
    symbol_col: str 
    target_col: str 

    train_blocks: int 
    aligned_subsample: str 

    compress_npz: bool

    ready_block_compression: str 

    base_feature_cols: tuple[str, ...] 

    # 训练集降频生成哪些频率
    train_downsample_freqs: tuple[str, ...] 

    # dataset 划窗模式:

    dataset_window_mode: str 
    train_days: int 
    validation_days: int 
    test_days: int 
    step_days: int 

    # 这些列在聚合时做 last / mean / std
    multi_agg_feature_cols: tuple[str, ...] 

# =========================================================
# Aggregation plan
# =========================================================
# 由于训练集和预测集的聚合逻辑完全一样（只是一个 group_by_dynamic，一个 rolling），
# 我们先定义一个统一的 agg_plan 来描述需要哪些聚合，输出什么列，哪些是 feature 列。
# 之后无论是训练集还是预测集，都用这个统一的 agg_plan 来生成对应的 Polars 表达式和 feature 列顺序。

@dataclass(frozen=True)
class AggItem:
    source_col: str
    op: str          # "last" | "mean" | "std"
    out_col: str
    is_feature: bool


def _build_agg_plan(
    *,
    raw_feature_cols: Sequence[str],
    cfg: RollingNPZFastConfig,
) -> list[AggItem]:
    """
    统一定义聚合计划：
    - target 永远 last，但不计入 feature_cols
    - multi agg 列: last / mean / std
    - 其他列: last
    """
    plan: list[AggItem] = []

    # target
    plan.append(
        AggItem(
            source_col=cfg.target_col,
            op="last",
            out_col=cfg.target_col,
            is_feature=False,
        )
    )

    # features
    for c in raw_feature_cols:
        if _need_multi_agg(c, cfg):
            plan.append(AggItem(c, "last", f"{c}_last", True))
            plan.append(AggItem(c, "mean", f"{c}_mean", True))
            plan.append(AggItem(c, "std",  f"{c}_std",  True))
        else:
            plan.append(AggItem(c, "last", f"{c}_last", True))

    return plan

def _agg_exprs_from_plan(plan: Sequence[AggItem]) -> list[pl.Expr]:
    """
    由统一 agg_plan 生成 Polars 聚合表达式
    """
    exprs: list[pl.Expr] = []

    for item in plan:
        if item.op == "last":
            exprs.append(
                pl.col(item.source_col).last().alias(item.out_col)
            )
        elif item.op == "mean":
            exprs.append(
                pl.col(item.source_col).mean().alias(item.out_col)
            )
        elif item.op == "std":
            exprs.append(
                pl.col(item.source_col)
                .std()
                .fill_null(0.0)
                .alias(item.out_col)
            )
        else:
            raise ValueError(f"Unsupported agg op: {item.op}")

    return exprs

def _feature_cols_from_plan(plan: Sequence[AggItem]) -> list[str]:
    """
    由统一 agg_plan 生成 feature 列顺序
    """
    return [item.out_col for item in plan if item.is_feature]

def _validate_agg_output(
    *,
    out_cols: Sequence[str],
    feature_cols: Sequence[str],
    cfg: RollingNPZFastConfig,
) -> None:
    required = {cfg.ts_col, cfg.symbol_col, cfg.target_col, *feature_cols}
    missing = [c for c in required if c not in out_cols]

    if missing:
        raise ValueError(
            f"Aggregated output missing required columns: {missing}"
        )
    
def _need_multi_agg(col: str, cfg: RollingNPZFastConfig) -> bool:
    """
    哪些列需要做 last / mean / std:
    1. 手动指定的基础列
    2. 所有以 _mean 结尾的 block 因子
    """
    return (col in cfg.multi_agg_feature_cols) or col.endswith("_mean")

# =========================================================
# 工具函数
# =========================================================
def list_neighbor_files(neighbor_dir: str) -> list[str]:
    """
    列出neighbor block的所有文件, 按文件名排序。每个文件对应一个 block。
    """
    files = sorted(glob.glob(os.path.join(neighbor_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {neighbor_dir}")
    return files


def _make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if is_dataclass(obj):
        return _make_json_serializable(asdict(obj))
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_make_json_serializable(data), f, ensure_ascii=False, indent=2)


def _np_savez(path: str, compress: bool, **kwargs) -> None:
    """
    将文件转成 npz 格式
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if compress:
        np.savez_compressed(path, **kwargs)
    else:
        np.savez(path, **kwargs)


def _safe_std_expr(col: str) -> pl.Expr:
    """
    std 在只有一个样本的窗口里可能是 null，这里统一补成 0
    """
    return pl.col(col).std().fill_null(0.0).alias(f"{col}_std")


def make_xy(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    X = df.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    y = (
        df.get_column(target_col)
        .cast(pl.Float64)
        .to_numpy()
        .astype(np.float32, copy=False)
    )
    return X, y


# =========================================================
# Step 1: Build ready blocks
# construct : neighbor block + base features + target
# =========================================================
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
    拿之前读取好的所有 symbols 和基础因子数据和新构造的 neighbor block 做 join
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


# =========================================================
# Step 2: Rolling cache build NPZ
# =========================================================
def list_ready_block_files(ready_block_dir: str) -> list[str]:
    """
    列出 ready block 的所有文件, 按文件名排序。每个文件对应一个 block。
    """
    files = sorted(glob.glob(os.path.join(ready_block_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No ready parquet files found in: {ready_block_dir}")
    return files


def add_symbol_code_single_df(
    df: pl.DataFrame,
    symbol_col: str,
    mapping: dict[str, int],
) -> pl.DataFrame:
    """
    给单个 df 加 symbol_code 列
    """
    return df.with_columns(
        pl.col(symbol_col).replace_strict(mapping).cast(pl.Int32).alias("symbol_code")
    )


def get_global_symbol_mapping_from_ready_blocks(
    ready_files: Sequence[str],
    symbol_col: str,
    scan_limit: int | None = None,
) -> dict[str, int]:
    """
    给所有 symbol 编号，返回一个全局的 symbol -> code 映射
    """
    files = ready_files if scan_limit is None else ready_files[:scan_limit]
    syms: list[str] = []

    for p in files:
        s = (
            pl.read_parquet(p, columns=[symbol_col])
            .get_column(symbol_col)
            .unique()
            .to_list()
        )
        syms.extend(s)

    uniq = sorted(set(syms))
    return {sym: i for i, sym in enumerate(uniq)}


def _prepare_raw_block_df(
    df: pl.DataFrame,
    cfg: RollingNPZFastConfig,
) -> pl.DataFrame:
    """
    保留 rolling / 聚合需要的列，并 drop_nulls
    """
    exclude = {cfg.ts_col, cfg.symbol_col, cfg.target_col}
    raw_feature_cols = [c for c in df.columns if c not in exclude]

    keep_cols = [cfg.ts_col, cfg.symbol_col, cfg.target_col, *raw_feature_cols]
    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in df.columns]))

    return (
        df.select(keep_cols)
        .sort([cfg.symbol_col, cfg.ts_col])
        .drop_nulls()
    )

def _downsample_train_df(
    train_df: pl.DataFrame,
    *,
    freq: str,
    cfg: RollingNPZFastConfig,
) -> tuple[pl.DataFrame, list[str]]:
    """
    训练集：
    按 symbol + time bucket 做 group_by_dynamic 下采样
    """
    exclude = {cfg.ts_col, cfg.symbol_col, cfg.target_col}
    raw_feature_cols = [c for c in train_df.columns if c not in exclude]

    plan = _build_agg_plan(
        raw_feature_cols=raw_feature_cols,
        cfg=cfg,
    )
    agg_exprs = _agg_exprs_from_plan(plan)

    out = (
        train_df
        .sort([cfg.symbol_col, cfg.ts_col])
        .group_by_dynamic(
            index_column=cfg.ts_col,
            every=freq,
            period=freq,
            group_by=cfg.symbol_col,
            label="right",
            closed="right",
        )
        .agg(agg_exprs)
        .sort([cfg.symbol_col, cfg.ts_col])
    )

    feature_cols = _feature_cols_from_plan(plan)

    _validate_agg_output(
        out_cols=out.columns,
        feature_cols=feature_cols,
        cfg=cfg,
    )

    keep_cols = [cfg.ts_col, cfg.symbol_col, cfg.target_col, *feature_cols]
    out = out.select(keep_cols).drop_nulls()

    return out, feature_cols


def _build_pred_rolling_features(
    pred_df: pl.DataFrame,
    *,
    freq: str,
    cfg: RollingNPZFastConfig,
) -> tuple[pl.DataFrame, list[str]]:
    """
    预测集：
    保持原始 20ms 行数，但每一行用过去一个 freq 窗口做 rolling 聚合，
    让列结构与 downsample 后的训练集一致。
    """
    exclude = {cfg.ts_col, cfg.symbol_col, cfg.target_col}
    raw_feature_cols = [c for c in pred_df.columns if c not in exclude]

    plan = _build_agg_plan(
        raw_feature_cols=raw_feature_cols,
        cfg=cfg,
    )
    agg_exprs = _agg_exprs_from_plan(plan)

    out = (
        pred_df
        .sort([cfg.symbol_col, cfg.ts_col])
        .rolling(
            index_column=cfg.ts_col,
            period=freq,
            group_by=cfg.symbol_col,
            closed="right",
        )
        .agg(agg_exprs)
        .sort([cfg.symbol_col, cfg.ts_col])
    )

    feature_cols = _feature_cols_from_plan(plan)

    _validate_agg_output(
        out_cols=out.columns,
        feature_cols=feature_cols,
        cfg=cfg,
    )

    keep_cols = [cfg.ts_col, cfg.symbol_col, cfg.target_col, *feature_cols]
    out = out.select(keep_cols).drop_nulls()

    return out, feature_cols


def save_block_npz(
    *,
    save_path: str,
    compress: bool,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    y_pred_true: np.ndarray,
    ts_pred: np.ndarray,
    symbol_pred: np.ndarray,
    feature_cols: list[str],
    train_files: list[str],
    pred_file: str,
    extra_meta: dict | None = None,
) -> None:
    payload = dict(
        X_train=X_train,
        y_train=y_train,
        X_pred=X_pred,
        y_pred_true=y_pred_true,
        ts_pred=ts_pred,
        symbol_pred=symbol_pred,
        feature_cols=np.array(feature_cols, dtype=object),
        train_files=np.array(train_files, dtype=object),
        pred_file=np.array([pred_file], dtype=object),
    )

    if extra_meta is not None:
        payload["extra_meta_json"] = np.array(
            [json.dumps(extra_meta, ensure_ascii=False)],
            dtype=object,
        )

    _np_savez(save_path, compress=compress, **payload)


def _freq_output_dirs(base_out_dir: str, freq: str) -> tuple[str, str, str]:
    """
    每个频率单独落一套目录
    """
    freq_dir = os.path.join(base_out_dir, f"freq_{freq}")
    train_dir = os.path.join(freq_dir, "train")
    pred_dir = os.path.join(freq_dir, "predict")
    meta_dir = os.path.join(freq_dir, "meta")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    return train_dir, pred_dir, meta_dir


def build_rolling_npz_data_with_cache(cfg: RollingNPZFastConfig) -> dict:
    """
    ready block -> rolling cache -> npz

    现在会为每个 freq 生成一套:
    - train: 训练集按 freq 降频后的 npz
    - predict: 轻量引用
    - meta: 每个 block 的 json
    """
    os.makedirs(cfg.out_dir, exist_ok=True)

    ready_files = list_ready_block_files(cfg.ready_block_dir)

    if len(ready_files) <= cfg.train_blocks:
        raise ValueError(
            f"Not enough ready blocks: len={len(ready_files)}, train_blocks={cfg.train_blocks}"
        )

    symbol_mapping = get_global_symbol_mapping_from_ready_blocks(
        ready_files=ready_files,
        symbol_col=cfg.symbol_col,
    )

    # 每个频率保存一份运行配置
    for freq in cfg.train_downsample_freqs:
        train_dir, pred_dir, meta_dir = _freq_output_dirs(cfg.out_dir, freq)

        save_json(
            {
                "saved_at": datetime.now().isoformat(),
                "date": cfg.date,
                "neighbor_dir": cfg.neighbor_dir,
                "ready_block_dir": cfg.ready_block_dir,
                "out_dir": os.path.join(cfg.out_dir, f"freq_{freq}"),
                "train_blocks": cfg.train_blocks,
                "aligned_subsample": cfg.aligned_subsample,
                "target_col": cfg.target_col,
                "ts_col": cfg.ts_col,
                "symbol_col": cfg.symbol_col,
                "base_feature_cols": list(cfg.base_feature_cols),
                "num_total_ready_blocks": len(ready_files),
                "compress_npz": cfg.compress_npz,
                "global_symbol_mapping": symbol_mapping,
                "train_downsample_freq": freq,
                "multi_agg_feature_cols": list(cfg.multi_agg_feature_cols),
            },
            os.path.join(meta_dir, "run_config.json"),
        )

    # -------------------------------------------------------
    # rolling window
    # -------------------------------------------------------
    window_blocks: deque[pl.DataFrame] = deque()
    window_file_names: deque[str] = deque()

    for p in ready_files[:cfg.train_blocks]:
        df = pl.read_parquet(p)
        df = add_symbol_code_single_df(df, cfg.symbol_col, symbol_mapping)
        df = _prepare_raw_block_df(df, cfg)
        window_blocks.append(df)
        window_file_names.append(os.path.basename(p))

    block_count_by_freq = {freq: 0 for freq in cfg.train_downsample_freqs}
    skipped_by_freq = {freq: 0 for freq in cfg.train_downsample_freqs}

    for i in range(cfg.train_blocks, len(ready_files)):
        pred_path = ready_files[i]
        block_name = os.path.splitext(os.path.basename(pred_path))[0]
        print(f"[ROLLING][BUILD] {block_name}")

        pred_df_raw = pl.read_parquet(pred_path)
        pred_df_raw = add_symbol_code_single_df(pred_df_raw, cfg.symbol_col, symbol_mapping)
        pred_df_raw = _prepare_raw_block_df(pred_df_raw, cfg)

        train_df_raw = pl.concat(list(window_blocks), how="vertical")

        if train_df_raw.is_empty() or pred_df_raw.is_empty():
            print(f"[ROLLING][WARN] skip {block_name}: raw train/pred empty")
            for freq in cfg.train_downsample_freqs:
                skipped_by_freq[freq] += 1

            window_blocks.popleft()
            window_file_names.popleft()
            window_blocks.append(pred_df_raw)
            window_file_names.append(os.path.basename(pred_path))
            continue

        for freq in cfg.train_downsample_freqs:
            train_dir, pred_dir, meta_dir = _freq_output_dirs(cfg.out_dir, freq)

            train_df_freq, feature_cols_train = _downsample_train_df(
                train_df_raw,
                freq=freq,
                cfg=cfg,
            )
            pred_df_freq, feature_cols_pred = _build_pred_rolling_features(
                pred_df_raw,
                freq=freq,
                cfg=cfg,
            )

            if feature_cols_train != feature_cols_pred:
                raise RuntimeError(
                    f"feature cols mismatch at freq={freq}, block={block_name}"
                )

            feature_cols = feature_cols_train

            if train_df_freq.is_empty() or pred_df_freq.is_empty():
                print(f"[ROLLING][WARN] skip {block_name} @ {freq}: empty after agg")
                skipped_by_freq[freq] += 1
                continue

            X_train, y_train = make_xy(train_df_freq, feature_cols, cfg.target_col)
            X_pred, y_pred_true = make_xy(pred_df_freq, feature_cols, cfg.target_col)

            ts_pred = pred_df_freq.get_column(cfg.ts_col).to_numpy()
            symbol_pred = pred_df_freq.get_column(cfg.symbol_col).to_numpy()

            train_npz_path = os.path.join(train_dir, f"{block_name}.npz")

            save_block_npz(
                save_path=train_npz_path,
                compress=cfg.compress_npz,
                X_train=X_train,
                y_train=y_train,
                X_pred=X_pred,
                y_pred_true=y_pred_true,
                ts_pred=ts_pred,
                symbol_pred=symbol_pred,
                feature_cols=feature_cols,
                train_files=list(window_file_names),
                pred_file=os.path.basename(pred_path),
                extra_meta={
                    "freq": freq,
                    "symbol_mapping": symbol_mapping,
                    "multi_agg_feature_cols": list(cfg.multi_agg_feature_cols),
                },
            )

            _np_savez(
                os.path.join(pred_dir, f"{block_name}.npz"),
                compress=False,
                block_name=np.array([block_name], dtype=object),
                train_npz_path=np.array([train_npz_path], dtype=object),
                freq=np.array([freq], dtype=object),
            )

            save_json(
                {
                    "block_name": block_name,
                    "freq": freq,
                    "train_files": list(window_file_names),
                    "pred_file": os.path.basename(pred_path),
                    "n_train": int(len(y_train)),
                    "n_pred": int(len(y_pred_true)),
                    "num_features": len(feature_cols),
                    "feature_cols": feature_cols,
                    "multi_agg_feature_cols": list(cfg.multi_agg_feature_cols),
                },
                os.path.join(meta_dir, f"{block_name}.json"),
            )

            block_count_by_freq[freq] += 1

        # ---------- rolling forward ----------
        window_blocks.popleft()
        window_file_names.popleft()
        window_blocks.append(pred_df_raw)
        window_file_names.append(os.path.basename(pred_path))

    if all(v == 0 for v in block_count_by_freq.values()):
        raise RuntimeError("No npz training data generated for any frequency.")

    return {
        "num_blocks_by_freq": block_count_by_freq,
        "num_skipped_by_freq": skipped_by_freq,
        "out_dir": cfg.out_dir,
        "freqs": list(cfg.train_downsample_freqs),
    }
