from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
import os
from typing import Dict, List, Literal, Optional, Sequence

import polars as pl

from data.config import *
from data.raw_loader import (
    load_and_align_volatility,
    load_snapshot_raw,
)
from factor_phase_I.snapshot import add_factors, construct_factors_from_snapshot
from data.utils import trim_first_15min, filter_time_range
from factor_phase_I.snapshot import add_factors


def _ensure_datetime_ns(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Best-effort cast to Datetime(ns) for timestamp comparisons."""
    if col not in df.columns:
        raise ValueError(f"missing time column: {col}")

    dtype = df.schema[col]
    if isinstance(dtype, pl.Datetime):
        return df.with_columns(pl.col(col).cast(pl.Datetime("ns")))
    if dtype == pl.Date:
        return df.with_columns(pl.col(col).cast(pl.Datetime("ns")))
    if dtype in pl.INTEGER_DTYPES:
        return df.with_columns(pl.from_epoch(pl.col(col).cast(pl.Int64), time_unit="ns").alias(col))
    if dtype in pl.FLOAT_DTYPES:
        return df.with_columns(
            pl.from_epoch(pl.col(col).cast(pl.Int64), time_unit="ns").alias(col)
        )

    return df.with_columns(
        pl.col(col).cast(pl.Utf8).str.to_datetime(time_unit="ns", strict=False).alias(col)
    )


def add_vol_as_features(
    origin_df: pl.DataFrame,
    df_vol: pl.DataFrame,
    vol_feats: List[str] = VOL_FEATURES,
    *,
    ts_col_snapshot: str = "ts",
    ts_col_vol: str = "ts_grid",
    suffix_template: str = "{col}_lag{lag}",
    merge_how: Literal["left", "right", "full", "inner", "cross"] = "left",
) -> pl.DataFrame:
    """
    Merge volatility columns into snapshot/factor table as lag-0 features.
    """
    if ts_col_snapshot not in origin_df.columns:
        raise ValueError(f"snapshot missing timestamp column: {ts_col_snapshot}")
    if ts_col_vol not in df_vol.columns:
        raise ValueError(f"vol table missing timestamp column: {ts_col_vol}")
    if not vol_feats:
        raise ValueError("vol_feats cannot be empty")

    left_df = _ensure_datetime_ns(origin_df, ts_col_snapshot)
    right_df = _ensure_datetime_ns(df_vol, ts_col_vol)

    select_exprs = [pl.col(ts_col_vol), pl.col(ts_col_vol).alias("__ts_check__")]
    for vol_col in vol_feats:
        if vol_col not in right_df.columns:
            raise ValueError(f"vol table missing feature column: {vol_col}")
        out_col = suffix_template.format(col=vol_col, lag=0)
        select_exprs.append(pl.col(vol_col).alias(out_col))

    time_map = right_df.select(select_exprs)

    merged = left_df.join(
        time_map,
        left_on=ts_col_snapshot,
        right_on=ts_col_vol,
        how=merge_how,
    )

    df_final = _validate_lag_and_drop_cols(
        merged,
        ts_left_col=ts_col_snapshot,
        ts_right_col="__ts_check__",
        expected_delta=timedelta(0),
        drop_cols=["__ts_check__"],
    )

    return df_final 


def add_vol_as_labels(
    origin_df: pl.DataFrame,
    df_vol: pl.DataFrame,
    vol_labels: List[str],
    *,
    ts_col_snapshot: str = "ts",
    ts_col_vol: str = "ts_grid",
    grid_resolution: Optional[str] = None,
    merge_how: Literal["left", "right", "full", "inner", "cross"] = "inner",
    vol_dict: Dict[str, int] = VOL_LABEL,
) -> pl.DataFrame:
    """
    Merge forward-shifted volatility columns as labels.
    """
    if ts_col_snapshot not in origin_df.columns:
        raise ValueError(f"snapshot missing timestamp column: {ts_col_snapshot}")
    if ts_col_vol not in df_vol.columns:
        raise ValueError(f"vol table missing timestamp column: {ts_col_vol}")
    if not vol_labels or any(label not in vol_dict for label in vol_labels):
        raise ValueError("vol_labels must be a non-empty subset of vol_dict")

    left_df = _ensure_datetime_ns(origin_df, ts_col_snapshot).sort(ts_col_snapshot)
    right_df = _ensure_datetime_ns(df_vol, ts_col_vol).sort(ts_col_vol)

    shifts = {label: int(vol_dict[label]) for label in vol_labels}
    if any(step <= 0 for step in shifts.values()):
        raise ValueError(f"all label shifts must be positive: {shifts}")

    # 这里我们假设vol的ts_grid是等间隔的，且间隔可以从shifts里推断出来
    # 我们会检查这个假设，如果不满足就raise，避免后续对齐出问题
    diffs = (
        right_df
        .select(pl.col(ts_col_vol).diff().drop_nulls().unique().sort().alias("diffs"))
        .get_column("diffs")
        .to_list()
    )

    if not diffs:
        raise ValueError("cannot infer base step from vol ts_grid")

    if len(diffs) != 1:
        raise ValueError(
            f"vol ts_grid is not evenly spaced; expected exactly one unique diff, got {diffs}"
        )

    base_step = diffs[0]

    if grid_resolution is not None:
        grid_value, grid_unit = _parse_duration(grid_resolution)
        expected_base_step = pl.select(
            pl.duration(**{grid_unit: grid_value}).alias("step")
        ).item()

        if expected_base_step is None:
            raise ValueError(
                f"cannot parse expected base step from grid_resolution={grid_resolution!r}"
            )

        if base_step != expected_base_step:
            raise ValueError(
                f"vol ts_grid step mismatch: inferred {base_step}, "
                f"expected {expected_base_step} from grid_resolution={grid_resolution!r}"
            )

    # Timestamp validation uses the first label's step, matching pandas implementation.
    first_label = vol_labels[0]
    first_step = shifts[first_label]

    select_exprs: list[pl.Expr] = [pl.col(ts_col_vol)]
    select_exprs.append(pl.col(ts_col_vol).shift(-first_step).alias("__ts_check__"))

    for label_name in vol_labels:
        if label_name not in right_df.columns:
            raise ValueError(f"vol table missing label source column: {label_name}")
        step = shifts[label_name]
        out_col = f"y_{label_name}"
        select_exprs.append(pl.col(label_name).shift(-step).alias(out_col))

    time_map = right_df.select(select_exprs)

    merged = left_df.join(
        time_map,
        left_on=ts_col_snapshot,
        right_on=ts_col_vol,
        how=merge_how,
    )

    df_final = _validate_lag_and_drop_cols(
        merged,
        ts_left_col=ts_col_snapshot,
        ts_right_col="__ts_check__",
        expected_delta=base_step * first_step,
        drop_cols=["__ts_check__"],
    )

    return df_final




def _validate_lag_and_drop_cols(
    df: pl.DataFrame,
    *,
    ts_left_col: str,
    ts_right_col: str,
    expected_delta: object,
    drop_cols: List[str],
    allow_empty_check: bool = False,
    sample_n: int = 5,
) -> pl.DataFrame:
    """
    在merged 完之后检查对齐，并删掉对齐检查用的列
    """
    if ts_left_col not in df.columns:
        raise ValueError(f"missing left timestamp column: {ts_left_col}")
    if ts_right_col not in df.columns:
        raise ValueError(f"missing right timestamp column: {ts_right_col}")

    out = _ensure_datetime_ns(df, ts_left_col)
    out = _ensure_datetime_ns(out, ts_right_col)

    checked = out.filter(pl.col(ts_left_col).is_not_null() & pl.col(ts_right_col).is_not_null())
    if checked.is_empty():
        if allow_empty_check:
            return out.drop([c for c in drop_cols if c in out.columns])
        raise ValueError("no rows available for timestamp alignment check")

    bad = checked.filter((pl.col(ts_right_col) - pl.col(ts_left_col)) != expected_delta)
    if bad.height > 0:
        sample = bad.select([ts_left_col, ts_right_col]).head(sample_n)
        raise ValueError(
            f"timestamp alignment check failed (expected={expected_delta}); sample:\n{sample}"
        )

    return out.drop([c for c in drop_cols if c in out.columns])


def add_lag_features(
    origin_df: pl.DataFrame,
    source_col: str = "vol_5m_lag0",
    lags: Sequence[str] = ("5min", "10min"),
    *,
    ts_col: str = "ts",
    out_col_template: str = "{prefix}_lag{lag}",
    validate_alignment: bool = True,
) -> pl.DataFrame:
    """
    Build lag features from `source_col`, where lag is specified by durations like `5min`.
    """
    if ts_col not in origin_df.columns:
        raise ValueError(f"missing timestamp column: {ts_col}")
    if source_col not in origin_df.columns:
        raise ValueError(f"missing source column: {source_col}")
    if not lags:
        raise ValueError("lags cannot be empty")

    out = _ensure_datetime_ns(origin_df, ts_col).sort(ts_col)

    base_step = out.select(pl.col(ts_col).diff().drop_nulls().median().alias("base_step")).item()
    if base_step is None:
        raise ValueError("cannot infer valid base step from ts")

    prefix = source_col[:-5] if source_col.endswith("_lag0") else source_col

    for lag in lags:
        lag_value, lag_unit = _parse_duration(lag)
        lag_ns_int = _duration_to_ns(lag_value, lag_unit)

        base_step_ns = int(base_step.total_seconds() * 1_000_000_000)
        steps = int(round(lag_ns_int / base_step_ns)) if base_step_ns > 0 else -1

        if steps <= 0 or abs(steps * base_step_ns - lag_ns_int) > 1:
            raise ValueError(f"lag={lag} is not aligned with step={base_step}")

        out_col = out_col_template.format(prefix=prefix, lag=lag)
        out = out.with_columns(pl.col(source_col).shift(steps).alias(out_col))

        if validate_alignment:
            src_ts_col = f"__src_ts_for_{out_col}__"
            out = out.with_columns(pl.col(ts_col).shift(steps).alias(src_ts_col))
            out = _validate_lag_and_drop_cols(
                out,
                ts_left_col=src_ts_col,
                ts_right_col=ts_col,
                expected_delta=base_step * steps,
                drop_cols=[src_ts_col],
                allow_empty_check=True,
            )

    return out


def _parse_duration(duration: str) -> tuple[int, str]:
    txt = duration.strip().lower()
    mapping = {
        "ns": "nanoseconds",
        "us": "microseconds",
        "ms": "milliseconds",
        "s": "seconds",
        "sec": "seconds",
        "m": "minutes",
        "min": "minutes",
        "h": "hours",
        "hr": "hours",
    }

    num = ""
    unit = ""
    for ch in txt:
        if ch.isdigit():
            num += ch
        else:
            unit += ch

    if not num or unit not in mapping:
        raise ValueError(f"unsupported duration format: {duration!r}")

    return int(num), mapping[unit]


def _duration_to_ns(value: int, unit: str) -> int:
    unit_to_ns = {
        "nanoseconds": 1,
        "microseconds": 1_000,
        "milliseconds": 1_000_000,
        "seconds": 1_000_000_000,
        "minutes": 60 * 1_000_000_000,
        "hours": 3600 * 1_000_000_000,
    }
    if unit not in unit_to_ns:
        raise ValueError(f"unsupported duration unit: {unit}")
    return value * unit_to_ns[unit]

# ==============================================================================
# 下面是小规模的multi symbol构建，主要用来做功能验证和调试。
# 它直接在内存里把snapshot和vol表对齐拼接成一个大表，适合少量symbol或者少量数据的情况。
# ==============================================================================

def build_aligned_snapshot_vol(
    symbol: str,
    date: str,
    vol_data_dir: str,
    snapshot_root: str,
    snapshot_feats: Optional[List[str]] = None,
    vol_feats: Optional[List[str]] = None,
    vol_labels: Optional[List[str]] = None,
    *,
    drop_leading_15min: bool = True,
    subsample: str = "20ms",
    snapshot_feature_list: List[str] = SNAPSHOT_FEATURES,
    vol_feature_list: List[str] = VOL_FEATURES,
) -> pl.DataFrame:
    """
    构造数据的完整pipeline:
        1. 加载原始snapshot和vol数据
        2. 对齐后把vol的特征作为snapshot的特征
        3. vol的标签作为snapshot的标签
        4. 截取开始15分钟  
    """
    if snapshot_feats is None:
        snapshot_feats = snapshot_feature_list
    if vol_feats is None:
        vol_feats = vol_feature_list

    bad_snapshot = [c for c in snapshot_feats if c not in snapshot_feature_list]
    if bad_snapshot:
        raise ValueError(f"snapshot_feats has invalid columns: {bad_snapshot}")

    bad_vol_feats = [c for c in vol_feats if c not in vol_feature_list]
    if bad_vol_feats:
        raise ValueError(f"vol_feats has invalid columns: {bad_vol_feats}")

    if not vol_labels:
        raise ValueError("vol_labels cannot be empty")

    snapshot_df = load_snapshot_raw(
        snapshot_root=snapshot_root,
        symbol=symbol,
        date=date,
        x_cols=snapshot_feats,
        subsample=subsample,
    )
    
    df_vol = load_and_align_volatility(
        symbol=symbol,
        date=date,
        vol_data_dir=vol_data_dir,
    )

    bad_labels = [c for c in vol_labels if c not in df_vol.columns]
    if bad_labels:
        raise ValueError(f"vol_labels not found in vol table: {bad_labels}")

    time_start = df_vol.select(pl.col("ts_grid").min()).item()
    time_end = df_vol.select(pl.col("ts_grid").max()).item()
    snap_df = _ensure_datetime_ns(snapshot_df, "ts").filter(
        (pl.col("ts") >= pl.lit(time_start)) & (pl.col("ts") <= pl.lit(time_end))
    )

    # Factor constructor is currently pandas-based, so bridge through pandas here.
    feat_df = construct_factors_from_snapshot(snap_df)

    if "ts" not in feat_df.columns:
        if feat_df.height == snap_df.height:
            feat_df = feat_df.with_columns(snap_df["ts"])
        else:
            raise ValueError("factor constructor result has no ts and cannot be recovered")

    df_with_vol_feats = add_vol_as_features(
        origin_df=feat_df,
        df_vol=df_vol,
        vol_feats=vol_feats,
    )

    df_features = add_factors(df_with_vol_feats)

    df_final = add_vol_as_labels(
        origin_df=df_features,
        df_vol=df_vol,
        vol_labels=vol_labels,
        grid_resolution=subsample,
    )

    if drop_leading_15min:
        df_final = trim_first_15min(
            df_final,
            time_col="ts"
        )

    return df_final


def build_aligned_multi_symbol(
    symbols: Sequence[str],
    date: str,
    vol_data_dir: str,
    snapshot_root: str,
    snapshot_feats: List[str],
    vol_feats: List[str],
    vol_labels: List[str],
    *,
    subsample: str = "20ms",
    on_error: str = "skip",
) -> pl.DataFrame:
    """
    Build per-symbol aligned tables in parallel and concatenate them.
    """
    if on_error not in {"skip", "raise"}:
        raise ValueError("on_error must be either 'skip' or 'raise'")
    if not symbols:
        raise ValueError("symbols cannot be empty")

    failed_symbols: list[str] = []
    symbol_to_df: dict[str, pl.DataFrame] = {}

    def _build_one(symbol: str) -> pl.DataFrame:
        df_symbol = build_aligned_snapshot_vol(
            symbol=symbol,
            date=date,
            vol_data_dir=vol_data_dir,
            snapshot_root=snapshot_root,
            snapshot_feats=snapshot_feats,
            vol_feats=vol_feats,
            vol_labels=vol_labels,
            subsample=subsample,
        )
        return df_symbol.with_columns(pl.lit(symbol).alias("symbol"))

    with ThreadPoolExecutor(max_workers=min(5, len(symbols))) as executor:
        futures = {executor.submit(_build_one, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                symbol_to_df[symbol] = future.result()
            except Exception:
                failed_symbols.append(symbol)
                if on_error == "raise":
                    raise

    all_aligned = [symbol_to_df[s] for s in symbols if s in symbol_to_df]
    if not all_aligned:
        raise RuntimeError(f"all symbols failed: {failed_symbols}")

    df_all = pl.concat(all_aligned, how="vertical")
    sort_cols = [c for c in ("symbol", "ts") if c in df_all.columns]
    if sort_cols:
        df_all = df_all.sort(sort_cols)

    return df_all
