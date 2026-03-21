from __future__ import annotations

import os
import re
from typing import List, Optional, Union

import numpy as np
import polars as pl
from datetime import time

from data.config import DATA_INTERVAL_NS, GAP_TIME_NS
from data.utils import trim_first_15min, filter_time_range

from datetime import datetime, timezone

def load_snapshot_raw(
    snapshot_root: str,
    symbol: str,
    date: str,
    x_cols: List[str],
    *,
    subsample: str = "20ms",
    time_start: Optional[Union[int, str, object]] = None,
    time_end: Optional[Union[int, str, object]] = None,
) -> pl.DataFrame:
    """
    从 raw snapshot 读取订单簿快照（优先 .csv.gz，其次 .csv.zst），
    可选时间过滤，重采样到固定间隔。

    注意: 
        虽然snapshot 是固定频率采样的，但仍可能存在同一时间戳不规律的情况
        这里的重采样逻辑是先按时间戳排序，再对同一时间戳的记录保留最后一条
    """
    # base_dir = os.path.join(snapshot_root, symbol)
    base_dir = os.path.join(snapshot_root, symbol, "book_snapshot_25")
    fname_prefix = f"binance-futures_book_snapshot_25_{date}_{symbol}.csv"

    path_gz = os.path.join(base_dir, f"{fname_prefix}.gz")
    path_zst = os.path.join(base_dir, f"{fname_prefix}.zst")

    if os.path.exists(path_gz):
        path = path_gz
    elif os.path.exists(path_zst):
        path = path_zst
    else:
        raise FileNotFoundError(
            f"snapshot file does not exist, tried:\n- {path_gz}\n- {path_zst}"
        )

    assert "ts" not in x_cols, "x_cols should not include 'ts'"

    usecols = ["timestamp"] + x_cols

    schema_overrides = {"timestamp": pl.Int64}
    for c in x_cols:
        schema_overrides[c] = pl.Float64

    df = pl.read_csv(
        path,
        columns=usecols,
        schema_overrides=schema_overrides,
        infer_schema_length=0 # 不做数据类型推断
    )

    assert "timestamp" in df.columns, "snapshot data must include 'timestamp' column"
    assert df.schema["timestamp"] in (pl.Int64, pl.Datetime), (
        f"unexpected timestamp dtype: {df.schema['timestamp']}, "
        "expected Int64 (us) or Datetime"
    )

    df = filter_time_range(df, time_start, time_end)

    if df.height == 0:
        return pl.DataFrame(schema={"ts": pl.Datetime, **{c: pl.Float64 for c in x_cols}})

    # timestamp -> ts
    if df.schema["timestamp"] == pl.Int64:
        df = df.with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="us").alias("ts")
        )
    else:
        df = df.with_columns(
            pl.col("timestamp").alias("ts")
        )

    df = df.sort("ts")

    # 同一 ts 多条时保留最后一条
    df = (
        df.group_by("ts", maintain_order=True)
        .agg([pl.col(c).last().alias(c) for c in x_cols])
        .sort("ts")
    )

    grid_ns = (
        pl.select(
            pl.duration(**_parse_fixed_duration(subsample))
            .dt.total_nanoseconds()
            .alias("ns")
        )
        .item()
    )

    ts_min_ns = df.select(pl.col("ts").dt.epoch(time_unit="ns").min()).item()
    ts_max_ns = df.select(pl.col("ts").dt.epoch(time_unit="ns").max()).item()

    # 起点向下取整，终点向上取整
    grid_start_ns = (ts_min_ns // grid_ns) * grid_ns
    grid_end_ns = ((ts_max_ns + grid_ns - 1) // grid_ns) * grid_ns

    grid_start_dt = datetime.fromtimestamp(grid_start_ns / 1e9, tz=timezone.utc).replace(tzinfo=None)
    grid_end_dt = datetime.fromtimestamp(grid_end_ns / 1e9, tz=timezone.utc).replace(tzinfo=None)

    ts_grid = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                start=grid_start_dt,
                end=grid_end_dt,
                interval=subsample,
                eager=True,
            )
        }
    )

    out = ts_grid.join_asof(
        df,
        on="ts",
        strategy="backward",
    )

    return out.select(["ts"] + x_cols)

def load_and_align_volatility(
    symbol: str,
    date: str,
    vol_data_dir: str,
    *,
    data_interval_ns: Optional[int] = None,
    grid_resolution: str = "20ms",
    start_time: Optional[Union[int, str, object]] = None,
    end_time: Optional[Union[int, str, object]] = None,
) -> pl.DataFrame:
    """
    读取单币种 volatility NPZ, 对齐到时间格 (ts_grid)
    """
    interval_ns = data_interval_ns if data_interval_ns is not None else DATA_INTERVAL_NS

    path = os.path.join(vol_data_dir, symbol, f"{date}_volatility.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"vol file does not exist: {path}")

    vol_raw = np.load(path)["data"]
    ts_ns_np: np.ndarray = vol_raw["timestamp"].astype(np.int64)

    if len(ts_ns_np) < 2:
        raise ValueError(f"[{symbol}] vol data is too short ({len(ts_ns_np)} rows)")

    diffs = np.unique(np.diff(ts_ns_np))
    if not np.all(diffs == interval_ns):
        raise ValueError(
            f"[{symbol}] vol interval distance mismatch, "
            f"expected {interval_ns}ns, got: {diffs}"
        )

    df_vol = pl.DataFrame(vol_raw)
    df_vol = filter_time_range(df_vol, start_time, end_time)
    
    if df_vol.schema["timestamp"] != pl.Int64:
        raise TypeError("vol timestamp must be Int64 (ns)")

    vol_value_cols = [c for c in df_vol.columns if c != "timestamp"]

    grid_resolution_ns = (
        pl.select(
            pl.duration(**_parse_fixed_duration(grid_resolution))
            .dt.total_nanoseconds()
            .alias("ns")
        )
        .item()
    )

    ts_ns = pl.col("timestamp")
    ts_grid_ns = ((ts_ns + grid_resolution_ns - 1) // grid_resolution_ns) * grid_resolution_ns

    df_final = (
        df_vol.with_columns(
            [
                pl.from_epoch("timestamp", time_unit="ns").alias("timestamp"),
                pl.from_epoch(ts_grid_ns, time_unit="ns").alias("ts_grid"),
            ]
        )
        .group_by("ts_grid", maintain_order=True)
        .agg(
            [
                pl.col("timestamp").last().alias("timestamp"),
                *[pl.col(c).last().alias(c) for c in vol_value_cols],
            ]
        )
    )

    return df_final

def _parse_fixed_duration(resolution: str) -> dict:
    """
    辅助函数: 帮助函数可以解析固定单位的时间字符串（如 "20ms"）并转换成 Polars 可接受的 duration 参数
    """
    match = re.fullmatch(r"\s*(\d+)\s*(ns|us|ms|s|m|h)\s*", resolution)
    if match is None:
        raise ValueError(
            f"unsupported grid_resolution: {resolution!r}; "
            "expected fixed units in {ns, us, ms, s, m, h}"
        )

    value = int(match.group(1))
    unit = match.group(2)

    mapping = {
        "ns": {"nanoseconds": value},
        "us": {"microseconds": value},
        "ms": {"milliseconds": value},
        "s": {"seconds": value},
        "m": {"minutes": value},
        "h": {"hours": value},
    }
    return mapping[unit]
