from __future__ import annotations

import os
import re
from typing import List, Optional, Union

import numpy as np
import polars as pl
from datetime import time

from data.config import DATA_INTERVAL_NS, GAP_TIME_NS

def trim_first_15min(df: pl.DataFrame, time_col: str = "ts") -> pl.DataFrame:
    assert df.is_empty() is False, "DataFrame is empty"
    assert df[time_col].dtype == pl.Datetime, f"{time_col} must be Datetime type"

    return df.filter(
        pl.col(time_col).dt.time() >= time(0, 15, 0)
    )

def filter_time_range(
    df: pl.DataFrame, 
    time_start: Optional[Union[int, str, object]], 
    time_end: Optional[Union[int, str, object]]
) -> pl.DataFrame:
    ts_start = _to_us_timestamp(time_start)
    ts_end = _to_us_timestamp(time_end)

    if ts_start is not None:
        df = df.filter(pl.col("timestamp") >= ts_start)
    if ts_end is not None:
        df = df.filter(pl.col("timestamp") <= ts_end)

    if df.is_empty():
        raise ValueError(
            f"[{symbol}] no data after time filtering, "
            f"check time_start/time_end (time_start={time_start}, time_end={time_end})"
        )
    return df

def _to_us_timestamp(value: Optional[Union[int, str, object]]) -> Optional[int]:
    """
    NOTE: This function should adapt to the downstream usage and may rewrite simplier later
    adapte start_time, end_time type in load_snapshot_raw
    """
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if hasattr(value, "value"):
        # Supports objects like pandas.Timestamp with ns-based integer value.
        try:
            return int(getattr(value, "value") // 1_000)
        except Exception:
            pass
    if isinstance(value, np.datetime64):
        ns_val = value.astype("datetime64[ns]").astype(np.int64)
        return int(ns_val // 1_000)

    ns_val = np.datetime64(value, "ns").astype(np.int64)
    return int(ns_val // 1_000)

