import math

import polars as pl
from polars.datatypes import Date, Datetime


def construct_factors_from_snapshot(df: pl.DataFrame) -> pl.DataFrame:
    """
    从 snapshot 原始列构造训练用的特征列(这里展示一些简单的因子):
    - ts
    - vwap
    - imbalance
    - spread
    - mid
    - log_return
    """
    levels = range(25)

    ask_p_cols = [f"asks[{i}].price" for i in levels]
    bid_p_cols = [f"bids[{i}].price" for i in levels]
    ask_a_cols = [f"asks[{i}].amount" for i in levels]
    bid_a_cols = [f"bids[{i}].amount" for i in levels]

    required_cols = ask_p_cols + bid_p_cols + ask_a_cols + bid_a_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required snapshot columns (sample): {missing[:10]}, total={len(missing)}"
        )

    ask_amt_sum = pl.sum_horizontal([pl.col(c).cast(pl.Float64) for c in ask_a_cols])
    bid_amt_sum = pl.sum_horizontal([pl.col(c).cast(pl.Float64) for c in bid_a_cols])

    ask_notional = pl.sum_horizontal(
        [pl.col(p).cast(pl.Float64) * pl.col(a).cast(pl.Float64) for p, a in zip(ask_p_cols, ask_a_cols)]
    )
    bid_notional = pl.sum_horizontal(
        [pl.col(p).cast(pl.Float64) * pl.col(a).cast(pl.Float64) for p, a in zip(bid_p_cols, bid_a_cols)]
    )

    total_amt = ask_amt_sum + bid_amt_sum
    vwap_expr = pl.when(total_amt != 0).then((ask_notional + bid_notional) / total_amt).otherwise(None)
    imbalance_expr = pl.when(total_amt != 0).then((ask_amt_sum - bid_amt_sum) / total_amt).otherwise(None)

    factors = df.select(
        [
            pl.col("ts"),
            vwap_expr.alias("vwap"),
            imbalance_expr.alias("imbalance"),
            (pl.col("asks[0].price").cast(pl.Float64) - pl.col("bids[0].price").cast(pl.Float64)).alias("spread"),
            (
                (pl.col("asks[0].price").cast(pl.Float64) + pl.col("bids[0].price").cast(pl.Float64)) / 2.0
            ).alias("mid"),
        ]
    ).with_columns(
        pl.col("mid").log().diff().alias("log_return")
    )

    numeric_cols = ["vwap", "imbalance", "spread", "mid", "log_return"]
    factors = factors.with_columns(
        [
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c in numeric_cols
        ]
    )

    return factors


def add_factors(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add derived features used by training pipeline.
    """
    out = df.with_columns(
        [
            (pl.col("spread") / pl.col("mid")).alias("spread_pct"),
            ((pl.col("vwap") - pl.col("mid")) / pl.col("mid")).alias("vwap_mid_diff"),
            (pl.col("vol_1m_lag0") / (pl.col("vol_5m_lag0") + 1e-9)).alias("vol_ratio_1_5"),
            (pl.col("vol_5m_lag0") / (pl.col("vol_30m_lag0") + 1e-9)).alias("vol_ratio_5_30"),
            (pl.col("vol_5m_lag0") / (pl.col("vol_60m_lag0") + 1e-9)).alias("vol_ratio_5_60"),
            (pl.col("vol_30m_lag0") / (pl.col("vol_120m_lag0") + 1e-9)).alias("vol_ratio_30_120"),
            (pl.col("imbalance") ** 2).alias("imbalance_sq"),
            pl.col("imbalance").abs().alias("abs_imbalance"),
            (pl.col("log_return") ** 2).alias("lr_sq"),
            pl.col("log_return").abs().alias("lr_abs"),
        ]
    )

    if "ts" not in out.columns:
        return out

    ts_dtype = out.schema.get("ts")
    if isinstance(ts_dtype, Datetime):
        ts_expr = pl.col("ts")
    else:
        return out

    minutes_expr = (pl.col("_ts_dt").dt.hour() * 60 + pl.col("_ts_dt").dt.minute()).cast(pl.Float64)

    return out.with_columns(ts_expr.alias("_ts_dt")).with_columns(
        [
            pl.col("_ts_dt").dt.hour().alias("hour"),
            pl.col("_ts_dt").dt.minute().alias("minute"),
            (minutes_expr * (2.0 * math.pi / 1440.0)).sin().alias("tod_sin"),
            (minutes_expr * (2.0 * math.pi / 1440.0)).cos().alias("tod_cos"),
        ]
    ).drop("_ts_dt")