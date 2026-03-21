from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

SNAPSHOT_ROOT: str = ""
VOL_DATA_DIR: str = ""

DATA_INTERVAL_NS: int = 20_000_000          # 20ms（vol NPZ 标准间隔）
GAP_TIME_NS: int = int(15 * 60 * 1e9)       # 前 15 分钟过滤（纳秒）

BASE_COINS: List[str] = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT" # 保留3个币种作为展示
]
COIN_UNIVERSE: List[str] = ['BTCUSDT', 'ETHUSDT']

SNAPSHOT_FEATURES: List[str] = (
    [f"asks[{i}].price" for i in range(25)] +
    [f"bids[{i}].price" for i in range(25)] +
    [f"asks[{i}].amount" for i in range(25)] +
    [f"bids[{i}].amount" for i in range(25)]
)

VOL_FEATURES: List[str] = [
    "vol_1m", "vol_5m", "vol_10m", "vol_15m", "vol_30m", "vol_60m", "vol_120m"
]

ALL_FEATURES: List[str] = [
    # snapshot 原始特征
    'vwap', 'imbalance', 'spread', 'mid', 'log_return',
    # vol 特征（默认 lag 0）
    'vol_1m_lag0','vol_5m_lag0','vol_10m_lag0','vol_15m_lag0','vol_30m_lag0','vol_60m_lag0','vol_120m_lag0',
    # 派生特征
    'spread_pct', 'vwap_mid_diff', 'vol_ratio_1_5', 'vol_ratio_5_30', 'vol_ratio_5_60', 'vol_ratio_30_120', 'imbalance_sq', 
    'abs_imbalance', 'lr_sq', 'lr_abs',
    # 时间特征
    'hour', 'minute', 'tod_sin', 'tod_cos',
    # 未来标签（默认 5m vol）
    'y_vol_5m'
]


# 当用 vol 作为 label 进行预测时，由于vol 是在[t-k,t]上构造的feature，
# 因此 label 需要向前 shift k 步（每步 20ms）才能与特征对齐
VOL_LABEL: Dict[str, int] = {
        "vol_1m" :  1 * 3_000,   # 1min  / 20ms = 60 * 1000 / 20
        "vol_5m" :  5 * 3_000,   # 5min  / 20ms
        "vol_10m": 10 * 3_000,   # 10min / 20ms
        "vol_15m": 15 * 3_000,   # 15min / 20ms
        "vol_30m": 30 * 3_000,   # 30min / 20ms
        "vol_60m": 60 * 3_000,   # 60min / 20ms
        "vol_120m": 120 * 3_000, # 120min / 20ms
}


@dataclass
class DataConfig:
    """
    数据 Pipeline 统一配置对象。

    所有路径、常量、列名、币种列表集中管理，
    函数签名接收 DataConfig，不在内部读全局变量。
    """

    # ── 路径 ──────────────────────────────────────────────────────────────────
    vol_data_dir: str = VOL_DATA_DIR
    snapshot_root: str = SNAPSHOT_ROOT

    # ── 日期 ──────────────────────────────────────────────────────────────────
    date: str = "2025-10-01"                              # "YYYY-MM-DD"

    # ── 币种 ──────────────────────────────────────────────────────────────────
    symbols: List[str] = field(default_factory=lambda: list(BASE_COINS))

    # ── 特征 / 标签列 ─────────────────────────────────────────────────────────
    snapshot_features: List[str] = field(
        default_factory=lambda: list(SNAPSHOT_FEATURES)
    )
    y_cols: List[str] = field(default_factory=lambda: ["y_vol_5m"])

    # ── 时间常量 ──────────────────────────────────────────────────────────────
    data_interval_ns: int = DATA_INTERVAL_NS    # 20ms（vol NPZ 标准间隔）
    gap_time_ns: int = GAP_TIME_NS              # vol 前 N 纳秒过滤（默认 15min）
    subsample: str = "20ms"                     # snapshot 重采样粒度

    @property
    def x_cols(self) -> List[str]:
        """snapshot 特征列（含 ts），与旧代码 x_cols 语义一致。"""
        return ["ts"] + self.snapshot_features

