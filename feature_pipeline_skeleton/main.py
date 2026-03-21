from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

from data.config import SNAPSHOT_FEATURES, SNAPSHOT_ROOT, VOL_DATA_DIR, VOL_FEATURES
from factor_phase_II.build_rolling_npz import RollingNPZFastConfig
from feature_pipeline_skeleton.builders import (
    AlignedDataBuilder,
    BlockCacheBuilder,
    NeighborFeatureBuilder,
    ReadyBlockBuilder,
    TrainFromNPZBuilder,
    TrainFromNPZConfig,
    TrainingDatasetBuilder,
    WindowCacheBuilder,
)
from feature_pipeline_skeleton.pipeline import FeaturePipeline
from feature_pipeline_skeleton.stages import (
    BuildAlignedStage,
    BuildBlockCacheStage,
    BuildNeighborStage,
    BuildReadyBlockStage,
    BuildTrainingDatasetStage,
    BuildWindowCacheStage,
    TrainFromNPZStage,
)
from neighbor.window_cache_pipeline import RollingNeighborConfig

_DEFAULT_TARGET_COL = "y_vol_5m"
_DEFAULT_VOL_LABELS = ("vol_5m",)
_DEFAULT_FEATURE_COLS = (
    "vwap",
    "imbalance",
    "spread",
    "log_return",
    "vol_5m_lag0",
)
_DEFAULT_BASE_COLS = (_DEFAULT_TARGET_COL,)
_DEFAULT_N_LIST = (5, 10, 20, 50, 100)
_DEFAULT_BASE_COINS = (
    "BTCUSDT",
    "ETHUSDT",
    "XRPUSDT",
)


@dataclass(frozen=True)
class PipelineConfig:
    polars_max_threads: str | None = "21"

    aligned_root: str = "data_cache"
    cache_root: str = "data_cache"
    neighbor_output_root: str = "data_cache"

    ts_col: str = "ts"
    symbol_col: str = "symbol"

    aligned_subsample: str = "20ms"
    block_freq: str = "1h"

    train_window: str = "7d"
    step: str = "1d"
    train_subsample: str = "200ms"
    query_subsample: str = "20ms"
    neighbor_max_workers: int = 5

    compress_npz: bool = False
    train_downsample_freqs: Sequence[str] = ("100ms", "200ms", "500ms", "1s")
    dataset_window_mode: str = "by_day"
    train_days: int = 21
    validation_days: int = 1
    test_days: int = 1
    step_days: int = 1

    def apply_runtime_env(self) -> None:
        if self.polars_max_threads is not None:
            os.environ["POLARS_MAX_THREADS"] = self.polars_max_threads

    def neighbor_dir(self, date: str) -> str:
        return (
            f"{self.neighbor_output_root}/"
            f"neighbor_tw_{self.train_window}_step_{self.step}"
            f"_train_{self.train_subsample}_query_{self.query_subsample}/{date}"
        )

    def ready_block_dir(self, date: str) -> str:
        return f"{self.cache_root}/ready_blocks/freq_{self.query_subsample}/{date}"

    def training_data_dir(self, date: str) -> str:
        return f"{self.cache_root}/lgbm_rolling_training_data/{date}"

    def training_result_dir(self, date: str) -> str:
        return f"{self.cache_root}/lgbm_rolling_npz_results/{date}"


def build_cache_pipeline(date: str, config: PipelineConfig | None = None) -> FeaturePipeline:
    cfg = config or PipelineConfig()

    aligned_builder = AlignedDataBuilder(
        vol_data_dir=VOL_DATA_DIR,
        snapshot_root=SNAPSHOT_ROOT,
        snapshot_feats=list(SNAPSHOT_FEATURES),
        vol_feats=list(VOL_FEATURES),
        vol_labels=list(_DEFAULT_VOL_LABELS),
        subsample=cfg.aligned_subsample,
    )

    block_builder = BlockCacheBuilder(
        ts_col=cfg.ts_col,
        symbol_col=cfg.symbol_col,
    )

    return FeaturePipeline(
        stages=[
            BuildAlignedStage(
                aligned_root=cfg.aligned_root,
                builder=aligned_builder,
                ts_col=cfg.ts_col,
                symbol_col=cfg.symbol_col,
                target_col=_DEFAULT_TARGET_COL,
            ),
            BuildBlockCacheStage(
                aligned_root=cfg.aligned_root,
                block_cache_root=cfg.cache_root,
                builder=block_builder,
                subsample=cfg.aligned_subsample,
                block_freq=cfg.block_freq,
                ts_col=cfg.ts_col,
                symbol_col=cfg.symbol_col,
            ),
        ]
    )


def build_main_pipeline(date: str, config: PipelineConfig | None = None) -> FeaturePipeline:
    cfg = config or PipelineConfig()

    window_builder = WindowCacheBuilder(
        ts_col=cfg.ts_col,
        symbol_col=cfg.symbol_col,
    )

    neighbor_cfg = RollingNeighborConfig(
        train_window=cfg.train_window,
        step=cfg.step,
        train_subsample=cfg.train_subsample,
        query_subsample=cfg.query_subsample,
        feature_cols=_DEFAULT_FEATURE_COLS,
        base_cols=_DEFAULT_BASE_COLS,
        n_list=_DEFAULT_N_LIST,
        base_coins=_DEFAULT_BASE_COINS,
    )

    ready_npz_cfg = RollingNPZFastConfig(
        neighbor_dir=cfg.neighbor_dir(date),
        aligned_root=cfg.aligned_root,
        date=date,
        ready_block_dir=cfg.ready_block_dir(date),
        out_dir=cfg.training_data_dir(date),
        compress_npz=cfg.compress_npz,
        train_downsample_freqs=tuple(cfg.train_downsample_freqs),
        dataset_window_mode=cfg.dataset_window_mode,
        train_days=cfg.train_days,
        validation_days=cfg.validation_days,
        test_days=cfg.test_days,
        step_days=cfg.step_days,
    )

    ready_builder = ReadyBlockBuilder(cfg=ready_npz_cfg)
    training_builder = TrainingDatasetBuilder(cfg=ready_npz_cfg)
    train_builder = TrainFromNPZBuilder(
        cfg=TrainFromNPZConfig(
            data_root=ready_npz_cfg.out_dir,
            out_root=cfg.training_result_dir(date),
        )
    )

    return FeaturePipeline(
        stages=[
            BuildWindowCacheStage(
                block_root=cfg.cache_root,
                out_root=cfg.cache_root,
                feature_cols=_DEFAULT_FEATURE_COLS,
                base_cols=_DEFAULT_BASE_COLS,
                train_window=cfg.train_window,
                step=cfg.step,
                block_freq=cfg.block_freq,
                train_subsample=cfg.train_subsample,
                query_subsample=cfg.query_subsample,
                builder=window_builder,
                ts_col=cfg.ts_col,
                symbol_col=cfg.symbol_col,
            ),
            BuildNeighborStage(
                cache_root=cfg.cache_root,
                output_root=cfg.neighbor_output_root,
                builder=NeighborFeatureBuilder(config=neighbor_cfg),
                max_workers=cfg.neighbor_max_workers,
            ),
            BuildReadyBlockStage(builder=ready_builder),
            BuildTrainingDatasetStage(builder=training_builder),
            TrainFromNPZStage(builder=train_builder),
        ]
    )


def iter_dates(start_date: str, end_date: str) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if start > end:
        raise ValueError(f"start_date must be <= end_date, got {start_date} > {end_date}")

    out: list[str] = []
    day = start
    while day <= end:
        out.append(day.strftime("%Y-%m-%d"))
        day += timedelta(days=1)
    return out


def run_main_pipeline_range(
    *,
    start_date: str,
    end_date: str,
    symbols: list[str],
    config: PipelineConfig | None = None,
) -> None:
    cfg = config or PipelineConfig()
    cfg.apply_runtime_env()
    for date in iter_dates(start_date, end_date):
        print(f"[RANGE] start date={date}")
        pipeline = build_main_pipeline(date, config=cfg)
        pipeline.run(date=date, symbols=symbols)
        print(f"[RANGE] done date={date}")


def run_cache_pipeline_range(
    *,
    start_date: str,
    end_date: str,
    symbols: list[str],
    config: PipelineConfig | None = None,
) -> None:
    cfg = config or PipelineConfig()
    cfg.apply_runtime_env()
    for date in iter_dates(start_date, end_date):
        print(f"[CACHE-RANGE] start date={date}")
        pipeline = build_cache_pipeline(date, config=cfg)
        pipeline.run(date=date, symbols=symbols)
        print(f"[CACHE-RANGE] done date={date}")


if __name__ == "__main__":
    config = PipelineConfig()
    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "XRPUSDT",
    ]

    warmup_start_date = "2025-10-01"
    usable_start_date = "2025-10-08"
    usable_end_date = "2026-02-01"

    print(
        "[RANGE] "
        f"warmup_start_date={warmup_start_date} "
        f"usable_start_date={usable_start_date} "
        f"usable_end_date={usable_end_date}"
    )
    run_cache_pipeline_range(
        start_date=warmup_start_date,
        end_date=usable_end_date,
        symbols=symbols,
        config=config,
    )
    run_main_pipeline_range(
        start_date=usable_start_date,
        end_date=usable_end_date,
        symbols=symbols,
        config=config,
    )
