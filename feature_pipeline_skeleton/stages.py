from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import polars as pl

from analysis.evals import evaluate
from data.massive_data_adapter import _offset_ts
from feature_pipeline_skeleton.builders import (
    AlignedDataBuilder,
    BlockCacheBuilder,
    NeighborFeatureBuilder,
    ReadyBlockBuilder,
    TrainFromNPZBuilder,
    TrainingDatasetBuilder,
    WindowCacheBuilder,
)
from feature_pipeline_skeleton.state import PipelineState
from feature_pipeline_skeleton.readers import (
    AlignedSnapshotVolReader,
    BaseFactorColumnInspector,
    BlockSliceReader,
    NeighborInputReader,
    ReadyBlockReader,
    TrainFromNPZReader,
    RollingDatasetReader,
    WindowCacheReader,
    infer_neighbor_feature_cols,
)
from feature_pipeline_skeleton.sinks import (
    BlockCacheSink,
    NeighborFeatureSink,
    ReadyBlockSink,
    TrainFromNPZSink,
    RollingDatasetSink,
    WindowCacheSink,
)


def _build_neighbor_window(
    *,
    window_dir: str,
    output_dir: str,
    builder: NeighborFeatureBuilder,
    reader: NeighborInputReader,
    sink: NeighborFeatureSink,
    overwrite: bool,
) -> str | None:
    window_name = os.path.basename(window_dir)
    print(f"[NEIGHBOR][START] {window_name}", flush=True)
    window_inputs = reader.read_window(window_dir)
    enriched = builder.build(window_inputs)
    out_path = sink.write(
        output_dir=output_dir,
        cutoff_name=window_name,
        enriched=enriched,
        overwrite=overwrite,
    )
    print(f"[NEIGHBOR][DONE] {window_name}", flush=True)
    return out_path


@dataclass
class BuildAlignedStage:
    aligned_root: str
    builder: AlignedDataBuilder
    reader: Optional[AlignedSnapshotVolReader] = None
    inspector: Optional[BaseFactorColumnInspector] = None
    ts_col: str = "ts"
    symbol_col: str = "symbol"
    target_col: str = "y_vol_5m"

    def __post_init__(self) -> None:
        if self.reader is None:
            self.reader = AlignedSnapshotVolReader(
                vol_data_dir=self.builder.vol_data_dir,
                snapshot_root=self.builder.snapshot_root,
                snapshot_feats=self.builder.snapshot_feats,
                vol_labels=self.builder.vol_labels,
                subsample=self.builder.subsample,
            )
        if self.inspector is None:
            self.inspector = BaseFactorColumnInspector(
                aligned_root=self.aligned_root,
                subsample=self.builder.subsample,
                ts_col=self.ts_col,
                symbol_col=self.symbol_col,
                target_col=self.target_col,
            )

    def run(self, state: PipelineState) -> PipelineState:
        if self.reader is None:
            raise RuntimeError("aligned stage components are not initialized")
        if self.builder.sink is None:
            raise RuntimeError("aligned stage sink is not initialized")
        if self.builder.on_error not in {"skip", "raise"}:
            raise ValueError("on_error must be either 'skip' or 'raise'")

        result: dict[str, str] = {}
        failed_symbols: list[str] = []
        for symbol in state.symbols:
            try:
                raw_inputs = self.reader.read(symbol=symbol, date=state.date)
                df_symbol = self.builder.build(symbol=symbol, raw_inputs=raw_inputs)
                out_path = self.builder.sink.write(
                    df_symbol,
                    out_root=self.aligned_root,
                    date=state.date,
                    symbol=symbol,
                    subsample=self.builder.subsample,
                    overwrite=self.builder.overwrite,
                )
                result[symbol] = out_path
            except Exception:
                failed_symbols.append(symbol)
                if self.builder.on_error == "raise":
                    raise

        if failed_symbols:
            print(f"[WARN] failed symbols: {failed_symbols}")

        ok_symbols = [s for s in state.symbols if s in result]
        if not ok_symbols:
            raise RuntimeError("all symbols failed in BuildAlignedStage")

        base_feature_cols = self.inspector.inspect_base_feature_cols(
            date=state.date,
            sample_symbol=ok_symbols[0],
        )

        state.symbols = ok_symbols
        state.put("aligned_root", self.aligned_root)
        state.put("aligned_paths", result)
        state.put("aligned_subsample", self.builder.subsample)
        state.manifest.target_col = self.target_col
        state.manifest.set_base_features(base_feature_cols)
        return state


@dataclass
class BuildBlockCacheStage:
    aligned_root: str
    block_cache_root: str
    builder: BlockCacheBuilder
    subsample: str = "20ms"
    block_freq: str = "5m"
    ts_col: str = "ts"
    symbol_col: str = "symbol"
    overwrite: bool = True
    compression: str = "zstd"
    compression_level: int = 3
    include_target: bool = True
    reader: Optional[BlockSliceReader] = None
    sink: Optional[BlockCacheSink] = None

    def __post_init__(self) -> None:
        if self.reader is None:
            self.reader = BlockSliceReader(
                aligned_root=self.aligned_root,
                subsample=self.subsample,
                ts_col=self.ts_col,
                symbol_col=self.symbol_col,
            )
        if self.sink is None:
            self.sink = BlockCacheSink(
                compression=self.compression,
                compression_level=self.compression_level,
            )

    def run(self, state: PipelineState) -> PipelineState:
        if self.reader is None or self.sink is None:
            raise RuntimeError("block cache stage components are not initialized")

        columns = list(state.manifest.base_feature_cols)
        if self.include_target and state.manifest.target_col not in columns:
            columns.append(state.manifest.target_col)

        block_files: list[str] = []
        block_starts = self.reader.list_block_starts(
            symbols=state.symbols,
            date=state.date,
            block_freq=self.block_freq,
        )
        for block_start in block_starts:
            block_end = _offset_ts(block_start, self.block_freq)
            parts = [
                self.reader.read_symbol_block(
                    symbol=symbol,
                    date=state.date,
                    block_start=block_start,
                    block_end=block_end,
                    columns=columns,
                )
                for symbol in state.symbols
            ]
            df_block = self.builder.build(parts)
            out_path = self.sink.write(
                out_root=self.block_cache_root,
                date=state.date,
                block_freq=self.block_freq,
                block_start=block_start,
                df_block=df_block,
                overwrite=self.overwrite,
            )
            if out_path is not None:
                block_files.append(out_path)

        state.put("block_cache_root", self.block_cache_root)
        state.put("block_freq", self.block_freq)
        state.put("block_files", block_files)
        return state
    
@dataclass
class BuildWindowCacheStage:
    block_root: str
    out_root: str
    feature_cols: Sequence[str]
    builder: WindowCacheBuilder
    base_cols: Sequence[str] = ("y_vol_5m",)
    ts_col: str = "ts"
    symbol_col: str = "symbol"
    train_window: str = "7d"
    step: str = "1d"
    block_freq: str = "5m"
    train_subsample: str = "200ms"
    query_subsample: str = "20ms"
    compression: str = "zstd"
    compression_level: int = 3
    overwrite: bool = True
    reader: Optional[WindowCacheReader] = None
    sink: Optional[WindowCacheSink] = None

    def __post_init__(self) -> None:
        if self.reader is None:
            self.reader = WindowCacheReader(
                block_root=self.block_root,
                ts_col=self.ts_col,
                symbol_col=self.symbol_col,
            )
        if self.sink is None:
            self.sink = WindowCacheSink(
                compression=self.compression,
                compression_level=self.compression_level,
            )

    def run(self, state: PipelineState) -> PipelineState:
        if self.reader is None or self.sink is None:
            raise RuntimeError("window cache stage components are not initialized")

        if not state.symbols:
            raise ValueError("symbols cannot be empty")

        all_blocks, current_day_start, current_day_end, has_previous_history = self.reader.read_all_blocks(
            date=state.date,
            block_freq=self.block_freq,
            train_window=self.train_window,
            feature_cols=self.feature_cols,
            base_cols=self.base_cols,
        )
        if all_blocks.is_empty():
            raise ValueError("block cache is empty")
        if current_day_start is None or current_day_end is None:
            raise ValueError("current day bounds are missing")
        if not has_previous_history:
            raise ValueError(
                f"incomplete history for date={state.date}: need full {self.train_window} before {current_day_start}"
            )

        payload = self.builder.build_for_day(
            all_blocks=all_blocks,
            symbols=state.symbols,
            train_window=self.train_window,
            target_day_start=current_day_start,
            target_day_end=current_day_end,
            train_subsample=self.train_subsample,
            feature_cols=self.feature_cols,
            base_cols=self.base_cols,
        )
        out_dir = self.sink.write(
            out_root=self.out_root,
            date=state.date,
            train_window=self.train_window,
            step=self.step,
            train_subsample=self.train_subsample,
            query_subsample=self.query_subsample,
            payload=payload,
            overwrite=self.overwrite,
        )
        window_dirs = [out_dir] if out_dir is not None else []
        if not window_dirs:
            raise RuntimeError(f"window cache stage produced no files for date={state.date}")

        state.put("window_cache_root", self.infer_output_dir(state.date))
        state.put("window_cache_dirs", window_dirs)
        return state

    def infer_output_dir(self, date: str) -> str:
        return os.path.join(
            self.out_root,
            (
                f"window_cache_"
                f"tw_{self.train_window}"
                f"_step_{self.step}"
                f"_train_{self.train_subsample}"
                f"_query_{self.query_subsample}"
            ),
            date,
        )


@dataclass
class BuildNeighborStage:
    cache_root: str
    output_root: str
    builder: NeighborFeatureBuilder
    max_workers: Optional[int] = None
    overwrite: bool = True
    compression: str = "zstd"
    compression_level: int = 3
    reader: Optional[NeighborInputReader] = None
    sink: Optional[NeighborFeatureSink] = None

    def __post_init__(self) -> None:
        if self.reader is None:
            self.reader = NeighborInputReader(
                cache_root=self.cache_root,
                config=self.builder.config,
            )
        if self.sink is None:
            self.sink = NeighborFeatureSink(
                compression=self.compression,
                compression_level=self.compression_level,
            )

    def infer_output_dir(self, date: str) -> str:
        config = self.builder.config
        return os.path.join(
            self.output_root,
            (
                f"neighbor_tw_{config.train_window}"
                f"_step_{config.step}"
                f"_train_{config.train_subsample}"
                f"_query_{config.query_subsample}"
            ),
            date,
        )

    def run(self, state: PipelineState) -> PipelineState:
        if self.reader is None or self.sink is None:
            raise RuntimeError("neighbor stage components are not initialized")

        window_dirs = self.reader.list_window_dirs(state.date)
        if not window_dirs:
            raise ValueError(f"no window dirs found for date={state.date}")

        neighbor_dir = self.infer_output_dir(state.date)
        os.makedirs(neighbor_dir, exist_ok=True)

        worker_n = self.max_workers or min(os.cpu_count() or 1, len(window_dirs))
        done: list[str] = []
        print(
            f"[NEIGHBOR] date={state.date} windows={len(window_dirs)} max_workers={worker_n}",
            flush=True,
        )

        with ProcessPoolExecutor(max_workers=worker_n) as ex:
            futures = {
                ex.submit(
                    _build_neighbor_window,
                    window_dir=window_dir,
                    output_dir=neighbor_dir,
                    builder=self.builder,
                    reader=self.reader,
                    sink=self.sink,
                    overwrite=self.overwrite,
                ): window_dir
                for window_dir in window_dirs
            }

            for idx, fut in enumerate(as_completed(futures), start=1):
                window_dir = futures[fut]
                try:
                    out_path = fut.result()
                    if out_path is not None:
                        done.append(out_path)
                        print(
                            f"[OK] neighbor ({idx}/{len(window_dirs)}) -> {out_path}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[NEIGHBOR][SKIP] ({idx}/{len(window_dirs)}) {os.path.basename(window_dir)}",
                            flush=True,
                        )
                except Exception as exc:
                    print(f"[FAIL] {window_dir}: {exc}", flush=True)

        if not done:
            raise RuntimeError(f"neighbor stage produced no files for date={state.date}")

        neighbor_feature_cols = infer_neighbor_feature_cols(
            output_root=self.output_root,
            date=state.date,
            config=self.builder.config,
        )

        state.put("neighbor_dir", neighbor_dir)
        state.put("neighbor_files", sorted(done))
        state.manifest.add_derived_group("neighbor", neighbor_feature_cols)
        return state



@dataclass
class BuildReadyBlockStage:
    builder: ReadyBlockBuilder
    reader: Optional[ReadyBlockReader] = None
    sink: Optional[ReadyBlockSink] = None

    def __post_init__(self) -> None:
        if self.reader is None:
            self.reader = ReadyBlockReader(cfg=self.builder.cfg)
        if self.sink is None:
            self.sink = ReadyBlockSink(cfg=self.builder.cfg)

    def run(self, state: PipelineState) -> PipelineState:
        if self.reader is None or self.sink is None:
            raise RuntimeError("ready block stage components are not initialized")

        os.makedirs(self.builder.cfg.ready_block_dir, exist_ok=True)
        neighbor_files = self.reader.list_neighbor_files()
        self.sink.write_meta(num_neighbor_blocks=len(neighbor_files))
        print(
            f"[READY] source_blocks={len(neighbor_files)} out_dir={self.builder.cfg.ready_block_dir}",
            flush=True,
        )

        built = 0
        skipped = 0
        for idx, path in enumerate(neighbor_files, start=1):
            block_name = os.path.splitext(os.path.basename(path))[0]
            print(f"[READY][PROCESS] ({idx}/{len(neighbor_files)}) {block_name}", flush=True)
            neighbor_df = self.reader.read_neighbor_block(path)
            base_df = self.reader.read_base_slice(neighbor_df)
            ready_df = self.builder.build(neighbor_df, base_df)
            if self.sink.write_block(block_name, ready_df):
                print(f"[READY][BUILD] {block_name}")
                built += 1
            else:
                skipped += 1

        result = {
            "num_neighbor_blocks": len(neighbor_files),
            "built": built,
            "skipped": skipped,
            "ready_block_dir": self.builder.cfg.ready_block_dir,
        }
        state.put("ready_block_dir", result["ready_block_dir"])
        state.put("ready_block_result", result)
        return state


@dataclass
class BuildTrainingDatasetStage:
    builder: TrainingDatasetBuilder
    reader: Optional[RollingDatasetReader] = None
    sink: Optional[RollingDatasetSink] = None

    def __post_init__(self) -> None:
        if self.reader is None:
            self.reader = RollingDatasetReader(cfg=self.builder.cfg)
        if self.sink is None:
            self.sink = RollingDatasetSink(cfg=self.builder.cfg)


    def _run_by_day(self, state: PipelineState) -> PipelineState:
        """
        按照天切数据, 对每个切好的样本降频成不同的频率的训练数据
        """
        if self.reader is None or self.sink is None:
            raise RuntimeError("training dataset stage components are not initialized")

        cfg = self.builder.cfg
        os.makedirs(cfg.out_dir, exist_ok=True)

        # 找到所有日期文件
        ready_groups = self.reader.list_ready_files_grouped_by_date()
        total_days = len(ready_groups)
        
        # 检验文件是否足够
        required_days = cfg.train_days + cfg.validation_days + cfg.test_days
        if total_days < required_days:
            raise ValueError(
                f"Not enough ready day directories: total_days={total_days}, "
                f"required_days={required_days}"
            )

        all_ready_files = [path for _, files in ready_groups for path in files]
        symbol_mapping = self.reader.get_symbol_mapping(all_ready_files)
        print(
            "[ROLLING][DAY] "
            f"ready_days_total={total_days} "
            f"train_days={cfg.train_days} "
            f"validation_days={cfg.validation_days} "
            f"test_days={cfg.test_days} "
            f"step_days={cfg.step_days}",
            flush=True,
        )
        for freq in cfg.train_downsample_freqs:
            self.sink.write_run_config(
                freq=freq,
                symbol_mapping=symbol_mapping,
                num_total_ready_blocks=len(all_ready_files),
                num_total_ready_days=total_days,
            )
            print(f"[ROLLING][DAY] initialized freq={freq}", flush=True)

        block_count_by_freq = {freq: 0 for freq in cfg.train_downsample_freqs}
        skipped_by_freq = {freq: 0 for freq in cfg.train_downsample_freqs}

        window_idx = 0
        for start_idx in range(0, total_days - required_days + 1, cfg.step_days):
            train_groups = ready_groups[start_idx : start_idx + cfg.train_days]
            val_groups = ready_groups[
                start_idx + cfg.train_days : start_idx + cfg.train_days + cfg.validation_days
            ]
            test_groups = ready_groups[
                start_idx + cfg.train_days + cfg.validation_days : start_idx + required_days
            ]
            window_idx += 1

            train_dates = [day for day, _ in train_groups]
            val_dates = [day for day, _ in val_groups]
            test_dates = [day for day, _ in test_groups]
            block_name = f"window_{window_idx:03d}_{test_dates[0]}"
            print(
                "[ROLLING][DAY][BUILD] "
                f"window={window_idx} "
                f"train={train_dates[0]}->{train_dates[-1]} "
                f"val={val_dates[0]}->{val_dates[-1]} "
                f"test={test_dates[0]}->{test_dates[-1]}",
                flush=True,
            )

            train_files = [path for _, files in train_groups for path in files]
            val_files = [path for _, files in val_groups for path in files]
            test_files = [path for _, files in test_groups for path in files]

            train_df_raw = pl.concat(
                [self.reader.load_prepared_block(path, symbol_mapping) for path in train_files],
                how="vertical",
            )
            val_df_raw = pl.concat(
                [self.reader.load_prepared_block(path, symbol_mapping) for path in val_files],
                how="vertical",
            )
            test_df_raw = pl.concat(
                [self.reader.load_prepared_block(path, symbol_mapping) for path in test_files],
                how="vertical",
            )

            if train_df_raw.is_empty() or val_df_raw.is_empty() or test_df_raw.is_empty():
                print(f"[ROLLING][DAY][WARN] skip window={window_idx}: raw train/val/test empty")
                for freq in cfg.train_downsample_freqs:
                    skipped_by_freq[freq] += 1
                continue

            for freq in cfg.train_downsample_freqs:
                val_payload = self.builder.build_for_freq(
                    train_df_raw=train_df_raw,
                    pred_df_raw=val_df_raw,
                    freq=freq,
                )
                test_payload = self.builder.build_for_freq(
                    train_df_raw=train_df_raw,
                    pred_df_raw=test_df_raw,
                    freq=freq,
                )
                if not val_payload or not test_payload:
                    print(
                        f"[ROLLING][DAY][WARN] skip window={window_idx} @ {freq}: empty after agg",
                        flush=True,
                    )
                    skipped_by_freq[freq] += 1
                    continue

                self.sink.write_block_outputs(
                    freq=freq,
                    block_name=block_name,
                    payload=test_payload,
                    train_files=[os.path.basename(path) for path in train_files],
                    pred_file=",".join(test_dates),
                    symbol_mapping=symbol_mapping,
                    validation_payload=val_payload,
                    extra_meta={
                        "window_index": window_idx,
                        "train_dates": train_dates,
                        "validation_dates": val_dates,
                        "test_dates": test_dates,
                        "validation_files": [os.path.basename(path) for path in val_files],
                        "test_files": [os.path.basename(path) for path in test_files],
                    },
                )
                block_count_by_freq[freq] += 1

        if all(v == 0 for v in block_count_by_freq.values()):
            raise RuntimeError("No npz training data generated for any frequency.")

        result = {
            "num_blocks_by_freq": block_count_by_freq,
            "num_skipped_by_freq": skipped_by_freq,
            "out_dir": cfg.out_dir,
            "freqs": list(cfg.train_downsample_freqs),
            "dataset_window_mode": "by_day",
            "train_days": cfg.train_days,
            "validation_days": cfg.validation_days,
            "test_days": cfg.test_days,
            "step_days": cfg.step_days,
        }
        state.put("training_out_dir", cfg.out_dir)
        state.put("training_result", result)
        state.put("train_downsample_freqs", list(cfg.train_downsample_freqs))
        return state

    def run(self, state: PipelineState) -> PipelineState:
        return self._run_by_day(state)


@dataclass
class TrainFromNPZStage:
    builder: TrainFromNPZBuilder
    reader: Optional[TrainFromNPZReader] = None
    sink: Optional[TrainFromNPZSink] = None

    def __post_init__(self) -> None:
        if self.reader is None:
            self.reader = TrainFromNPZReader(cfg=self.builder.cfg)
        if self.sink is None:
            self.sink = TrainFromNPZSink(cfg=self.builder.cfg)

    def run(self, state: PipelineState) -> PipelineState:
        if self.reader is None or self.sink is None:
            raise RuntimeError("train-from-npz stage components are not initialized")

        data_root = self.reader.resolve_data_root(state)
        freqs = self.reader.resolve_freqs(state, data_root)
        if not freqs:
            raise ValueError(f"no training frequencies found under {data_root}")

        os.makedirs(self.builder.cfg.out_root, exist_ok=True)
        summary: dict[str, Any] = {}
        for freq in freqs:
            npz_files = self.reader.list_train_npz_files(data_root=data_root, freq=freq)
            self.sink.prepare_freq_dirs(freq)
            built_blocks = 0
            skipped_blocks = 0

            for path in npz_files:
                block_name = os.path.splitext(os.path.basename(path))[0]
                pred_path = self.sink.block_prediction_path(freq=freq, block_name=block_name)
                if (not self.builder.cfg.overwrite) and os.path.exists(pred_path):
                    print(f"[TRAIN][SKIP] freq={freq} block={block_name}", flush=True)
                    skipped_blocks += 1
                    continue

                print(f"[TRAIN][BUILD] freq={freq} block={block_name}", flush=True)
                payload = self.reader.load_npz(path)
                built = self.builder.build(block_name=block_name, payload=payload)
                self.sink.write_block_outputs(freq=freq, built=built)
                print(
                    "[TRAIN][METRIC] "
                    f"freq={freq} block={block_name} "
                    f"RMSE={built['metrics']['rmse']:.6f} "
                    f"MAE={built['metrics']['mae']:.6f} "
                    f"CORR={built['metrics']['corr']:.6f} "
                    f"QLIKE={built['metrics']['qlike']:.6f}",
                    flush=True,
                )
                built_blocks += 1

            freq_out_dir = self.sink.freq_out_dir(freq)
            pred_block_files = self.reader.list_block_npz_files(out_dir=freq_out_dir, kind="pred_blocks")
            metric_block_files = self.reader.list_block_npz_files(out_dir=freq_out_dir, kind="metric_blocks")
            importance_block_files = self.reader.list_block_npz_files(
                out_dir=freq_out_dir,
                kind="importance_blocks",
            )
            if not pred_block_files or not metric_block_files or not importance_block_files:
                raise RuntimeError(f"no trained block outputs found for freq={freq}")

            pred_df = pl.concat(
                [self.reader.load_prediction_block(path) for path in pred_block_files],
                how="vertical",
            )
            metric_df = pl.concat(
                [self.reader.load_metric_block(path) for path in metric_block_files],
                how="vertical",
            )
            importance_df = pl.concat(
                [self.reader.load_importance_block(path) for path in importance_block_files],
                how="vertical",
            )

            overall = evaluate(
                y_true=pred_df.get_column("y_true").to_numpy(),
                y_pred=pred_df.get_column("y_pred").to_numpy(),
            )
            overall_df = pl.DataFrame(
                [
                    {
                        "model_type": "lgbm_from_npz",
                        "symbol": "ALL",
                        **overall,
                        "num_rows": int(pred_df.height),
                        "num_test_blocks": int(metric_df.height),
                    }
                ]
            )
            importance_mean_df = (
                importance_df.group_by("feature")
                .agg(
                    [
                        pl.col("importance_gain").mean().alias("importance_gain_mean"),
                        pl.col("importance_split").mean().alias("importance_split_mean"),
                    ]
                )
                .sort("importance_gain_mean", descending=True)
            )

            self.sink.write_freq_summary(
                freq=freq,
                pred_df=pred_df,
                metric_df=metric_df,
                overall_df=overall_df,
                importance_df=importance_df,
                importance_mean_df=importance_mean_df,
                overall=overall,
                train_run_config=self.reader.load_run_config(data_root=data_root, freq=freq),
            )

            summary[freq] = {
                "num_input_blocks": len(npz_files),
                "built_blocks": built_blocks,
                "skipped_blocks": skipped_blocks,
                "num_output_blocks": int(metric_df.height),
                "overall": overall,
                "out_dir": freq_out_dir,
            }

        self.sink.write_stage_summary(summary)
        state.put("model_out_root", self.builder.cfg.out_root)
        state.put("train_from_npz_result", summary)
        return state
