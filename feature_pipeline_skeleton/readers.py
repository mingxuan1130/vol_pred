from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, Sequence

import numpy as np
import polars as pl

from data.massive_data_adapter import (
    _offset_ts,
    _symbol_parquet_path,
    _window_cache_dir,
    load_aligned_symbol_parquet,
    load_block_cache,
)
from data.raw_loader import load_and_align_volatility, load_snapshot_raw
from factor_phase_II.build_rolling_npz import (
    _prepare_raw_block_df,
    add_symbol_code_single_df,
    get_global_symbol_mapping_from_ready_blocks,
    list_ready_block_files,
)
from factor_phase_II.build_training_blocks import (
    _load_aligned_slice_for_symbols,
    list_neighbor_files,
)
from neighbor.window_cache_pipeline import RollingNeighborConfig


@dataclass
class AlignedSnapshotVolReader:
    vol_data_dir: str
    snapshot_root: str
    snapshot_feats: list[str]
    vol_labels: list[str]
    subsample: str = "20ms"

    def read(self, symbol: str, date: str) -> dict[str, pl.DataFrame]:
        snapshot_df = load_snapshot_raw(
            snapshot_root=self.snapshot_root,
            symbol=symbol,
            date=date,
            x_cols=self.snapshot_feats,
            subsample=self.subsample,
        )
        vol_df = load_and_align_volatility(
            symbol=symbol,
            date=date,
            vol_data_dir=self.vol_data_dir,
        )

        bad_labels = [c for c in self.vol_labels if c not in vol_df.columns]
        if bad_labels:
            raise ValueError(f"vol_labels not found in vol table: {bad_labels}")

        return {
            "snapshot_df": snapshot_df,
            "vol_df": vol_df,
        }


@dataclass
class BaseFactorColumnInspector:
    aligned_root: str
    subsample: str = "20ms"
    ts_col: str = "ts"
    symbol_col: str = "symbol"
    target_col: str = "y_vol_5m"

    def inspect_base_feature_cols(self, date: str, sample_symbol: str) -> list[str]:
        lf = load_aligned_symbol_parquet(
            out_root=self.aligned_root,
            date=date,
            symbol=sample_symbol,
            subsample=self.subsample,
            lazy=True,
        )
        cols = list(lf.collect_schema().names())
        exclude = {self.ts_col, self.symbol_col, self.target_col}
        return [c for c in cols if c not in exclude]


@dataclass
class BlockSliceReader:
    aligned_root: str
    subsample: str = "20ms"
    ts_col: str = "ts"
    symbol_col: str = "symbol"

    def list_block_starts(self, symbols: Sequence[str], date: str, block_freq: str) -> list[Any]:
        # 拼出路径: aligned_{subsample}/{date}/{symbol}/data.parquet
        sample_path = _symbol_parquet_path(
            out_root=self.aligned_root,
            date=date,
            symbol=symbols[0],
            subsample=self.subsample,
        )
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"sample parquet not found: {sample_path}")

        # 读取第一个样本的 "ts" 列，找出最小和最大时间戳
        # 构成生成block时间戳的依据
        df_time = pl.read_parquet(sample_path, columns=[self.ts_col]).sort(self.ts_col)
        ts_min = df_time.select(pl.col(self.ts_col).min()).item()
        ts_max = df_time.select(pl.col(self.ts_col).max()).item()
        return pl.datetime_range(
            start=ts_min,
            end=ts_max,
            interval=block_freq,
            eager=True,
            time_unit="ns",
        ).to_list()

    def read_symbol_block(
        self,
        *,
        symbol: str,
        date: str,
        block_start: Any,
        block_end: Any,
        columns: Optional[Sequence[str]],
    ) -> pl.DataFrame:
        path = _symbol_parquet_path(
            out_root=self.aligned_root,
            date=date,
            symbol=symbol,
            subsample=self.subsample,
        )
        if not os.path.exists(path):
            return pl.DataFrame()

        lf = pl.scan_parquet(path)
        if columns is not None:
            use_cols = list(dict.fromkeys([self.ts_col, self.symbol_col, *columns]))
            lf = lf.select(use_cols)

        return (
            lf.filter((pl.col(self.ts_col) >= block_start) & (pl.col(self.ts_col) < block_end))
            .collect()
        )


@dataclass
class WindowCacheReader:
    block_root: str
    ts_col: str = "ts"
    symbol_col: str = "symbol"

    def read_curr_day(
        self,
        *,
        date: str,
        block_freq: str,
        needed_cols: Sequence[str],
    ) -> pl.DataFrame:
        return load_block_cache(
            out_root=self.block_root,
            date=date,
            block_freq=block_freq,
            columns=needed_cols,
            lazy=True,
        ).sort([self.ts_col, self.symbol_col])

    def read_history_blocks(
        self,
        *,
        date: str,
        block_freq: str,
        train_window: str,
        needed_cols: Sequence[str],
        current_day_start: Any,
    ) -> tuple[pl.DataFrame, Any, bool]:
        """
        加载所有历史数据 blocks, 直到 train_window 覆盖的历史起点为止。
        """
        
        # 注意这里train_window 的单位是天，所以我们往前移动对应的天数作为预热
        history_start = _offset_ts(current_day_start, f"-{train_window}")
        current_date = datetime.strptime(date, "%Y-%m-%d").date()
        scan_date = history_start.date()

        # 开始收集所有历史数据
        history_parts: list[pl.DataFrame] = []
        # 从 start_date 往前扫描，直到 current_date，找出所有可能包含历史数据的日期
        while scan_date < current_date:
            day_str = scan_date.strftime("%Y-%m-%d")
            try:
                day_blocks = load_block_cache(
                    out_root=self.block_root,
                    date=day_str,
                    block_freq=block_freq,
                    columns=needed_cols,
                    lazy=True,
                ).sort([self.ts_col, self.symbol_col])
            except FileNotFoundError:
                scan_date += timedelta(days=1)
                continue

            if not day_blocks.is_empty():
                history_parts.append(day_blocks)
            scan_date += timedelta(days=1)

        if not history_parts:
            return pl.DataFrame(), history_start, False

        # 收集完成，把它们拼起来，过滤出需要的时间范围
        history_blocks = (
            pl.concat(history_parts, how="vertical")
            .filter(
                (pl.col(self.ts_col) >= history_start) &
                (pl.col(self.ts_col) < current_day_start)
            )
            .sort([self.ts_col, self.symbol_col])
        )
        if history_blocks.is_empty():
            return history_blocks, history_start, False

        # 判断是否有完整的历史数据
        actual_history_start = history_blocks.select(pl.col(self.ts_col).min()).item()
        has_full_history = actual_history_start <= history_start
        return history_blocks, history_start, has_full_history

    def read_all_blocks(
        self,
        *,
        date: str,
        block_freq: str,
        train_window: str,
        feature_cols: Sequence[str],
        base_cols: Sequence[str],
    ) -> tuple[pl.DataFrame, Any, Any, bool]:
        needed_cols = list(dict.fromkeys([self.ts_col, self.symbol_col, *feature_cols, *base_cols]))

        current_blocks = self.read_curr_day(
            date=date,
            block_freq=block_freq,
            needed_cols=needed_cols,
        )
        if current_blocks.is_empty():
            return current_blocks, None, None, False

        current_day_start = current_blocks.select(pl.col(self.ts_col).min()).item()
        current_day_end = current_blocks.select(pl.col(self.ts_col).max()).item()

        history_blocks, history_start, has_full_history = self.read_history_blocks(
            date=date,
            block_freq=block_freq,
            train_window=train_window,
            needed_cols=needed_cols,
            current_day_start=current_day_start,
        )
        if history_blocks.is_empty():
            return current_blocks, current_day_start, current_day_end, False

        all_blocks = pl.concat([history_blocks, current_blocks], how="vertical").sort(
            [self.ts_col, self.symbol_col]
        )
        return all_blocks, current_day_start, current_day_end, has_full_history


@dataclass
class NeighborInputReader:
    cache_root: str
    config: RollingNeighborConfig

    def list_window_dirs(self, date: str) -> list[str]:
        window_root = _window_cache_dir(
            out_root=self.cache_root,
            date=date,
            train_window=self.config.train_window,
            step=self.config.step,
            train_subsample=self.config.train_subsample,
            query_subsample=self.config.query_subsample,
        )
        if not os.path.exists(window_root):
            raise FileNotFoundError(f"window cache root not found: {window_root}")
        return [window_root]

    def read_window(self, window_dir: str) -> dict[str, Any]:
        base_query_path = os.path.join(window_dir, "base_query.parquet")
        if not os.path.exists(base_query_path):
            return {}

        base_query = pl.read_parquet(base_query_path).sort(
            [self.config.ts_col, self.config.symbol_col]
        )
        if base_query.is_empty():
            return {}

        feature_frames: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}
        for feature_col in self.config.feature_cols:
            train_path = os.path.join(window_dir, f"{feature_col}_train.parquet")
            query_path = os.path.join(window_dir, f"{feature_col}_query.parquet")
            if (not os.path.exists(train_path)) or (not os.path.exists(query_path)):
                continue

            train_df = pl.read_parquet(train_path).sort([self.config.ts_col, self.config.symbol_col])
            query_df = pl.read_parquet(query_path).sort([self.config.ts_col, self.config.symbol_col])
            if train_df.is_empty() or query_df.is_empty():
                continue

            feature_frames[feature_col] = (train_df, query_df)

        return {
            "base_query": base_query,
            "feature_frames": feature_frames,
        }


@dataclass
class ReadyBlockReader:
    cfg: object

    def list_neighbor_files(self) -> list[str]:
        return list_neighbor_files(self.cfg.neighbor_dir)

    def read_neighbor_block(self, path: str) -> pl.DataFrame:
        return pl.read_parquet(path)

    def read_base_slice(self, neighbor_df: pl.DataFrame) -> pl.DataFrame:
        if neighbor_df.is_empty():
            return neighbor_df

        ts_min = neighbor_df.select(pl.col(self.cfg.ts_col).min()).item()
        ts_max = neighbor_df.select(pl.col(self.cfg.ts_col).max()).item()
        symbols = neighbor_df.get_column(self.cfg.symbol_col).unique().sort().to_list()

        return _load_aligned_slice_for_symbols(
            aligned_root=self.cfg.aligned_root,
            date=self.cfg.date,
            symbols=symbols,
            ts_min=ts_min,
            ts_max=ts_max,
            subsample=self.cfg.aligned_subsample,
            ts_col=self.cfg.ts_col,
            symbol_col=self.cfg.symbol_col,
            base_feature_cols=self.cfg.base_feature_cols,
        )


@dataclass
class RollingDatasetReader:
    cfg: object

    def _ready_block_root(self) -> str:
        return os.path.dirname(self.cfg.ready_block_dir)

    def list_ready_files_grouped_by_date(self) -> list[tuple[str, list[str]]]:
        root = self._ready_block_root()
        if not os.path.exists(root):
            raise FileNotFoundError(f"ready block root not found: {root}")

        groups: list[tuple[str, list[str]]] = []
        end_date = datetime.strptime(self.cfg.date, "%Y-%m-%d").date()
        for entry in sorted(os.listdir(root)):
            day_dir = os.path.join(root, entry)
            if not os.path.isdir(day_dir):
                continue

            try:
                day = datetime.strptime(entry, "%Y-%m-%d").date()
            except ValueError:
                continue

            if day > end_date:
                continue

            try:
                files = list_ready_block_files(day_dir)
            except FileNotFoundError:
                continue

            if files:
                groups.append((entry, files))

        if not groups:
            raise FileNotFoundError(f"No ready parquet day directories found under: {root}")
        return groups

    def list_curr_day_ready_files(self) -> list[str]:
        return list_ready_block_files(self.cfg.ready_block_dir)

    def list_prev_day_tail_ready_files(self) -> list[str]:
        prev_date = (datetime.strptime(self.cfg.date, "%Y-%m-%d") - timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        prev_ready_dir = os.path.join(os.path.dirname(self.cfg.ready_block_dir), prev_date)
        try:
            prev_ready_files = list_ready_block_files(prev_ready_dir)
        except FileNotFoundError:
            return []

        return prev_ready_files[-self.cfg.train_blocks :]

    def list_ready_files(self) -> tuple[list[str], int, int]:
        curr_ready_files = self.list_curr_day_ready_files()
        prev_tail_files = self.list_prev_day_tail_ready_files()

        if not prev_tail_files:
            return curr_ready_files, 0, len(curr_ready_files)

        return [*prev_tail_files, *curr_ready_files], len(prev_tail_files), len(curr_ready_files)

    def get_symbol_mapping(self, ready_files: Sequence[str]) -> dict[str, int]:
        return get_global_symbol_mapping_from_ready_blocks(
            ready_files=ready_files,
            symbol_col=self.cfg.symbol_col,
        )

    def load_prepared_block(self, path: str, symbol_mapping: dict[str, int]) -> pl.DataFrame:
        df = pl.read_parquet(path)
        df = add_symbol_code_single_df(df, self.cfg.symbol_col, symbol_mapping)
        return _prepare_raw_block_df(df, self.cfg)


@dataclass
class TrainFromNPZReader:
    cfg: object

    def resolve_data_root(self, state: object) -> str:
        data_root = self.cfg.data_root or state.get("training_out_dir")
        if not data_root:
            raise ValueError("train-from-npz data_root is not configured")
        return data_root

    def resolve_freqs(self, state: object, data_root: str) -> list[str]:
        if self.cfg.freqs is not None:
            return list(self.cfg.freqs)

        state_freqs = state.get("train_downsample_freqs")
        if state_freqs:
            return list(state_freqs)

        freq_dirs = sorted(glob.glob(os.path.join(data_root, "freq_*")))
        return [os.path.basename(path).removeprefix("freq_") for path in freq_dirs]

    def list_train_npz_files(self, *, data_root: str, freq: str) -> list[str]:
        train_dir = os.path.join(data_root, f"freq_{freq}", "train")
        files = sorted(glob.glob(os.path.join(train_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No npz files found in: {train_dir}")
        return files

    def load_npz(self, path: str) -> dict[str, np.ndarray]:
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    def list_block_npz_files(self, *, out_dir: str, kind: str) -> list[str]:
        return sorted(glob.glob(os.path.join(out_dir, kind, "*.npz")))

    def load_prediction_block(self, path: str) -> pl.DataFrame:
        payload = self.load_npz(path)
        return pl.DataFrame(
            {
                "ts": payload["ts"],
                "symbol": payload["symbol"],
                "symbol_code": payload["symbol_code"],
                "y_true": payload["y_true"],
                "y_pred": payload["y_pred"],
                "block_name": [payload["block_name"].tolist()[0]] * len(payload["y_true"]),
                "pred_file": [payload["pred_file"].tolist()[0]] * len(payload["y_true"]),
            }
        )

    def load_metric_block(self, path: str) -> pl.DataFrame:
        payload = self.load_npz(path)
        return pl.DataFrame(
            [
                {
                    "block_name": payload["block_name"].tolist()[0],
                    "pred_file": payload["pred_file"].tolist()[0],
                    "n_train": int(payload["n_train"][0]),
                    "n_pred": int(payload["n_pred"][0]),
                    "rmse": float(payload["rmse"][0]),
                    "mae": float(payload["mae"][0]),
                    "corr": float(payload["corr"][0]),
                    "qlike": float(payload["qlike"][0]),
                }
            ]
        )

    def load_importance_block(self, path: str) -> pl.DataFrame:
        payload = self.load_npz(path)
        feature_cols = payload["feature_cols"].tolist()
        block_name = payload["block_name"].tolist()[0]
        pred_file = payload["pred_file"].tolist()[0]
        return pl.DataFrame(
            {
                "feature": feature_cols,
                "importance_gain": payload["importance_gain"].tolist(),
                "importance_split": payload["importance_split"].tolist(),
                "block_name": [block_name] * len(feature_cols),
                "pred_file": [pred_file] * len(feature_cols),
            }
        )

    def load_run_config(self, *, data_root: str, freq: str) -> dict[str, Any]:
        path = os.path.join(data_root, f"freq_{freq}", "meta", "run_config.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def infer_neighbor_feature_cols(
    *,
    output_root: str,
    date: str,
    config: RollingNeighborConfig,
) -> list[str]:
    neighbor_dir = os.path.join(
        output_root,
        (
            f"neighbor_tw_{config.train_window}"
            f"_step_{config.step}"
            f"_train_{config.train_subsample}"
            f"_query_{config.query_subsample}"
        ),
        date,
    )
    files = sorted(glob.glob(os.path.join(neighbor_dir, "*.parquet")))
    if not files:
        return []

    lf = pl.scan_parquet(files[0])
    cols = list(lf.collect_schema().names())
    exclude = {config.ts_col, config.symbol_col}
    return [c for c in cols if c not in exclude]
