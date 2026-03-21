from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from data.massive_data_adapter import (
    _block_cache_dir,
    _block_file_name,
    _symbol_parquet_path,
    _window_cache_dir,
)
from factor_phase_II.build_rolling_npz import _freq_output_dirs, _np_savez, save_json


@dataclass
class AlignedParquetSink:
    compression: str = "zstd"
    compression_level: int = 3

    def write(
        self,
        df_symbol: pl.DataFrame,
        *,
        out_root: str,
        date: str,
        symbol: str,
        subsample: str,
        overwrite: bool,
    ) -> str:
        out_path = _symbol_parquet_path(
            out_root=out_root,
            date=date,
            symbol=symbol,
            subsample=subsample,
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if (not overwrite) and os.path.exists(out_path):
            return out_path

        df_symbol.write_parquet(
            out_path,
            compression=self.compression,
            compression_level=self.compression_level,
        )
        return out_path


@dataclass
class BlockCacheSink:
    compression: str = "zstd"
    compression_level: int = 3

    def write(
        self,
        *,
        out_root: str,
        date: str,
        block_freq: str,
        block_start: Any,
        df_block: pl.DataFrame,
        overwrite: bool,
    ) -> str | None:
        block_dir = _block_cache_dir(out_root=out_root, date=date, block_freq=block_freq)
        os.makedirs(block_dir, exist_ok=True)

        out_path = os.path.join(block_dir, _block_file_name(block_start))
        if (not overwrite) and os.path.exists(out_path):
            return out_path
        if df_block.is_empty():
            return None

        df_block.write_parquet(
            out_path,
            compression=self.compression,
            compression_level=self.compression_level,
        )
        print(f"[OK] block cache -> {out_path}")
        return out_path


@dataclass
class WindowCacheSink:
    compression: str = "zstd"
    compression_level: int = 3

    def write(
        self,
        *,
        out_root: str,
        date: str,
        train_window: str,
        step: str,
        train_subsample: str,
        query_subsample: str,
        payload: dict[str, pl.DataFrame],
        overwrite: bool,
    ) -> str | None:
        if not payload or "base_query" not in payload:
            return None

        window_root = _window_cache_dir(
            out_root=out_root,
            date=date,
            train_window=train_window,
            step=step,
            train_subsample=train_subsample,
            query_subsample=query_subsample,
        )
        os.makedirs(window_root, exist_ok=True)

        probe_key = next((k for k in payload if k.endswith("_train")), None)
        probe_path = os.path.join(window_root, probe_key + ".parquet") if probe_key else None
        base_query_path = os.path.join(window_root, "base_query.parquet")

        if (
            (not overwrite) and
            os.path.exists(base_query_path) and
            probe_path is not None and
            os.path.exists(probe_path)
        ):
            return window_root

        for name, df_piece in payload.items():
            df_piece.write_parquet(
                os.path.join(window_root, f"{name}.parquet"),
                compression=self.compression,
                compression_level=self.compression_level,
            )

        print(f"[OK] window cache -> {window_root}")
        return window_root


@dataclass
class NeighborFeatureSink:
    compression: str = "zstd"
    compression_level: int = 3

    def write(
        self,
        *,
        output_dir: str,
        cutoff_name: str,
        enriched: pl.DataFrame,
        overwrite: bool,
    ) -> str | None:
        if enriched.is_empty():
            return None

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{cutoff_name}.parquet")
        if (not overwrite) and os.path.exists(out_path):
            return out_path

        enriched.write_parquet(
            out_path,
            compression=self.compression,
            compression_level=self.compression_level,
        )
        return out_path


@dataclass
class ReadyBlockSink:
    cfg: object

    def write_meta(self, num_neighbor_blocks: int) -> None:
        meta_dir = os.path.join(self.cfg.ready_block_dir, "_meta")
        os.makedirs(meta_dir, exist_ok=True)

        save_json(
            {
                "saved_at": datetime.now().isoformat(),
                "date": self.cfg.date,
                "neighbor_dir": self.cfg.neighbor_dir,
                "aligned_root": self.cfg.aligned_root,
                "ready_block_dir": self.cfg.ready_block_dir,
                "aligned_subsample": self.cfg.aligned_subsample,
                "ts_col": self.cfg.ts_col,
                "symbol_col": self.cfg.symbol_col,
                "target_col": self.cfg.target_col,
                "base_feature_cols": list(self.cfg.base_feature_cols),
                "num_neighbor_blocks": num_neighbor_blocks,
            },
            os.path.join(meta_dir, "ready_block_config.json"),
        )

    def write_block(self, block_name: str, ready_df: pl.DataFrame) -> bool:
        os.makedirs(self.cfg.ready_block_dir, exist_ok=True)
        out_path = os.path.join(self.cfg.ready_block_dir, f"{block_name}.parquet")

        if os.path.exists(out_path):
            print(f"[READY][SKIP] {block_name}")
            return False

        ready_df.write_parquet(
            out_path,
            compression=self.cfg.ready_block_compression,
        )
        return True


@dataclass
class RollingDatasetSink:
    cfg: object

    def write_run_config(
        self,
        *,
        freq: str,
        symbol_mapping: dict[str, int],
        num_total_ready_blocks: int,
        num_total_ready_days: int | None = None,
    ) -> None:
        _, _, meta_dir = _freq_output_dirs(self.cfg.out_dir, freq)
        save_json(
            {
                "saved_at": datetime.now().isoformat(),
                "date": self.cfg.date,
                "neighbor_dir": self.cfg.neighbor_dir,
                "ready_block_dir": self.cfg.ready_block_dir,
                "out_dir": os.path.join(self.cfg.out_dir, f"freq_{freq}"),
                "train_blocks": self.cfg.train_blocks,
                "aligned_subsample": self.cfg.aligned_subsample,
                "target_col": self.cfg.target_col,
                "ts_col": self.cfg.ts_col,
                "symbol_col": self.cfg.symbol_col,
                "base_feature_cols": list(self.cfg.base_feature_cols),
                "num_total_ready_blocks": num_total_ready_blocks,
                "num_total_ready_days": num_total_ready_days,
                "compress_npz": self.cfg.compress_npz,
                "global_symbol_mapping": symbol_mapping,
                "train_downsample_freq": freq,
                "multi_agg_feature_cols": list(self.cfg.multi_agg_feature_cols),
                "dataset_window_mode": getattr(self.cfg, "dataset_window_mode", "by_block"),
                "train_days": getattr(self.cfg, "train_days", None),
                "validation_days": getattr(self.cfg, "validation_days", None),
                "test_days": getattr(self.cfg, "test_days", None),
                "step_days": getattr(self.cfg, "step_days", None),
            },
            os.path.join(meta_dir, "run_config.json"),
        )

    def write_block_outputs(
        self,
        *,
        freq: str,
        block_name: str,
        payload: dict[str, Any],
        train_files: list[str],
        pred_file: str,
        symbol_mapping: dict[str, int],
        validation_payload: dict[str, Any] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        train_dir, pred_dir, meta_dir = _freq_output_dirs(self.cfg.out_dir, freq)
        train_npz_path = os.path.join(train_dir, f"{block_name}.npz")

        npz_payload: dict[str, Any] = {
            "X_train": payload["X_train"],
            "y_train": payload["y_train"],
            "X_pred": payload["X_pred"],
            "y_pred_true": payload["y_pred_true"],
            "ts_pred": payload["ts_pred"],
            "symbol_pred": payload["symbol_pred"],
            "symbol_code_pred": payload["symbol_code_pred"],
            "feature_cols": payload["feature_cols"],
            "train_files": train_files,
            "pred_file": [pred_file],
            "extra_meta_json": [
                '{"freq": "%s", "symbol_mapping_size": %d}' % (
                    freq,
                    len(symbol_mapping),
                )
            ],
        }
        if validation_payload:
            npz_payload.update(
                {
                    "X_val": validation_payload["X_pred"],
                    "y_val_true": validation_payload["y_pred_true"],
                    "ts_val": validation_payload["ts_pred"],
                    "symbol_val": validation_payload["symbol_pred"],
                    "symbol_code_val": validation_payload["symbol_code_pred"],
                }
            )

        _np_savez(
            train_npz_path,
            compress=self.cfg.compress_npz,
            **npz_payload,
        )

        _np_savez(
            os.path.join(pred_dir, f"{block_name}.npz"),
            compress=False,
            block_name=[block_name],
            train_npz_path=[train_npz_path],
            freq=[freq],
        )

        meta_payload = {
            "block_name": block_name,
            "freq": freq,
            "train_files": train_files,
            "pred_file": pred_file,
            "n_train": int(len(payload["y_train"])),
            "n_pred": int(len(payload["y_pred_true"])),
            "num_features": len(payload["feature_cols"]),
            "feature_cols": payload["feature_cols"],
            "multi_agg_feature_cols": list(self.cfg.multi_agg_feature_cols),
        }
        if validation_payload:
            meta_payload.update(
                {
                    "n_val": int(len(validation_payload["y_pred_true"])),
                }
            )
        if extra_meta:
            meta_payload.update(extra_meta)

        save_json(meta_payload, os.path.join(meta_dir, f"{block_name}.json"))


@dataclass
class TrainFromNPZSink:
    cfg: object

    def freq_out_dir(self, freq: str) -> str:
        return os.path.join(self.cfg.out_root, f"freq_{freq}")

    def prepare_freq_dirs(self, freq: str) -> dict[str, str]:
        out_dir = self.freq_out_dir(freq)
        paths = {
            "out_dir": out_dir,
            "pred_blocks_dir": os.path.join(out_dir, "pred_blocks"),
            "metric_blocks_dir": os.path.join(out_dir, "metric_blocks"),
            "importance_blocks_dir": os.path.join(out_dir, "importance_blocks"),
            "meta_dir": os.path.join(out_dir, "meta"),
        }
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
        return paths

    def block_prediction_path(self, *, freq: str, block_name: str) -> str:
        return os.path.join(self.freq_out_dir(freq), "pred_blocks", f"{block_name}.npz")

    def write_block_outputs(self, *, freq: str, built: dict[str, Any]) -> None:
        dirs = self.prepare_freq_dirs(freq)
        block_name = built["block_name"]
        pred_file = built["pred_file"]

        np.savez_compressed(
            os.path.join(dirs["pred_blocks_dir"], f"{block_name}.npz"),
            ts=built["ts_pred"],
            symbol=built["symbol_pred"],
            symbol_code=built["symbol_code_pred"],
            y_true=built["y_true"],
            y_pred=built["y_pred"],
            block_name=np.array([block_name], dtype=object),
            pred_file=np.array([pred_file], dtype=object),
        )

        np.savez_compressed(
            os.path.join(dirs["metric_blocks_dir"], f"{block_name}.npz"),
            block_name=np.array([block_name], dtype=object),
            pred_file=np.array([pred_file], dtype=object),
            n_train=np.array([built["n_train"]], dtype=np.int64),
            n_pred=np.array([built["n_pred"]], dtype=np.int64),
            rmse=np.array([built["metrics"]["rmse"]], dtype=np.float64),
            mae=np.array([built["metrics"]["mae"]], dtype=np.float64),
            corr=np.array([built["metrics"]["corr"]], dtype=np.float64),
            qlike=np.array([built["metrics"]["qlike"]], dtype=np.float64),
        )

        np.savez_compressed(
            os.path.join(dirs["importance_blocks_dir"], f"{block_name}.npz"),
            feature_cols=np.array(built["feature_cols"], dtype=object),
            importance_gain=built["importance_gain"],
            importance_split=built["importance_split"],
            block_name=np.array([block_name], dtype=object),
            pred_file=np.array([pred_file], dtype=object),
        )

        save_json(
            {
                "block_name": block_name,
                "train_files": built["train_files"],
                "pred_file": pred_file,
                "n_train": built["n_train"],
                "n_pred": built["n_pred"],
                "feature_cols": built["feature_cols"],
                "use_log_target": self.cfg.use_log_target,
                "lgb_params": self.cfg.resolved_lgb_params(),
            },
            os.path.join(dirs["meta_dir"], f"{block_name}.json"),
        )

    def write_freq_summary(
        self,
        *,
        freq: str,
        pred_df: pl.DataFrame,
        metric_df: pl.DataFrame,
        overall_df: pl.DataFrame,
        importance_df: pl.DataFrame,
        importance_mean_df: pl.DataFrame,
        overall: dict[str, Any],
        train_run_config: dict[str, Any],
    ) -> None:
        dirs = self.prepare_freq_dirs(freq)
        pred_df.write_parquet(os.path.join(dirs["out_dir"], "rolling_predictions.parquet"))
        metric_df.write_parquet(os.path.join(dirs["out_dir"], "rolling_metrics.parquet"))
        overall_df.write_parquet(os.path.join(dirs["out_dir"], "overall_metrics.parquet"))
        importance_df.write_parquet(os.path.join(dirs["out_dir"], "feature_importance.parquet"))
        importance_mean_df.write_parquet(
            os.path.join(dirs["out_dir"], "feature_importance_mean.parquet")
        )

        save_json(
            {
                "num_blocks": int(metric_df.height),
                "use_log_target": self.cfg.use_log_target,
                "lgb_params": self.cfg.resolved_lgb_params(),
                "overall": overall,
                "train_data_config": train_run_config,
            },
            os.path.join(dirs["meta_dir"], "run_summary.json"),
        )

    def write_stage_summary(self, summary: dict[str, Any]) -> None:
        save_json(summary, os.path.join(self.cfg.out_root, "meta", "train_stage_summary.json"))
