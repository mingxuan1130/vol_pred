from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import lightgbm as lgb
import numpy as np
import polars as pl

from analysis.evals import evaluate
from data.massive_data_adapter import _offset_ts
from data.utils import trim_first_15min
from data.vol_adapter import _ensure_datetime_ns, add_vol_as_features, add_vol_as_labels
from factor_phase_I.snapshot import add_factors, construct_factors_from_snapshot
from factor_phase_II.build_rolling_npz import (
    _build_pred_rolling_features,
    _downsample_train_df,
    make_xy,
)
from neighbor.build_neighbors import make_symbol_nn_feature_oos, make_time_nn_feature_oos
from neighbor.window_cache_pipeline import RollingNeighborConfig

from feature_pipeline_skeleton.readers import BlockSliceReader, WindowCacheReader
from feature_pipeline_skeleton.sinks import (
    AlignedParquetSink,
    BlockCacheSink,
    WindowCacheSink,
)

@dataclass
class AlignedDataBuilder:
    vol_data_dir: str
    snapshot_root: str
    snapshot_feats: list[str]
    vol_feats: list[str]
    vol_labels: list[str]
    subsample: str = "20ms"
    on_error: str = "skip"
    overwrite: bool = True
    max_workers: Optional[int] = None
    drop_leading_15min: bool = True
    sink: AlignedParquetSink = field(default_factory=AlignedParquetSink)

    def build(self, symbol: str, raw_inputs: dict[str, pl.DataFrame]) -> pl.DataFrame:
        snapshot_df = raw_inputs["snapshot_df"]
        vol_df = raw_inputs["vol_df"]

        time_start = vol_df.select(pl.col("ts_grid").min()).item()
        time_end = vol_df.select(pl.col("ts_grid").max()).item()
        snap_df = _ensure_datetime_ns(snapshot_df, "ts").filter(
            (pl.col("ts") >= pl.lit(time_start)) & (pl.col("ts") <= pl.lit(time_end))
        )

        feat_df = construct_factors_from_snapshot(snap_df)
        if "ts" not in feat_df.columns:
            if feat_df.height != snap_df.height:
                raise ValueError(f"factor constructor result has no ts: {symbol}")
            feat_df = feat_df.with_columns(snap_df["ts"])

        df_with_vol_feats = add_vol_as_features(
            origin_df=feat_df,
            df_vol=vol_df,
            vol_feats=self.vol_feats,
        )
        df_features = add_factors(df_with_vol_feats)
        df_final = add_vol_as_labels(
            origin_df=df_features,
            df_vol=vol_df,
            vol_labels=self.vol_labels,
            grid_resolution=self.subsample,
        )

        if self.drop_leading_15min:
            df_final = trim_first_15min(df_final, time_col="ts")

        return df_final.with_columns(pl.lit(symbol).alias("symbol"))


@dataclass
class BlockCacheBuilder:
    ts_col: str = "ts"
    symbol_col: str = "symbol"

    def build(self, parts: Sequence[pl.DataFrame]) -> pl.DataFrame:
        valid_parts = [part for part in parts if not part.is_empty()]
        if not valid_parts:
            return pl.DataFrame()
        return pl.concat(valid_parts, how="vertical").sort([self.ts_col, self.symbol_col])


@dataclass
class WindowCacheBuilder:
    ts_col: str = "ts"
    symbol_col: str = "symbol"

    def build_for_day(
        self,
        *,
        all_blocks: pl.DataFrame,
        symbols: Sequence[str],
        train_window: str,
        target_day_start: Any,
        target_day_end: Any,
        train_subsample: str,
        feature_cols: Sequence[str],
        base_cols: Sequence[str],
    ) -> dict[str, pl.DataFrame]:
        train_start = _offset_ts(target_day_start, f"-{train_window}")

        window_df = (
            all_blocks
            .filter(
                (pl.col(self.ts_col) >= train_start) &
                (pl.col(self.ts_col) <= target_day_end) &
                (pl.col(self.symbol_col).is_in(symbols))
            )
            .sort([self.ts_col, self.symbol_col])
        )
        if window_df.is_empty():
            return {}

        query_df = (
            window_df
            .filter(
                (pl.col(self.ts_col) >= target_day_start) &
                (pl.col(self.ts_col) <= target_day_end)
            )
            .sort([self.ts_col, self.symbol_col])
        )
        if query_df.is_empty():
            return {}

        train_raw = (
            window_df
            .filter(
                (pl.col(self.ts_col) >= train_start) &
                (pl.col(self.ts_col) < target_day_start)
            )
            .sort([self.ts_col, self.symbol_col])
        )
        if train_raw.is_empty():
            return {}

        base_query = query_df.select(
            [c for c in [self.ts_col, self.symbol_col, *base_cols] if c in query_df.columns]
        )
        train_down = (
            train_raw
            .group_by_dynamic(
                index_column=self.ts_col,
                every=train_subsample,
                period=train_subsample,
                group_by=self.symbol_col,
                closed="left",
                label="left",
            )
            .agg([pl.col(c).mean().alias(c) for c in feature_cols if c in train_raw.columns])
            .sort([self.ts_col, self.symbol_col])
        )

        payload: dict[str, pl.DataFrame] = {"base_query": base_query}
        for feature_col in feature_cols:
            if feature_col not in train_down.columns or feature_col not in query_df.columns:
                continue
            payload[f"{feature_col}_train"] = train_down.select([self.ts_col, self.symbol_col, feature_col])
            payload[f"{feature_col}_query"] = query_df.select([self.ts_col, self.symbol_col, feature_col])

        return payload


@dataclass
class NeighborFeatureBuilder:
    config: RollingNeighborConfig

    def build(self, cutoff_inputs: dict[str, Any]) -> pl.DataFrame:
        if not cutoff_inputs:
            return pl.DataFrame()

        enriched = cutoff_inputs["base_query"]
        join_cols = [self.config.ts_col, self.config.symbol_col]
        feature_frames: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = cutoff_inputs["feature_frames"]

        for feature_col, (train_df, query_df) in feature_frames.items():
            for n in self.config.n_list:
                if self.config.use_time_neighbors:
                    feat_df = make_time_nn_feature_oos(
                        train_df=train_df,
                        query_df=query_df,
                        feature_col=feature_col,
                        n=n,
                        metric=self.config.metric,
                        p=self.config.p,
                        ts_col=self.config.ts_col,
                        symbol_col=self.config.symbol_col,
                        base_coins=self.config.base_coins,
                    )
                    enriched = enriched.join(feat_df, on=join_cols, how="left")

                if self.config.use_symbol_neighbors:
                    feat_df = make_symbol_nn_feature_oos(
                        train_df=train_df,
                        query_df=query_df,
                        feature_col=feature_col,
                        n=n,
                        metric=self.config.metric,
                        p=self.config.p,
                        ts_col=self.config.ts_col,
                        symbol_col=self.config.symbol_col,
                        base_coins=self.config.base_coins,
                    )
                    enriched = enriched.join(feat_df, on=join_cols, how="left")

        return enriched


@dataclass
class ReadyBlockBuilder:
    cfg: object

    def build(self, neighbor_df: pl.DataFrame, base_df: pl.DataFrame) -> pl.DataFrame:
        if neighbor_df.is_empty():
            return neighbor_df

        return neighbor_df.join(
            base_df,
            on=[self.cfg.ts_col, self.cfg.symbol_col],
            how="left",
            coalesce=True,
        )


@dataclass
class TrainingDatasetBuilder:
    cfg: object

    def _resolve_symbol_code_pred(
        self,
        *,
        pred_df_freq: pl.DataFrame,
        feature_cols: Sequence[str],
    ) -> np.ndarray:
        if "symbol_code" in pred_df_freq.columns:
            return pred_df_freq.get_column("symbol_code").to_numpy()

        if "symbol_code_last" in feature_cols:
            return pred_df_freq.get_column("symbol_code_last").to_numpy()

        symbol_pred = pred_df_freq.get_column(self.cfg.symbol_col).to_list()
        unique_symbols = sorted({str(symbol) for symbol in symbol_pred})
        symbol_mapping = {symbol: idx for idx, symbol in enumerate(unique_symbols)}
        return np.asarray(
            [symbol_mapping[str(symbol)] for symbol in symbol_pred],
            dtype=np.int32,
        )

    def build_for_freq(
        self,
        *,
        train_df_raw: pl.DataFrame,
        pred_df_raw: pl.DataFrame,
        freq: str,
    ) -> dict[str, Any]:
        train_df_freq, feature_cols_train = _downsample_train_df(
            train_df_raw,
            freq=freq,
            cfg=self.cfg,
        )
        pred_df_freq, feature_cols_pred = _build_pred_rolling_features(
            pred_df_raw,
            freq=freq,
            cfg=self.cfg,
        )

        if feature_cols_train != feature_cols_pred:
            raise RuntimeError(f"feature cols mismatch at freq={freq}")
        if train_df_freq.is_empty() or pred_df_freq.is_empty():
            return {}

        X_train, y_train = make_xy(train_df_freq, feature_cols_train, self.cfg.target_col)
        X_pred, y_pred_true = make_xy(pred_df_freq, feature_cols_train, self.cfg.target_col)
        symbol_code_pred = self._resolve_symbol_code_pred(
            pred_df_freq=pred_df_freq,
            feature_cols=feature_cols_train,
        )

        return {
            "feature_cols": feature_cols_train,
            "X_train": X_train,
            "y_train": y_train,
            "X_pred": X_pred,
            "y_pred_true": y_pred_true,
            "ts_pred": pred_df_freq.get_column(self.cfg.ts_col).to_numpy(),
            "symbol_pred": pred_df_freq.get_column(self.cfg.symbol_col).to_numpy(),
            "symbol_code_pred": symbol_code_pred,
        }


@dataclass
class TrainFromNPZConfig:
    data_root: str | None = None
    out_root: str = ""
    freqs: Sequence[str] | None = None
    use_log_target: bool = True
    overwrite: bool = False
    lgb_params: dict | None = None

    def resolved_lgb_params(self) -> dict:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_data_in_leaf": 200,
            "lambda_l1": 0.0,
            "lambda_l2": 1.0,
            "num_threads": 26,
        }
        if self.lgb_params is not None:
            params.update(self.lgb_params)
        return params


@dataclass
class TrainFromNPZBuilder:
    cfg: TrainFromNPZConfig

    def _resolve_symbol_code_pred(self, payload: dict[str, Any]) -> np.ndarray:
        if "symbol_code_pred" in payload:
            return payload["symbol_code_pred"]

        feature_cols = payload["feature_cols"].tolist()
        if "symbol_code_last" in feature_cols:
            idx = feature_cols.index("symbol_code_last")
            return payload["X_pred"][:, idx].astype(np.int32, copy=False)

        symbol_pred = payload["symbol_pred"]
        unique_symbols = sorted({str(symbol) for symbol in symbol_pred.tolist()})
        symbol_mapping = {symbol: idx for idx, symbol in enumerate(unique_symbols)}
        return np.asarray(
            [symbol_mapping[str(symbol)] for symbol in symbol_pred.tolist()],
            dtype=np.int32,
        )

    def build(self, *, block_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        X_train = payload["X_train"].astype(np.float32, copy=False)
        y_train = payload["y_train"].astype(np.float32, copy=False)
        X_pred = payload["X_pred"].astype(np.float32, copy=False)
        y_true = payload["y_pred_true"].astype(np.float32, copy=False)

        ts_pred = payload["ts_pred"]
        symbol_pred = payload["symbol_pred"]
        symbol_code_pred = self._resolve_symbol_code_pred(payload)
        feature_cols = payload["feature_cols"].tolist()
        train_files = payload["train_files"].tolist()
        pred_file = payload["pred_file"].tolist()[0]

        model = lgb.LGBMRegressor(**self.cfg.resolved_lgb_params())
        model.fit(X_train, y_train)

        y_pred = model.predict(X_pred)
        metrics = evaluate(y_true=y_true, y_pred=y_pred)
        gain = model.booster_.feature_importance(importance_type="gain").astype(np.float64)
        split = model.booster_.feature_importance(importance_type="split").astype(np.int64)

        return {
            "block_name": block_name,
            "pred_file": pred_file,
            "train_files": train_files,
            "feature_cols": feature_cols,
            "n_train": int(len(y_train)),
            "n_pred": int(len(y_true)),
            "ts_pred": ts_pred,
            "symbol_pred": symbol_pred,
            "symbol_code_pred": symbol_code_pred,
            "y_true": y_true,
            "y_pred": y_pred,
            "metrics": metrics,
            "importance_gain": gain,
            "importance_split": split,
        }
