from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from feature_pipeline_skeleton.state import PipelineState


class Stage(Protocol):
    def run(self, state: PipelineState) -> PipelineState:
        ...


@dataclass
class FeaturePipeline:
    stages: list[Stage]

    def run(self, date: str, symbols: list[str]) -> PipelineState:
        total_start = perf_counter()
        state = PipelineState(date=date, symbols=symbols)
        for stage in self.stages:
            stage_name = stage.__class__.__name__
            stage_start = perf_counter()
            print(f"[PIPELINE] start {stage_name}")
            state = stage.run(state)
            stage_elapsed = perf_counter() - stage_start
            print(f"[PIPELINE] done {stage_name} elapsed={stage_elapsed:.2f}s")
        total_elapsed = perf_counter() - total_start
        print(f"[PIPELINE] finished total_elapsed={total_elapsed:.2f}s")
        return state
