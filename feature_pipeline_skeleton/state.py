from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from feature_pipeline_skeleton.manifest import FeatureManifest


@dataclass
class PipelineState:
    """
    阶段之间只传路径、配置和列信息，不传大 DataFrame。
    """

    date: str
    symbols: list[str]
    artifacts: dict[str, Any] = field(default_factory=dict)
    manifest: FeatureManifest = field(default_factory=FeatureManifest)

    def put(self, key: str, value: Any) -> None:
        self.artifacts[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.artifacts.get(key, default)
