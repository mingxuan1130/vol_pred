from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeatureManifest:
    """
    记录当前 pipeline 各阶段产出的特征集合。
    """

    target_col: str = ""
    base_feature_cols: list[str] = field(default_factory=list)
    derived_feature_groups: dict[str, list[str]] = field(default_factory=dict)
    ready_feature_cols: list[str] = field(default_factory=list)

    def set_base_features(self, cols: list[str]) -> None:
        self.base_feature_cols = list(cols)
        self._refresh_ready_features()

    def add_derived_group(self, name: str, cols: list[str]) -> None:
        self.derived_feature_groups[name] = list(cols)
        self._refresh_ready_features()

    def _refresh_ready_features(self) -> None:
        ready: list[str] = []
        seen: set[str] = set()

        for col in self.base_feature_cols:
            if col not in seen and col != self.target_col:
                ready.append(col)
                seen.add(col)

        for cols in self.derived_feature_groups.values():
            for col in cols:
                if col not in seen and col != self.target_col:
                    ready.append(col)
                    seen.add(col)

        self.ready_feature_cols = ready
