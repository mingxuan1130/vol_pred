"""src.pipeline — 数据 pipeline 各阶段模块。"""

from importlib import import_module

__all__ = [
    "build_aligned_snapshot_vol",
    "build_aligned_multi_symbol",
    "construct_factors_from_snapshot",
    "load_aligned_data",
    "load_snapshot_raw",
    "load_and_align_volatility",
]

_EXPORT_MAP = {
    "build_aligned_multi_symbol": (".vol_adapter", "build_aligned_multi_symbol"),
    "build_aligned_snapshot_vol": (".vol_adapter", "build_aligned_snapshot_vol"),
    "construct_factors_from_snapshot": (".vol_adapter", "construct_factors_from_snapshot"),
    "load_aligned_data": (".warper", "load_aligned_data"),
    "load_snapshot_raw": (".raw_loaders", "load_snapshot_raw"),
    "load_and_align_volatility": (".raw_loaders", "load_and_align_volatility"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value
