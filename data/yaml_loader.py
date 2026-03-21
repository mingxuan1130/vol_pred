from pathlib import Path
import re

from typing import Sequence, Union
    
def load_symbols_from_list_yaml(list_yaml_path: Union[str, Path]) -> list[str]:
    """
    从 list.yaml 读取币种（忽略注释行和行尾注释），返回去重且保序的 symbol 列表。
    """
    path = Path(list_yaml_path)
    symbols: list[str] = []
    seen = set()

    pattern = re.compile(r"^\s*-\s*([A-Z0-9]+USDT)\b")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        m = pattern.match(raw_line)
        if not m:
            continue

        sym = m.group(1)
        if sym not in seen:
            seen.add(sym)
            symbols.append(sym)

    return symbols


def chunk_symbols(symbols: Sequence[str], group_size: int = 15) -> list[list[str]]:
    """
    把 symbols 按 group_size 分组。
    """
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    return [list(symbols[i:i + group_size]) for i in range(0, len(symbols), group_size)]


def build_symbol_groups(
    list_yaml_path: Union[str, Path, None] = None,
    group_size: int = 15,
) -> list[list[str]]:
    """
    读取 list.yaml 并按 group_size 分组，默认 15 个一组。
    """
    if list_yaml_path is None:
        # Notebook-safe: use current working directory
        list_yaml_path = Path.cwd() / "list.yaml"

    symbols = load_symbols_from_list_yaml(list_yaml_path)
    return chunk_symbols(symbols, group_size=group_size)