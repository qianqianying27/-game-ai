from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

# 由低到高的方块字符,用于无依赖的 ASCII 折线(sparkline)
_SPARK = "▁▂▃▅▆▇█"


def sparkline(values: List[float], width: int = 48) -> str:
    """把一串数值渲染成一行 ASCII 折线图(零依赖可视化)。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    if len(vals) > width:  # 等间隔降采样
        step = len(vals) / width
        vals = [vals[int(i * step)] for i in range(width)]
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng == 0:
        return _SPARK[len(_SPARK) // 2] * len(vals)
    out = []
    for v in vals:
        idx = int((v - lo) / rng * (len(_SPARK) - 1))
        out.append(_SPARK[idx])
    return "".join(out)


class MetricsLogger:
    """轻量训练指标记录器:内存累积 -> CSV,并提供 ASCII 可视化。

    不依赖 matplotlib,任何环境都能记录与快速查看趋势。
    """

    def __init__(self, path: str, step_key: str = "step") -> None:
        self.path = path
        self.step_key = step_key
        self.rows: List[Dict[str, float]] = []

    def log(self, step: Optional[int] = None, **metrics: float) -> None:
        row: Dict[str, float] = {}
        if step is not None:
            row[self.step_key] = step
        row.update({k: float(v) for k, v in metrics.items()})
        self.rows.append(row)

    def _columns(self) -> List[str]:
        cols: List[str] = []
        if self.rows and self.step_key in self.rows[0]:
            cols.append(self.step_key)
        for r in self.rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        return cols

    def save(self) -> str:
        if not self.rows:
            return self.path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        cols = self._columns()
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in self.rows:
                writer.writerow({c: r.get(c, "") for c in cols})
        return self.path

    def series(self, key: str) -> List[float]:
        return [r[key] for r in self.rows if key in r]

    def print_summary(self, keys: Optional[List[str]] = None) -> None:
        """在终端打印每个指标的 ASCII 趋势与首尾值。"""
        keys = keys or [k for k in self._columns() if k != self.step_key]
        for k in keys:
            s = self.series(k)
            if not s:
                continue
            print(f"{k:>16}  {s[0]:>8.3f} -> {s[-1]:<8.3f}  {sparkline(s)}")
