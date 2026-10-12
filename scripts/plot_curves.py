"""训练曲线可视化:读取 MetricsLogger 产出的 CSV。

- 若安装了 matplotlib:绘制 PNG 曲线图;
- 否则:回退到终端 ASCII 折线图(零依赖)。

用法:
    python scripts/plot_curves.py runs/alphazero.csv
    python scripts/plot_curves.py runs/alphazero.csv --metric loss --out runs/loss.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import _bootstrap  # noqa: F401

from utils.metrics import sparkline


def read_csv(path: str):
    if not os.path.exists(path):
        sys.exit(f"[plot] 找不到文件: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("[plot] CSV 为空")
    step_key = "step" if "step" in rows[0] else None
    cols = [c for c in rows[0] if c != step_key]
    data = {c: [float(r[c]) for r in rows if r.get(c, "") != ""] for c in cols}
    steps = [float(r["step"]) for r in rows] if step_key else list(range(len(rows)))
    return steps, data


def try_matplotlib(steps, data, metrics, out):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in metrics:
        ax.plot(steps, data[m], label=m)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[plot] 已保存曲线图 -> {out}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="指标 CSV 路径")
    parser.add_argument("--metric", action="append", default=None, help="仅绘制指定指标(可多次)")
    parser.add_argument("--out", default=None, help="输出 PNG 路径")
    args = parser.parse_args()

    steps, data = read_csv(args.csv)
    metrics = args.metric or list(data.keys())
    metrics = [m for m in metrics if m in data]

    out = args.out or (os.path.splitext(args.csv)[0] + ".png")
    if try_matplotlib(steps, data, metrics, out):
        return

    print("[plot] 未检测到 matplotlib,回退到 ASCII 折线图:\n")
    for m in metrics:
        s = data[m]
        print(f"{m:>16}  {s[0]:>8.3f} -> {s[-1]:<8.3f}  {sparkline(s)}")


if __name__ == "__main__":
    main()
