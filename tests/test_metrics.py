from __future__ import annotations

import csv
import os

from utils.metrics import MetricsLogger, sparkline


def test_sparkline_basic():
    assert sparkline([]) == ""
    line = sparkline([0.0, 1.0, 2.0, 3.0])
    assert len(line) == 4
    # 常数序列应渲染为单一字符
    assert len(set(sparkline([5.0, 5.0, 5.0]))) == 1


def test_metrics_logger_csv(tmp_path):
    path = os.path.join(tmp_path, "m.csv")
    m = MetricsLogger(path)
    m.log(step=1, loss=0.5, acc=0.1)
    m.log(step=2, loss=0.3, acc=0.4)
    out = m.save()
    assert os.path.exists(out)
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert float(rows[1]["loss"]) == 0.3
    assert m.series("loss") == [0.5, 0.3]
