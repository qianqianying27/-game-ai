"""脚本路径引导:允许在未 `pip install -e .` 的情况下直接运行。

将项目根目录(game-ai)加入 sys.path,使 ``envs / agents / models`` 等
顶层包可被直接导入。
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
