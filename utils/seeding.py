from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """统一设置随机种子,保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch 为可选依赖(测试环境可能未安装)
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
