from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class Env(ABC):
    """类 Gym 的自包含环境接口,避免外部依赖。

    约定:
    - ``reset`` 返回初始观测。
    - ``step`` 返回 ``(obs, reward, terminated, truncated, info)``。
    """

    #: 观测形状,例如 (n,) 或 (C, H, W)
    observation_shape: Tuple[int, ...]
    #: 离散动作数量
    action_dim: int

    @abstractmethod
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        ...

    @abstractmethod
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        ...

    def render(self) -> str:  # pragma: no cover - 便于调试
        return repr(self)

    @property
    def is_single_agent(self) -> bool:
        return True
