from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class Agent(ABC):
    """所有游戏 AI 智能体的统一接口。"""

    @abstractmethod
    def act(self, obs: np.ndarray, **kwargs: Any) -> int:
        ...

    def update(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, float]]:
        return None

    def save(self, path: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def load(self, path: str) -> None:  # pragma: no cover
        raise NotImplementedError
