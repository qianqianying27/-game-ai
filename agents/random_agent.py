from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import Agent


class RandomAgent(Agent):
    """基线:在合法动作中均匀随机选择。"""

    def __init__(self, action_dim: int, seed: Optional[int] = None) -> None:
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, *, legal_actions: Optional[np.ndarray] = None, **kwargs: Any) -> int:
        if legal_actions is not None and len(legal_actions) > 0:
            return int(self._rng.choice(legal_actions))
        return int(self._rng.integers(self.action_dim))
