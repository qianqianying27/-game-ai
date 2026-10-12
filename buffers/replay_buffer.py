from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np


class ReplayBuffer:
    """DQN 经验回放缓冲区(环形数组实现,支持任意形状观测)。"""

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []
        self.pos = 0
        self._rng = random.Random(seed)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        item = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "terminated": terminated,
            "mask": mask,
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        batch = self._rng.sample(self.buffer, min(batch_size, len(self.buffer)))
        out = {
            "obs": np.stack([b["obs"] for b in batch]),
            "action": np.array([b["action"] for b in batch], dtype=np.int64),
            "reward": np.array([b["reward"] for b in batch], dtype=np.float32),
            "next_obs": np.stack([b["next_obs"] for b in batch]),
            "terminated": np.array([b["terminated"] for b in batch], dtype=np.float32),
        }
        if batch[0]["mask"] is not None:
            out["mask"] = np.stack([b["mask"] for b in batch])
        return out

    def __len__(self) -> int:
        return len(self.buffer)
