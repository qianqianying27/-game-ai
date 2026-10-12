from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class RolloutBuffer:
    """PPO 用的轨迹缓冲区,内含 GAE(广义优势估计)计算。"""

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.logprobs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, value) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.capacity

    def clear(self) -> None:
        self.ptr = 0

    def compute_returns_and_advantages(self, last_value: float) -> None:
        n = self.ptr
        last_gae = 0.0
        for t in reversed(range(n)):
            non_terminal = 1.0 - self.dones[t]
            next_value = last_value if t == n - 1 else self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_value * non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get(self) -> Dict[str, np.ndarray]:
        n = self.ptr
        adv = self.advantages[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "obs": self.obs[:n],
            "actions": self.actions[:n],
            "logprobs": self.logprobs[:n],
            "advantages": adv,
            "returns": self.returns[:n],
        }
