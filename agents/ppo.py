from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from buffers.rollout_buffer import RolloutBuffer
from models.networks import ActorCritic

from .base import Agent


class PPOAgent(Agent):
    """Clip-PPO(离散动作),用于单智能体任务(如 GridWorld)。"""

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_steps: int = 512,
        n_epochs: int = 10,
        minibatch: int = 64,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.minibatch = minibatch

        obs_dim = int(np.prod(obs_shape))
        self.net = ActorCritic(obs_dim, action_dim).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer(n_steps, (obs_dim,), gamma, gae_lambda)
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, *, training: bool = True, **kwargs: Any) -> int:
        return int(self.sample(obs)[0])

    def sample(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """返回 (action, logprob, value),供训练循环记录轨迹。"""
        torch = self.torch
        x = torch.as_tensor(np.asarray(obs).reshape(1, -1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, logprob, _, value = self.net.act(x)
        return int(action.item()), float(logprob.item()), float(value.item())

    def store(self, obs, action, logprob, reward, done, value) -> None:
        self.buffer.add(np.asarray(obs).reshape(-1), action, logprob, reward, done, value)

    def update(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, float]]:
        torch = self.torch
        if self.buffer.ptr == 0:
            return None
        last_obs = self.buffer.obs[self.buffer.ptr - 1][None]
        with torch.no_grad():
            _, _, _, last_value = self.net.act(torch.as_tensor(last_obs, dtype=torch.float32, device=self.device))
        self.buffer.compute_returns_and_advantages(float(last_value.item()))
        data = self.buffer.get()
        n = len(data["actions"])
        idx = np.arange(n)

        stats = {"policy_loss": 0.0, "value_loss": 0.0}
        n_updates = 0
        for _ in range(self.n_epochs):
            self._rng.shuffle(idx)
            for start in range(0, n, self.minibatch):
                mb = idx[start : start + self.minibatch]
                obs = torch.as_tensor(data["obs"][mb], device=self.device)
                acts = torch.as_tensor(data["actions"][mb], device=self.device)
                old_lp = torch.as_tensor(data["logprobs"][mb], device=self.device)
                adv = torch.as_tensor(data["advantages"][mb], device=self.device)
                ret = torch.as_tensor(data["returns"][mb], device=self.device)

                lp, entropy, values = self.net.evaluate(obs, acts)
                ratio = torch.exp(lp - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                value_loss = torch.nn.functional.mse_loss(values, ret)

                loss = policy_loss + self.value_coef * value_loss
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                n_updates += 1

        self.buffer.clear()
        n_updates = max(1, n_updates)
        return {"policy_loss": stats["policy_loss"] / n_updates, "value_loss": stats["value_loss"] / n_updates}

    def save(self, path: str) -> None:
        self.torch.save({"net": self.net.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = self.torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
