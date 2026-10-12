from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from buffers.replay_buffer import ReplayBuffer
from models.networks import build_qnet

from .base import Agent


class DQNAgent(Agent):
    """带目标网络 + 合法动作掩码 + 线性 epsilon 衰减的 DQN。

    ``zero_sum=True`` 时,Bellman 自举项取负(下一步为对手视角),
    用于双人零和竞技游戏(如四子棋)的自对弈学习。
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 128,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 20_000,
        target_update: int = 500,
        zero_sum: bool = False,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.obs_shape = tuple(int(s) for s in obs_shape)
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.zero_sum = zero_sum
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self._rng = np.random.default_rng(seed)

        self.qnet = build_qnet(self.obs_shape, action_dim).to(self.device)
        self.target = copy.deepcopy(self.qnet)
        self.opt = torch.optim.Adam(self.qnet.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, seed=seed)
        self._grad_steps = 0

    # ---- 决策 ----
    def epsilon(self) -> float:
        frac = min(1.0, self._grad_steps / self.eps_decay)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _mask_of(self, legal_actions: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if legal_actions is None or len(legal_actions) == 0:
            return None
        mask = np.zeros(self.action_dim, dtype=np.float32)
        mask[np.asarray(legal_actions, dtype=np.int64)] = 1.0
        return mask

    def _q(self, net, obs: np.ndarray, mask: Optional[np.ndarray]):
        torch = self.torch
        x = torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = net(x).squeeze(0)
        if mask is not None:
            m = torch.as_tensor(mask, device=self.device)
            q = q.masked_fill(~m.bool(), -math.inf)
        return q

    def greedy_with(self, net, obs: np.ndarray, legal_actions: Optional[np.ndarray] = None) -> int:
        """用给定的网络(如对手快照 qnet)做贪心决策。"""
        q = self._q(net, obs, self._mask_of(legal_actions))
        return int(q.argmax().item())

    def act(self, obs: np.ndarray, *, legal_actions: Optional[np.ndarray] = None, training: bool = True, **kwargs: Any) -> int:
        legal = None if legal_actions is None else np.asarray(legal_actions, dtype=np.int64)
        mask = self._mask_of(legal)

        if training and self._rng.random() < self.epsilon():
            if legal is not None and len(legal) > 0:
                return int(self._rng.choice(legal))
            return int(self._rng.integers(self.action_dim))

        return self.greedy_with(self.qnet, obs, legal)

    def greedy(self, obs: np.ndarray, legal_actions: Optional[np.ndarray] = None) -> int:
        return self.act(obs, legal_actions=legal_actions, training=False)

    # ---- 学习 ----
    def observe(self, transition: Dict[str, Any]) -> None:
        self.buffer.add(
            transition["obs"],
            transition["action"],
            transition["reward"],
            transition["next_obs"],
            transition["terminated"],
            transition.get("mask"),
        )

    def update(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, float]]:
        torch = self.torch
        if len(self.buffer) < max(self.batch_size, 1000):
            return None

        batch = self.buffer.sample(self.batch_size)
        obs = torch.as_tensor(batch["obs"], device=self.device)
        acts = torch.as_tensor(batch["action"], device=self.device)[:, None]
        rew = torch.as_tensor(batch["reward"], device=self.device)[:, None]
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device)
        done = torch.as_tensor(batch["terminated"], device=self.device)[:, None]

        q = self.qnet(obs).gather(1, acts)
        with torch.no_grad():
            next_q = self.target(next_obs).max(1, keepdim=True).values
            boot = -next_q if self.zero_sum else next_q
            target = rew + self.gamma * boot * (1.0 - done)

        loss = torch.nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 1.0)
        self.opt.step()

        self._grad_steps += 1
        if self._grad_steps % self.target_update == 0:
            self.target.load_state_dict(self.qnet.state_dict())
        return {"loss": float(loss.item()), "epsilon": self.epsilon()}

    # ---- 持久化 ----
    def save(self, path: str) -> None:
        self.torch.save({"qnet": self.qnet.state_dict(), "obs_shape": self.obs_shape, "action_dim": self.action_dim}, path)

    def load(self, path: str) -> None:
        ckpt = self.torch.load(path, map_location=self.device)
        self.qnet.load_state_dict(ckpt["qnet"])
        self.target.load_state_dict(self.qnet.state_dict())
