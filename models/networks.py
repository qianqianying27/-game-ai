from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def _mlp(sizes: Sequence[int]) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i != len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MLPQNet(nn.Module):
    """用于扁平观测(如 GridWorld)的 Q 值网络。"""

    def __init__(self, obs_dim: int, action_dim: int, hidden: Tuple[int, ...] = (128, 128)):
        super().__init__()
        self.net = _mlp((obs_dim, *hidden, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class CNNQNet(nn.Module):
    """用于图像/棋盘观测(如四子棋)的卷积 Q 值网络。"""

    def __init__(self, obs_shape: Tuple[int, int, int], action_dim: int, hidden: int = 128):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * h * w, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        return self.head(feat.flatten(1))


class ActorCritic(nn.Module):
    """共享主干的 Actor-Critic 网络,用于 PPO。"""

    def __init__(self, obs_dim: int, action_dim: int, hidden: Tuple[int, ...] = (64, 64)):
        super().__init__()
        self.base = _mlp((obs_dim, *hidden))
        base_out = hidden[-1]
        self.actor = nn.Linear(base_out, action_dim)
        self.critic = nn.Linear(base_out, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.base(x.flatten(1))
        return self.actor(feat), self.critic(feat).squeeze(-1)

    def act(self, x: torch.Tensor):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, x: torch.Tensor, actions: torch.Tensor):
        logits, values = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


def build_qnet(obs_shape, action_dim: int, hidden: Tuple[int, ...] = (128, 128)) -> nn.Module:
    """根据观测形状自动选择 MLP 或 CNN 架构。"""
    obs_shape = tuple(int(s) for s in obs_shape)
    if len(obs_shape) == 3:  # (C, H, W) -> 卷积
        return CNNQNet(obs_shape, action_dim)
    return MLPQNet(int(np.prod(obs_shape)), action_dim, hidden)
