from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroNet(nn.Module):
    """AlphaZero 风格网络:共享 CNN 主干 + 策略头(policy) + 价值头(value)。

    - 输入:观测张量 (B, C, H, W),采用"行棋方视角"编码。
    - 策略头输出每个动作的 logits;价值头输出 ``tanh`` 压缩到 [-1, 1] 的
      行棋方胜率评估。
    """

    def __init__(self, obs_shape: Tuple[int, int, int], action_dim: int, channels: int = 64) -> None:
        super().__init__()
        c, h, w = obs_shape
        self.trunk = nn.Sequential(
            nn.Conv2d(c, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        flat = channels * h * w
        self.policy_head = nn.Sequential(nn.Linear(flat, 128), nn.ReLU(), nn.Linear(128, action_dim))
        self.value_head = nn.Sequential(nn.Linear(flat, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(x).flatten(1)
        return self.policy_head(feat), self.value_head(feat).squeeze(-1)

    @torch.no_grad()
    def predict(self, obs: np.ndarray, device: str = "cpu") -> Tuple[np.ndarray, float]:
        """对单个观测返回 (policy 概率分布, value)。"""
        self.eval()
        x = torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32, device=device)
        logits, value = self.forward(x)
        policy = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return policy, float(value.item())
