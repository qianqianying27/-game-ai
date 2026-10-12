from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import Env

# 动作: 上/下/左/右
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


class GridWorld(Env):
    """带障碍与陷阱的单智能体导航环境。

    智能体从起点出发,尽量少的步数到达目标(+1),
    踩到陷阱(-5),每步有微小成本(-0.02)以鼓励走最短路径。

    观测为位置 one-hot 向量,动作 4 个方向。
    """

    def __init__(
        self,
        grid: Optional[list] = None,
        step_cost: float = -0.02,
        goal_reward: float = 1.0,
        trap_reward: float = -5.0,
        max_steps: int = 100,
    ) -> None:
        # 默认地图: S 起点, G 目标, # 障碍, T 陷阱, . 空地
        if grid is None:
            grid = [
                "S..#...",
                ".###.#.",
                "...#.#.",
                ".#....#",
                ".#.T.#.",
                "...#.#G",
            ]
        self.grid = [list(row) for row in grid]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.max_steps = max_steps

        self.observation_shape = (self.height * self.width,)
        self.action_dim = 4

        self._start = self._find("S")
        self._goal = self._find("G")
        self._state = self._start
        self._steps = 0

    def _find(self, ch: str) -> int:
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == ch:
                    return r * self.width + c
        raise ValueError(f"grid 中缺少字符 {ch!r}")

    def _is_walkable(self, idx: int) -> bool:
        r, c = divmod(idx, self.width)
        return self.grid[r][c] != "#"

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        obs[self._state] = 1.0
        return obs

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        self._state = self._start
        self._steps = 0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        dr, dc = ACTIONS[int(action)]
        r, c = divmod(self._state, self.width)
        nr, nc = r + dr, c + dc
        self._steps += 1

        if 0 <= nr < self.height and 0 <= nc < self.width:
            nxt = nr * self.width + nc
            if self._is_walkable(nxt):
                self._state = nxt

        reward = self.step_cost
        terminated = False
        cell = self.grid[self._state // self.width][self._state % self.width]
        if cell == "G":
            reward += self.goal_reward
            terminated = True
        elif cell == "T":
            reward += self.trap_reward
            terminated = True

        truncated = self._steps >= self.max_steps and not terminated
        return self._obs(), reward, terminated, truncated, {}

    def render(self) -> str:
        lines = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                idx = r * self.width + c
                row.append("@" if idx == self._state else self.grid[r][c])
            lines.append(" ".join(row))
        return "\n".join(lines)
