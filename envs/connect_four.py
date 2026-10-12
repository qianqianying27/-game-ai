from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import Env

EMPTY = 0
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))  # 横 / 竖 / 两条对角线


class ConnectFour(Env):
    """竞技对抗类双人零和游戏 —— 四子棋 (Connect Four)。

    - 棋盘 6 行 x 7 列,玩家轮流落子,棋子受重力下落。
    - 率先横/竖/斜四连者获胜。
    - 观测采用"当前行棋方视角"的两通道平面 (2, H, W):
        通道 0 = 我方棋子,通道 1 = 对方棋子。
      这样同一套策略网络可服务双方,便于自对弈。
    """

    def __init__(self, height: int = 6, width: int = 7) -> None:
        self.height = height
        self.width = width
        self.observation_shape = (2, height, width)
        self.action_dim = width
        self.board = np.zeros((height, width), dtype=np.int8)
        self.current_player = 1  # 1 或 2
        self.winner = 0
        self.terminated = False

    # ---- Env 接口 ----
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.current_player = 1
        self.winner = 0
        self.terminated = False
        return self.observe(self.current_player)

    def legal_actions(self) -> np.ndarray:
        """返回可用列(未落满的列)。"""
        return np.flatnonzero(self.board[0] == EMPTY).astype(np.int64)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        if self.terminated or action < 0 or action >= self.width or not self._can_drop(action):
            # 非法动作: 给予惩罚并结束,便于训练时约束策略
            return (
                self.observe(self.current_player),
                -1.0,
                True,
                False,
                {"invalid": True, "winner": 3 - self.current_player},
            )

        mover = self.current_player
        row = self._drop(action, mover)
        info: Dict[str, Any] = {"invalid": False, "mover": mover}

        if self._check_win(row, action, mover):
            self.terminated = True
            self.winner = mover
            reward = 1.0
            info["winner"] = mover
        elif (self.board != EMPTY).all():
            self.terminated = True
            self.winner = 0  # 平局
            reward = 0.0
            info["winner"] = 0
        else:
            reward = 0.0
            self.current_player = 3 - mover
            info["winner"] = None

        return self.observe(self.current_player), reward, self.terminated, False, info

    # ---- 视角观测 ----
    def observe(self, player: int) -> np.ndarray:
        opp = 3 - player
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        obs[0] = self.board == player
        obs[1] = self.board == opp
        return obs

    # ---- 规则内核 ----
    def _can_drop(self, col: int) -> bool:
        return self.board[0, col] == EMPTY

    def _drop(self, col: int, player: int) -> int:
        for row in range(self.height - 1, -1, -1):
            if self.board[row, col] == EMPTY:
                self.board[row, col] = player
                return row
        raise RuntimeError(f"column {col} is full")

    def _check_win(self, row: int, col: int, player: int) -> bool:
        for dr, dc in DIRS:
            count = 1
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while (
                    0 <= r < self.height
                    and 0 <= c < self.width
                    and self.board[r, c] == player
                ):
                    count += 1
                    r += dr * sign
                    c += dc * sign
                if count >= 4:
                    return True
        return False

    @property
    def is_single_agent(self) -> bool:
        return False

    def render(self) -> str:
        symbols = {EMPTY: ".", 1: "X", 2: "O"}
        lines = [" ".join(symbols[v] for v in row) for row in self.board]
        return "\n".join(reversed(lines))
