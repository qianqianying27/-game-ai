from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

EMPTY = 0
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


class ConnectFourState:
    """MCTS 使用的四子棋状态机(轻量、可克隆)。

    约定:``player_to_move`` 始终为该落子的行棋方;``apply`` 之后**总是切换**
    行棋方(即便上一手已获胜),从而终局节点的 ``value_for_player_to_move``
    表示"刚输的一方视角",配合极小极大反传。
    """

    def __init__(self, height: int = 6, width: int = 7) -> None:
        self.height = height
        self.width = width
        self.action_dim = width
        self.board = np.zeros((height, width), dtype=np.int8)
        self.player_to_move = 1
        self.done = False
        self.winner = 0  # 0=未结束或平局,1/2=获胜方

    def clone(self) -> "ConnectFourState":
        s = ConnectFourState(self.height, self.width)
        s.board = self.board.copy()
        s.player_to_move = self.player_to_move
        s.done = self.done
        s.winner = self.winner
        return s

    def legal_actions(self) -> List[int]:
        return [int(c) for c in np.flatnonzero(self.board[0] == EMPTY)]

    def observe(self) -> np.ndarray:
        """行棋方视角的两通道观测 (2, H, W)。"""
        me, opp = self.player_to_move, 3 - self.player_to_move
        obs = np.zeros((2, self.height, self.width), dtype=np.float32)
        obs[0] = self.board == me
        obs[1] = self.board == opp
        return obs

    def apply(self, action: int) -> None:
        if self.done:
            raise RuntimeError("game already finished")
        col = int(action)
        row = self._drop(col, self.player_to_move)
        if self._check_win(row, col, self.player_to_move):
            self.done = True
            self.winner = self.player_to_move
        elif (self.board != EMPTY).all():
            self.done = True
            self.winner = 0
        self.player_to_move = 3 - self.player_to_move  # 总是切换

    def value_for_player_to_move(self) -> float:
        """终局时从当前行棋方视角的取值:胜 +1 / 负 -1 / 平 0。"""
        if not self.done:
            raise RuntimeError("state not terminal")
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == self.player_to_move else -1.0

    # ---- 规则内核 ----
    def _drop(self, col: int, player: int) -> int:
        for row in range(self.height - 1, -1, -1):
            if self.board[row, col] == EMPTY:
                self.board[row, col] = player
                return row
        raise ValueError(f"column {col} is full")

    def _check_win(self, row: int, col: int, player: int) -> bool:
        for dr, dc in DIRS:
            count = 1
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while 0 <= r < self.height and 0 <= c < self.width and self.board[r, c] == player:
                    count += 1
                    r += dr * sign
                    c += dc * sign
                if count >= 4:
                    return True
        return False

    def render(self) -> str:
        symbols = {EMPTY: ".", 1: "X", 2: "O"}
        lines = [" ".join(symbols[v] for v in row) for row in self.board]
        return "\n".join(reversed(lines))


class _Node:
    __slots__ = ("N", "W", "P", "legal", "children")

    def __init__(self) -> None:
        self.N = 0
        self.W = 0.0
        self.P: Optional[np.ndarray] = None
        self.legal: List[int] = []
        self.children: Dict[int, "_Node"] = {}


class AlphaZeroMCTS:
    """基于策略-价值网络的 P-UCT 蒙特卡洛树搜索(AlphaZero 风格)。

    - ``select``:  最大化 ``-Q_child + cpuct * P * sqrt(N_parent) / (1 + N_child)``
    - ``expand``:  用网络给出策略先验与局面价值(行棋方视角)
    - ``backprop``: 逐层取负反传(零和)
    - 根节点可加 Dirichlet 噪声鼓励探索
    """

    def __init__(
        self,
        net,
        device: str = "cpu",
        n_simulations: int = 64,
        cpuct: float = 1.5,
        dirichlet_alpha: float = 0.5,
        noise_frac: float = 0.25,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.net = net
        self.device = device
        self.n_simulations = n_simulations
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_frac = noise_frac
        self._rng = rng or np.random.default_rng()

    def _expand(self, node: _Node, state: ConnectFourState) -> float:
        policy, value = self.net.predict(state.observe(), self.device)
        legal = state.legal_actions()
        masked = np.zeros_like(policy)
        for a in legal:
            masked[a] = policy[a]
        total = masked.sum()
        node.P = masked / total if total > 0 else np.full(len(legal), 1.0 / max(1, len(legal)))
        node.legal = legal
        return value

    def _select(self, node: _Node) -> int:
        sqrt_n = math.sqrt(node.N + 1e-8)
        best_u, best_a = -math.inf, node.legal[0]
        for a in node.legal:
            child = node.children.get(a)
            if child is None:
                child = node.children[a] = _Node()
            if child.N == 0:
                return a  # 优先探索未访问节点
            q_parent = -child.W / child.N
            u = q_parent + self.cpuct * node.P[a] * sqrt_n / (1 + child.N)
            if u > best_u:
                best_u, best_a = u, a
        return best_a

    def _playout(self, node: _Node, state: ConnectFourState) -> float:
        if state.done:
            v = state.value_for_player_to_move()
            node.N += 1
            node.W += v
            return v
        if node.P is None:  # 叶子:用网络评估
            v = self._expand(node, state)
            node.N += 1
            node.W += v
            return v
        a = self._select(node)
        child = node.children[a]
        next_state = state.clone()
        next_state.apply(a)
        v_child = self._playout(child, next_state)
        v = -v_child  # 反传到当前行棋方视角
        node.N += 1
        node.W += v
        return v

    def search(self, state: ConnectFourState, add_noise: bool = False) -> Tuple[np.ndarray, _Node]:
        root = _Node()
        self._expand(root, state)
        if add_noise and root.legal:
            noise = self._rng.dirichlet(np.full(len(root.legal), self.dirichlet_alpha))
            for i, a in enumerate(root.legal):
                root.P[a] = (1 - self.noise_frac) * root.P[a] + self.noise_frac * noise[i]
        for _ in range(self.n_simulations):
            self._playout(root, state)

        visits = np.zeros(state.action_dim, dtype=np.float32)
        for a, child in root.children.items():
            visits[a] = child.N
        total = visits.sum()
        policy = visits / total if total > 0 else np.full(state.action_dim, 1.0 / state.action_dim)
        return policy, root
