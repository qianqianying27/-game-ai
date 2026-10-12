from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from search.mcts import AlphaZeroMCTS, ConnectFourState


class AlphaZeroTrainer:
    """AlphaZero 风格自对弈训练器。

    流程:
    1. 自对弈产生数据:每步用 MCTS 得到访问分布作为策略目标,记录落子方视角观测;
    2. 以最终胜负作为价值目标 z(落子方视角 +1/-1/0);
    3. 用交叉熵(策略)+ 均方误差(价值)+ L2 正则更新网络。
    """

    def __init__(
        self,
        net,
        device: str = "cpu",
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        n_simulations: int = 64,
        temp_moves: int = 6,
        seed: Optional[int] = None,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self._rng = np.random.default_rng(seed)
        self.mcts = AlphaZeroMCTS(
            self.net, device=device, n_simulations=n_simulations, rng=self._rng
        )
        self.temp_moves = temp_moves

    # ---- 自对弈 ----
    def _choose_action(self, policy: np.ndarray, move_idx: int) -> int:
        if move_idx < self.temp_moves:
            return int(self._rng.choice(len(policy), p=policy))
        return int(np.argmax(policy))

    def self_play_game(self) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int]], int]:
        state = ConnectFourState()
        examples: List[Tuple[np.ndarray, np.ndarray, int]] = []
        move_idx = 0
        while not state.done:
            obs = state.observe()
            player = state.player_to_move
            policy, _ = self.mcts.search(state, add_noise=True)
            action = self._choose_action(policy, move_idx)
            examples.append((obs, policy, player))
            state.apply(action)
            move_idx += 1
        return examples, state.winner

    def _build_batch(self, examples, winner):
        torch = self.torch
        obs = np.stack([e[0] for e in examples])
        pol = np.stack([e[1] for e in examples])
        players = np.array([e[2] for e in examples], dtype=np.float32)
        z = np.where(players == winner, 1.0, np.where(winner == 0, 0.0, -1.0)).astype(np.float32)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        pol_t = torch.as_tensor(pol, dtype=torch.float32, device=self.device)
        z_t = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        return obs_t, pol_t, z_t

    def train_step(self, games: int) -> Dict[str, float]:
        torch = self.torch
        self.net.train()
        all_obs, all_pol, all_z = [], [], []
        wins = 0
        for _ in range(games):
            examples, winner = self.self_play_game()
            if winner != 0:
                wins += 1
            obs_t, pol_t, z_t = self._build_batch(examples, winner)
            all_obs.append(obs_t)
            all_pol.append(pol_t)
            all_z.append(z_t)

        obs = torch.cat(all_obs)
        pol = torch.cat(all_pol)
        z = torch.cat(all_z)

        logits, value = self.net(obs)
        logpi = torch.log_softmax(logits, dim=-1)
        policy_loss = -(pol * logpi).sum(1).mean()
        value_loss = torch.nn.functional.mse_loss(value, z)
        loss = policy_loss + value_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "loss": float(loss.item()),
            "decisive_rate": wins / max(1, games),
        }

    def train(self, iterations: int, games_per_iter: int = 4, log_every: int = 5, metrics=None) -> Dict[str, float]:
        last: Dict[str, float] = {}
        for it in range(1, iterations + 1):
            last = self.train_step(games_per_iter)
            if metrics is not None:
                metrics.log(step=it, **last)
            if it % log_every == 0:
                print(f"[alphazero] iter={it}  " + "  ".join(f"{k}={v:.4f}" for k, v in last.items()))
        return last

    def save(self, path: str) -> None:
        self.torch.save({"net": self.net.state_dict()}, path)
