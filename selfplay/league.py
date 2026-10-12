from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np

from envs.connect_four import ConnectFour


class SelfPlayLeague:
    """四子棋自对弈训练器(竞技对抗)。

    - 学习方固定执 ``learner_side``(默认玩家 1),对手为历史权重快照
      (首次快照前为随机策略)。
    - 由于观测采用"行棋方视角"编码,同一网络可服务双方,学到的策略与
      玩家编号无关。
    - 借助 :class:`DQNAgent` 的 ``zero_sum`` 目标(自举项取负)实现
      极大极小型的自对弈 Q 学习。
    """

    def __init__(self, agent, learner_side: int = 1, seed: Optional[int] = None) -> None:
        self.agent = agent
        self.learner_side = learner_side
        self.opp_net = None  # 对手网络快照(torch.nn.Module)
        self._rng = np.random.default_rng(seed)

    def _sync_opponent(self) -> None:
        self.opp_net = copy.deepcopy(self.agent.qnet).eval()

    def play_episode(self, training: bool = True) -> Dict[str, float]:
        env = ConnectFour()
        obs = env.reset()
        learner = self.learner_side
        steps = 0
        info: Dict[str, object] = {}
        last_loss = float("nan")

        while True:
            player = env.current_player
            legal = env.legal_actions()
            is_learner = player == learner

            if is_learner:
                action = self.agent.act(obs, legal_actions=legal, training=training)
            elif self.opp_net is not None:
                action = self.agent.greedy_with(self.opp_net, obs, legal)
            else:
                action = int(self._rng.choice(legal))

            obs_before = obs
            next_obs, reward, terminated, truncated, info = env.step(action)

            if is_learner and training:
                mask = np.zeros(env.action_dim, dtype=np.float32)
                mask[legal] = 1.0
                self.agent.observe(
                    {
                        "obs": obs_before,
                        "action": action,
                        "reward": reward,
                        "next_obs": next_obs,
                        "terminated": terminated,
                        "mask": mask,
                    }
                )
                upd = self.agent.update()
                if upd:
                    last_loss = upd["loss"]

            obs = next_obs
            steps += 1
            if terminated or truncated:
                break

        winner = info.get("winner") or 0
        return {"winner": float(winner), "learner": float(learner), "steps": float(steps), "loss": last_loss}

    def train(self, episodes: int, snapshot_every: int = 200, log_every: int = 100, metrics=None) -> Dict[str, float]:
        wins = draws = losses = 0
        learner = self.learner_side
        for ep in range(1, episodes + 1):
            stats = self.play_episode(training=True)
            winner = stats["winner"]
            if winner == learner:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
            if metrics is not None:
                metrics.log(step=ep, loss=stats["loss"], win=float(winner == learner), win_rate=wins / ep)
            if ep % snapshot_every == 0:
                self._sync_opponent()
            if ep % log_every == 0:
                print(f"[selfplay] ep={ep}  win={wins}  draw={draws}  loss={losses}")
        return {"win_rate": wins / episodes, "draw_rate": draws / episodes, "loss_rate": losses / episodes}
