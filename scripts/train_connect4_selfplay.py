"""四子棋自对弈训练(竞技对抗)。

用法:
    python scripts/train_connect4_selfplay.py --episodes 2000
"""
from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401

from agents.dqn import DQNAgent
from envs.connect_four import ConnectFour
from selfplay.league import SelfPlayLeague
from utils import MetricsLogger, describe_device, resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="checkpoints/connect4_dqn.pt")
    parser.add_argument("--metrics", default="runs/connect4_selfplay.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[device] {describe_device(args.device)}")

    env = ConnectFour()
    agent = DQNAgent(env.observation_shape, env.action_dim, zero_sum=True, device=device)
    league = SelfPlayLeague(agent, seed=args.seed)
    metrics = MetricsLogger(args.metrics)
    stats = league.train(episodes=args.episodes, metrics=metrics)
    print(f"[result] 学习方胜率={stats['win_rate']:.3f}  平局={stats['draw_rate']:.3f}  负率={stats['loss_rate']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    agent.save(args.out)
    print(f"[done] 模型已保存到 {args.out}")
    metrics.save()
    print("\n=== 训练指标趋势 ===")
    metrics.print_summary()


if __name__ == "__main__":
    main()
