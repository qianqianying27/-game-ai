"""AlphaZero 风格四子棋自对弈训练(MCTS + 策略价值网络)。

用法:
    python scripts/train_alphazero.py --iterations 20 --games-per-iter 4 --sims 64
"""
from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401

from envs.connect_four import ConnectFour
from models.alphazero_net import AlphaZeroNet
from selfplay.alphazero_trainer import AlphaZeroTrainer
from utils import MetricsLogger, describe_device, resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--games-per-iter", type=int, default=4)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="checkpoints/connect4_alphazero.pt")
    parser.add_argument("--metrics", default="runs/alphazero.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[device] {describe_device(args.device)}")

    env = ConnectFour()
    net = AlphaZeroNet(env.observation_shape, env.action_dim)
    trainer = AlphaZeroTrainer(net, device=device, lr=args.lr, n_simulations=args.sims, seed=args.seed)

    metrics = MetricsLogger(args.metrics)
    trainer.train(args.iterations, games_per_iter=args.games_per_iter, log_every=5, metrics=metrics)

    metrics.save()
    print("\n=== 训练指标趋势 ===")
    metrics.print_summary()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    trainer.save(args.out)
    print(f"[done] 模型已保存到 {args.out}\n[done] 指标已保存到 {args.metrics}")


if __name__ == "__main__":
    main()
