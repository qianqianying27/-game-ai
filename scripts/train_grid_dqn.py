"""在 GridWorld 上训练 DQN 智能体。

用法:
    python scripts/train_grid_dqn.py --episodes 400
"""
from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401  (路径引导)

from agents.dqn import DQNAgent
from envs.grid_world import GridWorld
from utils import MetricsLogger, describe_device, resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="checkpoints/grid_dqn.pt")
    parser.add_argument("--metrics", default="runs/grid_dqn.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[device] {describe_device(args.device)}")

    env = GridWorld()
    agent = DQNAgent(env.observation_shape, env.action_dim, lr=args.lr, gamma=args.gamma, device=device)

    returns = []
    metrics = MetricsLogger(args.metrics)
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            action = agent.act(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.observe({"obs": obs, "action": action, "reward": reward, "next_obs": next_obs, "terminated": terminated})
            agent.update()
            obs = next_obs
            total += reward
            done = terminated or truncated
        returns.append(total)
        metrics.log(step=ep, return_=total, epsilon=agent.epsilon())
        if ep % 20 == 0:
            avg = sum(returns[-20:]) / 20
            print(f"ep={ep:4d}  return(20)={avg:+.3f}  eps={agent.epsilon():.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    agent.save(args.out)
    print(f"[done] 模型已保存到 {args.out}")
    metrics.save()
    print("\n=== 训练指标趋势 ===")
    metrics.print_summary()

    # 贪心评估一条轨迹
    obs = env.reset()
    for _ in range(200):
        a = agent.act(obs, training=False)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    print("=== 评估最终状态 (@ 为智能体) ===")
    print(env.render())


if __name__ == "__main__":
    main()
