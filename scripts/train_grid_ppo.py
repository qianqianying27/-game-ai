"""在 GridWorld 上训练 PPO 智能体。

用法:
    python scripts/train_grid_ppo.py --updates 40
"""
from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401

from agents.ppo import PPOAgent
from envs.grid_world import GridWorld
from utils import MetricsLogger, describe_device, resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="checkpoints/grid_ppo.pt")
    parser.add_argument("--metrics", default="runs/grid_ppo.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[device] {describe_device(args.device)}")

    env = GridWorld()
    agent = PPOAgent(env.observation_shape, env.action_dim, n_steps=args.n_steps, device=device)

    obs = env.reset()
    ep_return, ep_steps = 0.0, 0
    metrics = MetricsLogger(args.metrics)
    for upd in range(1, args.updates + 1):
        while not agent.buffer.is_full():
            action, logprob, value = agent.sample(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store(obs, action, logprob, reward, float(terminated or truncated), value)
            obs = next_obs
            ep_return += reward
            ep_steps += 1
            if terminated or truncated:
                obs = env.reset()

        stats = agent.update()
        metrics.log(step=upd, return_=ep_return, **stats)
        print(f"update={upd:3d}  last_return={ep_return:+.2f}  " + "  ".join(f"{k}={v:.3f}" for k, v in stats.items()))
        ep_return, ep_steps = 0.0, 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    agent.save(args.out)
    print(f"[done] 模型已保存到 {args.out}")
    metrics.save()
    print("\n=== 训练指标趋势 ===")
    metrics.print_summary()


if __name__ == "__main__":
    main()
