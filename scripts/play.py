"""命令行对战:人类 vs 训练好的四子棋 AI。

用法:
    python scripts/play.py --model checkpoints/connect4_dqn.pt
    python scripts/play.py --ai-first
"""
from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401

from agents.dqn import DQNAgent
from agents.random_agent import RandomAgent
from envs.connect_four import ConnectFour
from utils import resolve_device


def load_agent(env: ConnectFour, path: str, device: str):
    if os.path.exists(path):
        agent = DQNAgent(env.observation_shape, env.action_dim, device=resolve_device(device))
        agent.load(path)
        agent.qnet.eval()
        print(f"[play] 已加载模型 {path}")
        return agent
    print(f"[play] 未找到 {path},使用随机基线。请先运行 train_connect4_selfplay.py")
    return RandomAgent(env.action_dim)


def ai_action(agent, env: ConnectFour, obs):
    if isinstance(agent, DQNAgent):
        return agent.greedy(obs, env.legal_actions())
    return agent.act(obs, legal_actions=env.legal_actions())


def human_input(env: ConnectFour) -> int:
    legal = set(int(x) for x in env.legal_actions())
    while True:
        try:
            col = int(input(f"选择列 0-{env.width - 1}: "))
        except ValueError:
            print("请输入数字。")
            continue
        if col in legal:
            return col
        print("该列已满或越界,请重试。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/connect4_dqn.pt")
    parser.add_argument("--ai-first", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    env = ConnectFour()
    agent = load_agent(env, args.model, args.device)
    obs = env.reset()
    human = 2 if args.ai_first else 1

    if args.ai_first:
        obs, _, _, _, _ = env.step(ai_action(agent, env, obs))

    print(env.render())
    while not env.terminated:
        if env.current_player == human:
            action = human_input(env)
        else:
            action = ai_action(agent, env, obs)
        obs, _, _, _, _ = env.step(action)
        print(env.render())

    print("胜者:", {0: "平局", 1: "玩家1(X)", 2: "玩家2(O)"}.get(env.winner))


if __name__ == "__main__":
    main()
