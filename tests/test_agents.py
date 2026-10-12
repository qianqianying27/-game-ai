from __future__ import annotations

import numpy as np

from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from agents.random_agent import RandomAgent
from envs.connect_four import ConnectFour
from envs.grid_world import GridWorld


def test_random_agent_respects_legal_actions():
    env = ConnectFour()
    env.reset()
    agent = RandomAgent(env.action_dim, seed=0)
    legal = set(int(x) for x in env.legal_actions())
    for _ in range(20):
        assert agent.act(env.observe(1), legal_actions=env.legal_actions()) in legal


def test_dqn_act_returns_valid_action():
    env = GridWorld()
    agent = DQNAgent(env.observation_shape, env.action_dim, seed=0)
    obs = env.reset()
    assert 0 <= agent.act(obs, training=False) < env.action_dim


def test_dqn_update_runs():
    env = GridWorld()
    agent = DQNAgent(env.observation_shape, env.action_dim, batch_size=32, seed=0)
    obs = env.reset()
    for _ in range(1500):
        a = agent.act(obs, training=True)
        next_obs, r, term, trunc, _ = env.step(a)
        agent.observe({"obs": obs, "action": a, "reward": r, "next_obs": next_obs, "terminated": term})
        agent.update()
        obs = next_obs if not (term or trunc) else env.reset()
    assert len(agent.buffer) > 0


def test_ppo_update_runs():
    env = GridWorld()
    agent = PPOAgent(env.observation_shape, env.action_dim, n_steps=128, seed=0)
    obs = env.reset()
    for _ in range(128):
        a, lp, v = agent.sample(obs)
        next_obs, r, term, trunc, _ = env.step(a)
        agent.store(obs, a, lp, r, float(term or trunc), v)
        obs = next_obs if not (term or trunc) else env.reset()
    stats = agent.update()
    assert "policy_loss" in stats and "value_loss" in stats
