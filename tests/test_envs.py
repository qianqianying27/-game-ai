from __future__ import annotations

import numpy as np

from envs.connect_four import ConnectFour
from envs.grid_world import GridWorld


def test_gridworld_shapes_and_onehot():
    env = GridWorld()
    obs = env.reset(seed=0)
    assert obs.shape == env.observation_shape
    assert obs.sum() == 1.0  # one-hot
    total = 0.0
    for _ in range(100):
        _, r, term, trunc, _ = env.step(int(np.random.randint(env.action_dim)))
        total += r
        if term or trunc:
            break
    assert isinstance(total, float)


def test_connect_four_win_horizontal():
    env = ConnectFour()
    env.reset()
    # 玩家1 在第 0,1,2,3 列各下一子即可能在底部形成横向四连
    for col in [0, 0, 1, 0, 2, 0, 3]:
        env.step(col)
        if env.terminated:
            break
    assert env.terminated and env.winner == 1


def test_connect_four_observe_perspective():
    env = ConnectFour()
    env.reset()
    env.step(3)  # 玩家1 落在第 3 列
    obs1 = env.observe(1)
    obs2 = env.observe(2)
    # 玩家1 落子后:对我(1)是第一通道,对(2)是第二通道
    assert obs1[0].sum() == 1 and obs1[1].sum() == 0
    assert obs2[0].sum() == 0 and obs2[1].sum() == 1


def test_connect_four_illegal_column_penalized():
    env = ConnectFour()
    env.reset()
    env.board[:, 0] = [1, 1, 1, 1, 1, 1]  # 第 0 列已满
    _, reward, terminated, _, info = env.step(0)
    assert terminated and reward < 0 and info["invalid"]
