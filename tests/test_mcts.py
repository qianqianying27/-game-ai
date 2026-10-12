from __future__ import annotations

import numpy as np

from search.mcts import AlphaZeroMCTS, ConnectFourState


def test_state_apply_switches_player():
    s = ConnectFourState()
    assert s.player_to_move == 1
    s.apply(3)
    assert s.player_to_move == 2
    assert s.board[5, 3] == 1  # 底部落子


def test_state_terminal_value_perspective():
    s = ConnectFourState()
    # 玩家1 在第 0..3 列底部横连获胜
    seq = [0, 0, 1, 0, 2, 0, 3]  # 交替,玩家2 都填第 0 列
    for a in seq:
        if s.done:
            break
        s.apply(a)
    assert s.done and s.winner == 1
    # 结束后 player_to_move 已切换到玩家2(输家),其视角应为 -1
    assert s.value_for_player_to_move() == -1.0


def test_mcts_returns_valid_policy():
    pytest_torch = __import__("pytest")
    pytest_torch.importorskip("torch")
    from models.alphazero_net import AlphaZeroNet

    net = AlphaZeroNet((2, 6, 7), 7)
    mcts = AlphaZeroMCTS(net, device="cpu", n_simulations=24)
    s = ConnectFourState()
    policy, root = mcts.search(s, add_noise=False)
    assert policy.shape == (7,)
    assert abs(policy.sum() - 1.0) < 1e-5
    assert (policy >= 0).all()
    assert root.N >= 1  # 根节点被反传访问
