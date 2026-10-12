from __future__ import annotations

import pytest

pytest.importorskip("torch")

import numpy as np

from envs.connect_four import ConnectFour
from models.alphazero_net import AlphaZeroNet
from selfplay.alphazero_trainer import AlphaZeroTrainer


def test_alphazero_net_predict():
    net = AlphaZeroNet((2, 6, 7), 7)
    obs = np.zeros((2, 6, 7), dtype=np.float32)
    policy, value = net.predict(obs, device="cpu")
    assert policy.shape == (7,)
    assert abs(policy.sum() - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0


def test_alphazero_trainer_step():
    net = AlphaZeroNet((2, 6, 7), 7)
    trainer = AlphaZeroTrainer(net, device="cpu", n_simulations=8, seed=0)
    stats = trainer.train_step(games=1)
    assert {"policy_loss", "value_loss", "loss"} <= set(stats)
    assert stats["loss"] >= 0.0
