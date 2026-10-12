from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from service.app import create_app  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(create_app(model_path="__non_existent__.pt"))


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False  # 无权重时退化随机


def test_move(client):
    board = [[0] * 7 for _ in range(6)]
    resp = client.post("/v1/move", json={"board": board, "player": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["action"] < 7
    assert 0.0 <= body["confidence"] <= 1.0


def test_move_full_board_error(client):
    board = [[1] * 7 for _ in range(6)]
    resp = client.post("/v1/move", json={"board": board, "player": 2})
    assert resp.status_code == 400
