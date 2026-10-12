from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from envs.connect_four import ConnectFour
from utils.device import resolve_device

from .schemas import HealthResponse, MoveRequest, MoveResponse
from .web import INDEX_HTML

__version__ = "0.1.0"
_MODEL_PATH = os.environ.get("GAMEAI_MODEL_PATH", "checkpoints/connect4_dqn.pt")


class GameEngine:
    """封装模型加载与推理,提供稳定的对局决策接口。

    若未找到已训练权重,则退化为随机策略,保证服务始终可用(优雅降级)。
    """

    def __init__(self, model_path: str = _MODEL_PATH, device: str = "auto") -> None:
        self.env = ConnectFour()
        self.model_path = model_path
        self.device = resolve_device(device)
        self.agent = None
        self.loaded = False
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.model_path):
            return
        try:
            from agents.dqn import DQNAgent

            agent = DQNAgent(self.env.observation_shape, self.env.action_dim, device=self.device)
            agent.load(self.model_path)
            agent.qnet.eval()
            self.agent = agent
            self.loaded = True
            print(f"[GameEngine] 已加载模型 {self.model_path} -> {self.device}")
        except Exception as exc:  # pragma: no cover - 保证服务健壮性
            print(f"[GameEngine] 模型加载失败,使用随机策略: {exc}")

    def decide(self, board: list, player: int) -> tuple:
        env = ConnectFour()
        arr = np.array(board, dtype=np.int8)
        if arr.shape != (env.height, env.width):
            raise ValueError(f"棋盘形状应为 ({env.height}, {env.width})")
        env.board = arr
        env.current_player = player
        legal = env.legal_actions()
        if len(legal) == 0:
            raise ValueError("没有合法落子位置")

        if self.agent is None:
            action = int(np.random.choice(legal))
            return action, 1.0 / len(legal)

        obs = env.observe(player)
        mask = np.zeros(env.action_dim, dtype=np.float32)
        mask[legal] = 1.0
        q = self.agent._q(self.agent.qnet, obs, mask).detach().cpu().numpy()
        masked = np.where(mask > 0, q, -math.inf)
        action = int(np.argmax(masked))
        return action, self._softmax_confidence(masked)

    @staticmethod
    def _softmax_confidence(logits: np.ndarray) -> float:
        finite = np.where(np.isfinite(logits), logits, 0.0)
        e = np.exp(finite - finite.max())
        probs = e / e.sum()
        return float(probs.max())


def create_app(model_path: str = _MODEL_PATH, device: str = "auto"):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="GameAI 推理服务", version=__version__)
    engine = GameEngine(model_path, device)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__, model_loaded=engine.loaded)

    @app.post("/v1/move", response_model=MoveResponse)
    def move(req: MoveRequest) -> MoveResponse:
        try:
            action, conf = engine.decide(req.board, req.player)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return MoveResponse(
            action=action,
            player=req.player,
            model=os.path.basename(engine.model_path),
            confidence=conf,
        )

    return app
