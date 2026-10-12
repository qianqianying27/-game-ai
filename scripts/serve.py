"""启动游戏 AI 推理服务(FastAPI + Uvicorn)。

用法:
    python scripts/serve.py --port 8000 --model checkpoints/connect4_dqn.pt

启动后可访问:
    GET  /health      健康检查
    POST /v1/move     落子决策
    GET  /docs        交互式 API 文档
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="checkpoints/connect4_dqn.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import uvicorn

    from service.app import create_app

    uvicorn.run(create_app(args.model, args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
