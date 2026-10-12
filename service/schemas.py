from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class MoveRequest(BaseModel):
    """落子请求:board 为二维数组,0=空,1=玩家1,2=玩家2。"""

    board: List[List[int]] = Field(..., description="当前棋盘状态")
    player: int = Field(..., ge=1, le=2, description="请求 AI 决策的行棋方")
    model: Optional[str] = Field(None, description="模型标识,默认使用在线模型")


class MoveResponse(BaseModel):
    action: int = Field(..., description="AI 选择的列(0-based)")
    player: int
    model: str
    confidence: float = Field(..., description="所选动作的 softmax 置信度")


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
