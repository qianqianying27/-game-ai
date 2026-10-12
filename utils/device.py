from __future__ import annotations

from typing import Optional


def resolve_device(prefer: str = "auto", index: Optional[int] = None) -> str:
    """解析计算设备字符串(对 GPU 友好)。

    参数:
        prefer: "auto" / "cuda" / "cpu" / "mps"。
            - "auto": 优先使用 GPU(CUDA),其次 Apple Silicon 的 MPS,最后回退 CPU。
            - "cuda": 显式要求 GPU;若不可用则回退到 CPU 并打印告警。
        index: 多卡时指定 GPU 编号,返回形如 "cuda:1";为 None 时用 "cuda"。

    返回:
        可直接传给 ``torch.device(...)`` 或 ``Tensor.to(...)`` 的设备字符串。
    """
    prefer = (prefer or "auto").lower()

    if prefer == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:  # pragma: no cover - 未安装 torch 时降级
        return "cpu"

    # 显式或自动优先尝试 CUDA
    if torch.cuda.is_available():
        return f"cuda:{index}" if index is not None else "cuda"

    # Apple Silicon GPU 支持(仅在 auto 时启用,避免误判)
    if prefer == "auto":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"

    if prefer == "cuda":
        print("[resolve_device] 请求使用 CUDA,但当前环境不可用,已回退到 CPU。")
    return "cpu"


def get_device(prefer: str = "auto", index: Optional[int] = None):
    """返回 ``torch.device`` 对象(便捷封装)。"""
    import torch

    return torch.device(resolve_device(prefer, index))


def describe_device(prefer: str = "auto") -> str:
    """返回可读的设备描述,便于训练脚本打印日志。"""
    device = resolve_device(prefer)
    if device.startswith("cuda"):
        import torch

        idx = int(device.split(":")[1]) if ":" in device else 0
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        return f"{device} ({name}, sm_{cap[0]}{cap[1]}, CUDA {torch.version.cuda})"
    return device
