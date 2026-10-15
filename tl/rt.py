import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
F32 = torch.float32


def use(device=None, dtype=None):
    global DEV, DT
    if device:
        DEV = device
    if dtype:
        DT = dtype
    return DEV, DT


def t(x, dtype=None, device=None):
    return torch.as_tensor(x, dtype=dtype or DT, device=device or DEV)


def sync():
    if str(DEV).startswith("cuda"):
        torch.cuda.synchronize()


def peak_gb():
    if str(DEV).startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2 ** 30
    return float("nan")
