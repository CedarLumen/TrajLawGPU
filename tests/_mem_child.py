"""Child process: builds only the distance tensor of the old version, to measure peak memory."""
import ctypes
import sys
import time

import numpy as np


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_mb():
    k32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    p = PMC()
    p.cb = ctypes.sizeof(PMC)
    h = k32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(h, ctypes.byref(p), p.cb):
        return -1.0
    return p.PeakWorkingSetSize / 2 ** 20


def main():
    rows = int(sys.argv[1])
    K = int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "old"
    rng = np.random.default_rng(1)
    Un = rng.normal(0, 1, (rows, 3))
    cen = rng.normal(0, 1, (K, 3))
    base = peak_mb()
    t0 = time.time()
    if mode == "old":
        d = ((Un[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        del d
    else:
        sub = Un[rng.choice(len(Un), min(30000, len(Un)), replace=False)]
        for _ in range(10):
            a = ((sub[:, None, :] - cen[None, :, :]) ** 2).sum(-1).argmin(1)
    print("%d %d %s %.1f %.2f" % (rows, K, mode, time.time() - t0, peak_mb() - base))


if __name__ == "__main__":
    main()
