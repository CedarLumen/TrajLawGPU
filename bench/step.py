"""Cost accounting for one integration step: how long numpy / GPU eager / GPU graph / torch CPU take for the same work, and at what utilization."""
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, r"D:\FAST\TrajLaw\sim")
import trajlaw_pairsim as old  # noqa: E402
from tl import rt, scenes, sim  # noqa: E402
from tl.forces import Mix  # noqa: E402


def nv_util(stop, out):
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            out.append(r.stdout.strip())
        except Exception:
            pass
        time.sleep(0.2)


def watch(fn):
    stop = threading.Event()
    out = []
    threading.Thread(target=nv_util, args=(stop, out), daemon=True).start()
    t0 = time.time()
    fn()
    rt.sync()
    el = time.time() - t0
    stop.set()
    u = [int(x.split(",")[0]) for x in out if x]
    m = [int(x.split(",")[1]) for x in out if x]
    return el, (float(np.mean(u)) if u else float("nan")), (max(m) if m else 0)


def bench(tag, fn, steps, warm=None):
    if warm is not None:
        warm()
        rt.sync()
    el, u, mem = watch(fn)
    print("%-30s %7.2fs %6.2f ms/step  utilization %5.1f%%  process peak memory %d MiB"
          % (tag, el, el / steps * 1e3, u, mem))
    return el


STEPS = int(os.environ.get("STEPS", "1500"))
EVERY = int(os.environ.get("EVERY", "4"))
print("device %s   STEPS=%d   every=%d" % (torch.cuda.get_device_name(0), STEPS, EVERY))

scn = scenes.build_scene("magnus", n=40)
sco = old.build_scene("magnus", n=40)
bench("numpy 1 truth",
      lambda: old.rk4([*sco["known"].terms, *sco["unknown"].terms], sco["ext"], sco["X0"], sco["V0"],
                      STEPS * 2.5e-4, 2.5e-4, every=EVERY), STEPS)

X1, V1 = scn["X0"][None], scn["V0"][None]
t1 = [*scn["known"].terms, *scn["unknown"].terms]
W = lambda **kw: (lambda: sim.rk4(t1, scn["ext"], X1, V1, 20 * 2.5e-4, 2.5e-4, every=EVERY, **kw))
bench("GPU eager 1 truth",
      lambda: sim.rk4(t1, scn["ext"], X1, V1, STEPS * 2.5e-4, 2.5e-4, every=EVERY, graph=False),
      STEPS, warm=W(graph=False))

XB = scn["X0"][None].repeat(6, 1, 1)
VB = scn["V0"][None].repeat(6, 1, 1)
tb = [*scn["known"].terms,
      Mix(["magnus", "drag", "vlong", "quartic_r", "yukawa", "coulomb"],
          [0.07, 0.55, 0.35, 0.18, 0.0024, 0.0019])]
WB = lambda **kw: (lambda: sim.rk4(tb, scn["ext"], XB, VB, 20 * 2.5e-4, 2.5e-4, every=EVERY, **kw))
bench("GPU eager 6 truths (batched)",
      lambda: sim.rk4(tb, scn["ext"], XB, VB, STEPS * 2.5e-4, 2.5e-4, every=EVERY, graph=False),
      STEPS, warm=WB(graph=False))
bench("GPU graph 1 truth",
      lambda: sim.rk4(t1, scn["ext"], X1, V1, STEPS * 2.5e-4, 2.5e-4, every=EVERY, graph=True),
      STEPS, warm=W(graph=True))
bench("GPU graph 6 truths (batched)",
      lambda: sim.rk4(tb, scn["ext"], XB, VB, STEPS * 2.5e-4, 2.5e-4, every=EVERY, graph=True),
      STEPS, warm=WB(graph=True))

rt.use("cpu", torch.float64)
bench("torch CPU 1 truth",
      lambda: sim.rk4(t1, scn["ext"], scn["X0"].cpu(), scn["V0"].cpu(),
                      STEPS * 2.5e-4, 2.5e-4, every=EVERY, graph=False), STEPS)
rt.use("cuda", torch.float64)
