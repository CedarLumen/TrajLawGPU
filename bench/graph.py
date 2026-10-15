"""Can CUDA Graph replay hide the RK4 launch overhead? Measure first, then decide whether it goes into the core."""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tl import rt, scenes, sim  # noqa: E402
from tl.forces import Mix  # noqa: E402

STEPS = int(os.environ.get("STEPS", "1500"))
scn = scenes.build_scene("magnus", n=40)


def make_step(terms, ext, dt):
    def step(X, V, Xn, Vn):
        k1x, k1v = V, sim.accel(terms, ext, X, V)
        k2x = V + 0.5 * dt * k1v
        k2v = sim.accel(terms, ext, X + 0.5 * dt * k1x, V + 0.5 * dt * k1v)
        k3x = V + 0.5 * dt * k2v
        k3v = sim.accel(terms, ext, X + 0.5 * dt * k2x, V + 0.5 * dt * k2v)
        k4x = V + dt * k3v
        k4v = sim.accel(terms, ext, X + dt * k3x, V + dt * k3v)
        Xn.copy_(X + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x))
        Vn.copy_(V + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))
    return step


def run_graph(terms, ext, X0, V0, steps, every, dt, chunk=64):
    X = X0.clone().cuda()
    V = V0.clone().cuda()
    Xn = torch.empty_like(X)
    Vn = torch.empty_like(V)
    step = make_step(terms, ext, dt)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step(X, V, Xn, Vn)
            X.copy_(Xn)
            V.copy_(Vn)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    X.copy_(X0.cuda())
    V.copy_(V0.cuda())

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step(X, V, Xn, Vn)
        X.copy_(Xn)
        V.copy_(Vn)

    nf = steps // every + 1
    Xs = torch.empty((nf,) + tuple(X.shape), dtype=X.dtype, device=X.device)
    j = 0
    t0 = time.time()
    for i in range(steps + 1):
        if i % every == 0:
            Xs[j].copy_(X)
            j += 1
        if i == steps:
            break
        g.replay()
    rt.sync()
    return time.time() - t0, Xs


def main():
    dt = 2.5e-4
    for tag, terms in (("1 truth", [*scn["known"].terms, *scn["unknown"].terms]),
                       ("6 truths batched", [*scn["known"].terms,
                                     Mix(["magnus", "drag", "vlong", "quartic_r", "yukawa", "coulomb"],
                                         [0.07, 0.55, 0.35, 0.18, 0.0024, 0.0019])])):
        for B in ((1,) if tag == "1 truth" else (6,)):
            X0 = scn["X0"][None].repeat(B, 1, 1)
            V0 = scn["V0"][None].repeat(B, 1, 1)
            t, Xs = run_graph(terms, scn["ext"], X0, V0, STEPS, 4, dt)
            print("%-14s B=%d  graph replay %d steps  %6.3fs  %6.3f ms/step"
                  % (tag, B, STEPS, t, t / STEPS * 1e3))
            Xr, Vr = sim.rk4(terms, scn["ext"], X0, V0, STEPS * dt, dt, every=4)
            d = float((Xs - Xr).abs().max())
            print("               difference from the eager trajectory %.3g" % d)


if __name__ == "__main__":
    main()
