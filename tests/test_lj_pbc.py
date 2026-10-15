"""Independent recomputation of PBC + LJ: the torch paths (eager / CUDA Graph) against a hand-written numpy RK4 and brute-force forces."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tl import obs, rt, sim  # noqa: E402
from tl.forces import LJ  # noqa: E402

N, D = 32, 3
BOX = 12.0
EPS, SIG = 1.0, 1.0
DT = 1e-4
NSTEP = 200
BAD = []


def chk(name, a, b, tol):
    d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    s = "OK " if d <= tol else "BAD"
    if d > tol:
        BAD.append(name)
    print("%s %-42s max diff %.3e   tol %.1e" % (s, name, d, tol))


def nforce(X):
    F = np.zeros_like(X)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r = X[i] - X[j]
            r = r - BOX * np.round(r / BOX)
            r2 = max(float(r @ r), 1e-12)
            f = 48.0 * EPS * SIG ** 12 / r2 ** 7 - 24.0 * EPS * SIG ** 6 / r2 ** 4
            F[i] += f * r
    return F


def nacc(X):
    return nforce(X)


def nrk4(X, V, n):
    for _ in range(n):
        k1x, k1v = V, nacc(X)
        k2x = V + 0.5 * DT * k1v
        k2v = nacc(X + 0.5 * DT * k1x)
        k3x = V + 0.5 * DT * k2v
        k3v = nacc(X + 0.5 * DT * k2x)
        k4x = V + DT * k3v
        k4v = nacc(X + DT * k3x)
        X = X + DT / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        V = V + DT / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return X, V


def main():
    rng = np.random.default_rng(3)
    X = rng.uniform(0.0, BOX, (N, D))
    V = rng.normal(0.0, 0.3, (N, D))
    lb = LJ(EPS, SIG)

    R = np.zeros((N, N, D))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r = X[i] - X[j]
            R[i, j] = r - BOX * np.round(r / BOX)
    Rt = torch.tensor(R).unsqueeze(0)
    chk("force: torch vs brute-force numpy", obs.np_(lb.f(Rt, Rt * 0)[0].sum(-2)), nforce(X), 1e-12)

    Xt, Vt = torch.tensor(X), torch.tensor(V)
    at = obs.np_(sim.accel([lb], [], Xt, Vt, box=BOX))
    chk("acceleration: torch(PBC) vs numpy", at, nacc(X), 1e-12)

    Xe, Ve = sim.rk4([lb], [], Xt, Vt, None, DT, steps=NSTEP, every=1, graph=False, box=BOX)
    Xg, Vg = sim.rk4([lb], [], Xt, Vt, None, DT, steps=NSTEP, every=1, graph=True, box=BOX)
    Xn, Vn = nrk4(X.copy(), V.copy(), NSTEP)
    chk("trajectory: torch eager vs numpy RK4", obs.np_(Xe[-1]), Xn, 1e-10)
    chk("trajectory: torch graph vs numpy RK4", obs.np_(Xg[-1]), Xn, 1e-10)
    chk("trajectory: graph vs eager", obs.np_(Xg[-1]), obs.np_(Xe[-1]), 1e-14)

    Xb, Vb = sim.rk4([lb], [], Xt, Vt, None, DT, steps=NSTEP, every=1, graph=False)
    print("displacement difference with the box removed (should be clearly non-zero): %.4g" %
          float(np.max(np.abs(obs.np_(Xe[-1]) - obs.np_(Xb[-1])))))

    print("\nfree-flight check: with all forces off, the displacement after 20 steps should equal V*t")
    zero = sim.rk4([], [], Xt, Vt, None, DT, steps=20, every=20, graph=False, box=BOX)
    chk("free flight x = x0 + v*20dt", obs.np_(zero[0][-1]), X + V * DT * 20, 1e-12)

    v_rms = float(np.sqrt((V ** 2).sum(-1).mean()))
    print("\ninitial v_rms = %.4g (code units); ballistic displacement over 200 steps with no potential = v_rms*t = %.4g"
          % (v_rms, v_rms * NSTEP * DT))
    m = obs.np_(((Xg - Xg[0]) ** 2).sum(-1).mean(-1))
    print("MSD(t) with LJ: step 1 %.3e, step 10 %.3e, step 100 %.3e, step %d %.3e"
          % (m[1], m[10], m[100], NSTEP, m[-1]))
    step_disp = obs.np_(((Xg[1:] - Xg[:-1]) ** 2).sum(-1).mean())
    print("mean squared displacement between adjacent frames = %.3e, i.e. per-step speed %.4g (should be of the order of v_rms)"
          % (step_disp, np.sqrt(step_disp) / DT))
    print("\nBAD: %s" % (BAD if BAD else "none"))


if __name__ == "__main__":
    main()
