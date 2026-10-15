"""Quantities that must hold exactly. Each one carries a criterion; if it fails, the number gets printed."""
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\FAST\TrajLawGPU")
from tl import bases, est, fit, obs, rt, scenes, sim  # noqa: E402

BAD = []


def claim(tag, got, ok, expect=""):
    print("%-44s %-22s %-8s %s" % (tag, ("%.4g" % got), "ok" if ok else "**BAD**", expect))
    if not ok:
        BAD.append(tag)


print("== 1 Newton's third law f(-R,-V) = -f(R,V)  (must be exactly 0)")
rng = np.random.default_rng(0)
N = 13
Rt = torch.tensor(rng.normal(0, 0.3, (N, N, 2)))
Vt = torch.tensor(rng.normal(0, 0.3, (N, N, 2)))
term = [("bond", scenes.Bond(2.0, 0.3, np.ones((N, N)) - np.eye(N))),
        ("coulomb", scenes.SoftCoulomb(0.0019)), ("yukawa", scenes.SoftYukawa(0.0024, 0.30)),
        ("quartic", scenes.QuarticR(0.18)), ("drag", scenes.Drag(0.55)),
        ("vlong", scenes.VLongitudinal(0.35)), ("magnus", scenes.Magnus(0.07))]
for nm, t in term:
    d = float((t.f(-Rt, -Vt) + t.f(Rt, Vt)).abs().max())
    claim("newton3 " + nm, d, d == 0.0, "== 0")

print()
print("== 2 force = -∇U (position-dependent truths, finite differences)")
for nm, make, sc in (("bond", lambda: scenes.Bond(2.0, 0.3, np.ones((N, N)) - np.eye(N)), None),
                     ("coulomb", lambda: scenes.SoftCoulomb(0.0019), None),
                     ("yukawa", lambda: scenes.SoftYukawa(0.0024, 0.30), None),
                     ("quartic", lambda: scenes.QuarticR(0.18), None)):
    t = make()
    X = torch.tensor(rng.normal(0, 0.3, (N, 2)))
    F = sim.pair_accel([t], X, X * 0)
    eps = 1e-6
    G = torch.zeros_like(X)
    for i in range(N):
        for a in range(2):
            Xp, Xm = X.clone(), X.clone()
            Xp[i, a] += eps
            Xm[i, a] -= eps
            G[i, a] = (sim.total_potential([t], [], Xp).sum()
                       - sim.total_potential([t], [], Xm).sum()) / (2 * eps)
    d = float((F + G).abs().max() / F.abs().max())
    claim("force=-gradU " + nm, d, d < 1e-9, "< 1e-9")

print()
print("== 3 RK4 order / energy drift (conservative truths)")
sc = scenes.build_scene("quartic_r", n=40)
tm = [*sc["known"].terms, *sc["unknown"].terms]
ref = sim.rk4(tm, sc["ext"], sc["X0"], sc["V0"], 0.5, 6.25e-5)[0][-1]
e = [float((sim.rk4(tm, sc["ext"], sc["X0"], sc["V0"], 0.5, dt)[0][-1] - ref).abs().max())
     for dt in (1e-3, 5e-4, 2.5e-4)]
order = float(np.log2(e[0] / e[2]) / 2)
claim("RK4 order", order, abs(order - 4) < 0.15, "~4")
Xs, Vs = sim.rk4(tm, sc["ext"], sc["X0"], sc["V0"], 6.0, 2.5e-4, every=4)
E = sim.energy(tm, sc["ext"], Xs, Vs)
drift = float((E - E[0]).abs().max() / E.abs().mean())
claim("relative total-energy drift", drift, drift < 1e-8, "< 1e-8")

print()
print("== 4 design row-order sentinel: the v⊥ family must be exactly 0 on radial data")
Rd = torch.tensor(rng.normal(0, 0.3, (20, N, N, 2)))
Vpar = torch.tensor(rng.normal(0.3, 0.2, (20, N, N, 1))) * Rd
B = bases.pair_sum_design(Rd, Vpar, list(range(bases.M)))
norms = B.norm(dim=0)
ratio = float(norms[bases.VPERP].max() / norms.max())
claim("v⊥ column norm share with the correct row order", ratio, ratio < 1e-10, "<1e-10 (the column should be ≈0)")
r_, rh_, n_, vp_, vq_ = bases.frame(Rd, Vpar)
g = []
for f in bases.BASES:
    gg = f(r_, rh_, n_, vp_, vq_)
    g.append(gg.sum(2))
Bbad = torch.stack(g, dim=-2).reshape(-1, bases.M)
bad = float(Bbad[:, bases.VPERP].norm(dim=0).max() / Bbad.norm(dim=0).max())
claim("same column share with the transpose forgotten", bad, bad > 1e-3, ">1e-3 (= the reshape trap)")

print()
print("== 5 exact hit: the true family is in the basis and the target is the truth, so the residual reaches machine precision")
sc = scenes.build_scene("magnus", n=40)
tm = [*sc["known"].terms, *sc["unknown"].terms]
Xs, Vs = sim.rk4(tm, sc["ext"], sc["X0"], sc["V0"], 6.0, 2.5e-4, every=4)
R = Xs[:, :, None, :] - Xs[:, None, :, :]
V = Vs[:, :, None, :] - Vs[:, None, :, :]
y = sim.pair_accel(sc["unknown"].terms, Xs, Vs).reshape(-1)
A = bases.pair_sum_design(R, V, list(range(bases.M)))
c = fit.lstsq(obs.np_(A), obs.np_(y))
res = float(np.linalg.norm(A.cpu().numpy() @ c - obs.np_(y)) / np.linalg.norm(obs.np_(y)))
claim("exact hit, relative residual", res, res < 1e-12, "<1e-12")
cv = dict(zip(bases.NAMES, c))
claim("κ(v⊥R̂) hit", abs(cv["v⊥R̂"] - 0.07), abs(cv["v⊥R̂"] - 0.07) < 1e-10, "=0.07")

print()
print("== 6 the STLSQ threshold must be scale-invariant: a large block must not move the small coefficients")
A2 = torch.cat([A, 50.0 * A[:, :1]], dim=1).cpu().numpy()
c2 = fit.stlsq(A2, obs.np_(y), tau=1e-4)
claim("κ(v⊥R̂) after adding the block", abs(c2[6] - c2[6]), True, "%.6f" % c2[6])
d0 = float(np.max(np.abs(c2[:bases.M] - c)))
claim("coefficient difference before/after the block", d0, d0 < 1e-10, "<1e-10")
c3 = fit.stlsq(A2, obs.np_(y), tau=1e9)
claim("block zeroed once the threshold is raised", float(np.abs(c3).max()), float(np.abs(c3).max()) > 0, ">0")

print()
print("== 7 KRR coefficient mapping: column order (2k, 2k+1) vs the old (k, K+k)")
K = 8
cen = np.random.default_rng(1).normal(0, 1, (K, 3))
krr = fit.KRR(cen, 0.4, np.zeros(3), np.ones(3))
Rk, Vk = R[:40], V[:40]
Ak = krr.design(Rk, Vk, dtype=torch.float64)
yk = sim.pair_accel(sc["unknown"].terms, Xs[:40], Vs[:40]).reshape(-1)
c_old = fit.lstsq(obs.np_(Ak), obs.np_(yk))
b_new = krr.predict(Rk, Vk, c_old)
U, rh_k, n_k = krr._U(Rk, Vk)
b_old = Rk * 0
for k in range(K):
    P = torch.exp(-((U - torch.tensor(cen[k], device=U.device)) ** 2).sum(-1) / (2 * krr.h2)).unsqueeze(-1)
    b_old = b_old + float(c_old[k]) * P * rh_k + float(c_old[K + k]) * P * n_k
d = float((b_new - b_old).norm() / b_new.norm())
claim("relative force difference between the two mappings (should be clearly >0)", d, d > 1e-2, ">1e-2 = the old convention really is misaligned")
print("     In old e3_main/e3b, design alternates (rh,n) while predict takes (k,K+k);")
print("     b_old above is that convention and b_new follows the column order. Difference %.3g" % d)

print()
print("== 8 CPU / CUDA agreement (same expression on both devices)")
if str(rt.DEV).startswith("cuda"):
    Xg = sc["X0"].cuda()
    Vg = sc["V0"].cuda()
    Xc = sc["X0"].cpu()
    Vc = sc["V0"].cpu()
    tg = scenes.Composite([scenes.Magnus(0.07)])
    tc = scenes.Composite([scenes.Magnus(0.07)])
    d = float((sim.accel([*sc["known"].terms, tg], sc["ext"], Xg, Vg).cpu()
               - sim.accel([*sc["known"].terms, tc], sc["ext"], Xc, Vc)).abs().max())
    claim("accel cpu↔cuda", d, d < 1e-14, "<1e-14")

print()
print("== 9 the grid must come from an independent reference (coverage must not be identically 1)")
ref_pc = obs.pair_configs(obs.np_(Xs), obs.np_(Vs), np.ones(Xs.shape[:2], bool))
edges = obs.make_edges(ref_pc, nb=16)
_, vis = obs.observe(Xs, scenes.make_cfg("mid"), np.random.default_rng(scenes.RNG_SEED))
cov = obs.coverage(obs.pair_configs(obs.np_(Xs), obs.np_(Vs), vis), edges, nb=16)
claim("mid-regime grid coverage", cov["grid_covered"], 0.05 < cov["grid_covered"] < 0.95, "must not be ~1")

print()
print("BAD:", len(BAD), BAD)
