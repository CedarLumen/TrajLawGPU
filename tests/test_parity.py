"""Term-by-term equivalence check of the new core against the old numpy implementation, kept as reference in D:\\FAST\\TrajLaw\\sim."""
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\FAST\TrajLaw\sim")
sys.path.insert(0, r"D:\FAST\TrajLawGPU")
import trajlaw_pairsim as old  # noqa: E402
import e2_accel_estimator as o2  # noqa: E402
import e3_main as o3  # noqa: E402

from tl import bases, est, obs, scenes, sim  # noqa: E402

FAIL = []


def chk(tag, a, b, tol=1e-12):
    a = obs.np_(a).astype(np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        FAIL.append((tag, "shape"))
        print("%-46s SHAPE %s vs %s" % (tag, a.shape, b.shape))
        return
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        FAIL.append((tag, "nanmask %d" % int((np.isnan(a) != np.isnan(b)).sum())))
        print("%-46s NAN-MASK differs" % tag)
        return
    aa, bb = np.nan_to_num(a), np.nan_to_num(b)
    d = np.max(np.abs(aa - bb)) if aa.size else 0.0
    sc = max(np.max(np.abs(bb)) if bb.size else 0.0, 1e-30)
    ok = d <= tol * sc + tol
    if not ok:
        FAIL.append((tag, d, sc))
    print("%-46s max|d|=%-12.3g rel=%-12.3g %s" % (tag, d, d / sc, "ok" if ok else "**FAIL**"))


rng = np.random.default_rng(3)
N, d = 11, 2
X = rng.normal(0, 0.35, (N, d))
V = rng.normal(0, 0.35, (N, d))
R = X[:, None, :] - X[None, :, :]
Vv = V[:, None, :] - V[None, :, :]
Rt, Vt = torch.tensor(R), torch.tensor(Vv)

pairs = [("bond", old.Bond(2.0, 0.3, np.ones((N, N)) - np.eye(N)), scenes.Bond(2.0, 0.3, np.ones((N, N)) - np.eye(N))),
         ("coulomb", old.SoftCoulomb(0.0019), scenes.SoftCoulomb(0.0019)),
         ("yukawa", old.SoftYukawa(0.0024, 0.30), scenes.SoftYukawa(0.0024, 0.30)),
         ("quartic", old.QuarticR(0.18), scenes.QuarticR(0.18)),
         ("drag", old.Drag(0.55), scenes.Drag(0.55)),
         ("vlong", old.VLongitudinal(0.35), scenes.VLongitudinal(0.35)),
         ("magnus", old.Magnus(0.07), scenes.Magnus(0.07))]
for nm, o, n in pairs:
    chk("force " + nm, n.f(Rt, Vt), o.force(R, Vv))
for nm, o, n in pairs[:4]:
    chk("potential " + nm, n.u(Rt), o.potential(R))

sc_o = old.build_scene("magnus", n=40)
sc_n = scenes.build_scene("magnus", n=40)
chk("scene X0", sc_n["X0"], sc_o["X0"])
chk("scene V0", sc_n["V0"], sc_o["V0"])
chk("scene adj", sc_n["adj"], sc_o["adj"])

to = [*sc_o["known"].terms, *sc_o["unknown"].terms]
tn = [*sc_n["known"].terms, *sc_n["unknown"].terms]
chk("accel (N,d)", sim.accel(tn, sc_n["ext"], sc_n["X0"], sc_n["V0"]),
    old.accel(to, sc_o["ext"], sc_o["X0"], sc_o["V0"]))
chk("pair_accel (N,d)", sim.pair_accel(tn, sc_n["X0"], sc_n["V0"]),
    old.pair_accel(to, sc_o["X0"], sc_o["V0"]))
chk("total_potential", sim.total_potential(sc_n["known"].terms, sc_n["ext"], sc_n["X0"]),
    old.total_potential(sc_o["known"].terms, sc_o["ext"], sc_o["X0"]))

for var in ("magnus", "quartic_r", "yukawa"):
    so = old.build_scene(var, n=40)
    sn = scenes.build_scene(var, n=40)
    o_t = [*so["known"].terms, *so["unknown"].terms]
    n_t = [*sn["known"].terms, *sn["unknown"].terms]
    Xo, Vo = old.rk4(o_t, so["ext"], so["X0"], so["V0"], 0.5, 2.5e-4, every=4)
    Xn, Vn = sim.rk4(n_t, sn["ext"], sn["X0"], sn["V0"], 0.5, 2.5e-4, every=4)
    chk("rk4 X " + var, Xn, Xo)
    chk("rk4 V " + var, Vn, Vo)

Xo = obs.np_(Xn).astype(np.float64)
for reg in ("full", "mid", "hard", "random"):
    cfg = scenes.make_cfg(reg)
    o_obs, o_vis = old.observe(Xo, cfg, np.random.default_rng(11))
    n_obs, n_vis = obs.observe(Xn, cfg, np.random.default_rng(11))
    chk("vis " + reg, n_vis, o_vis)
    chk("X_obs " + reg, n_obs, o_obs)

S = torch.tensor(old.rk4(to, sc_o["ext"], sc_o["X0"], sc_o["V0"], 4.0, 2.5e-4, every=1)[0])
for m, p in ((4, 4), (2, 3), (6, 3), (8, 4)):
    chk("est_sg(%d,%d)" % (m, p), est.est_sg(S, 1e-3, m, p), o2.est_sg(obs.np_(S), 1e-3, m, p), 1e-8)
chk("est_fd", est.est_fd(S, 1e-3), o2.est_fd(obs.np_(S), 1e-3))
chk("est_smooth_fd", est.est_smooth_fd(S, 1e-3, 4), o2.est_smooth_fd(obs.np_(S), 1e-3, 4), 1e-8)

Xs_o, Vs_o = old.rk4(to, sc_o["ext"], sc_o["X0"], sc_o["V0"], 1.0, 2.5e-4, every=4)
Rn = torch.tensor(Xs_o)[:, :, None, :] - torch.tensor(Xs_o)[:, None, :, :]
Vn2 = torch.tensor(Vs_o)[:, :, None, :] - torch.tensor(Vs_o)[:, None, :, :]
chk("pair_sum_design full10", bases.pair_sum_design(Rn, Vn2, list(range(10))),
    o3.pair_sum_design(Xs_o[:, :, None, :] - Xs_o[:, None, :, :],
                       Vs_o[:, :, None, :] - Vs_o[:, None, :, :]))

adj = sc_o["adj"].astype(np.float32)
Ro = Xs_o[:, :, None, :] - Xs_o[:, None, :, :]
Vo = Vs_o[:, :, None, :] - Vs_o[:, None, :, :]
ref = []
for m in (0, 1):
    g = o3.BASES[m][1](*o3.frame(Ro, Vo)) * adj[..., None]
    ii = np.arange(Xs_o.shape[1])
    g[:, ii, ii] = 0.0
    ref.append(g.sum(axis=2))
chk("pair_sum_design masked-bond",
    bases.pair_sum_design(Rn, Vn2, [0, 1], mask=torch.tensor(adj)),
    np.stack(ref, axis=-2).transpose(0, 1, 3, 2).reshape(-1, 2))

chk("pair_configs r", obs.pair_configs(Xs_o, Vs_o, np.ones(Xs_o.shape[:2], bool))["r"],
    old.pair_configs(Xs_o, Vs_o, np.ones(Xs_o.shape[:2], bool))["r"])

print()
print("== the three rk4 paths (old numpy / new eager / new CUDA Graph) must agree for every value of every ==")
for var in ("magnus", "quartic_r", "drag"):
    so = old.build_scene(var, n=40)
    sn = scenes.build_scene(var, n=40)
    o_t = [*so["known"].terms, *so["unknown"].terms]
    n_t = [*sn["known"].terms, *sn["unknown"].terms]
    for ev in (1, 4, 5, 32, 64):
        Xo, Vo = old.rk4(o_t, so["ext"], so["X0"], so["V0"], 1.0, 2.5e-4, every=ev)
        Xg, Vg = sim.rk4(n_t, sn["ext"], sn["X0"], sn["V0"], 1.0, 2.5e-4, every=ev, graph=True)
        Xe, Ve = sim.rk4(n_t, sn["ext"], sn["X0"], sn["V0"], 1.0, 2.5e-4, every=ev, graph=False)
        chk("rk4 %s every=%d old-vs-graph" % (var, ev), Xg, Xo)
        chk("rk4 %s every=%d old-vs-eager" % (var, ev), Xe, Xo)

print()
print("== does the batch wrapping Mix / Triplet agree with the single-item classes and the old f3 ==")
from tl import forces as F  # noqa: E402
import e6_pairwise_falsify as o6  # noqa: E402

kinds = ["magnus", "drag", "vlong", "quartic_r", "yukawa", "coulomb"]
amps = [0.07, 0.55, 0.35, 0.18, 0.0024, 0.0019]
Rb = torch.tensor(R)[None].repeat(6, 1, 1, 1)
Vb = torch.tensor(Vv)[None].repeat(6, 1, 1, 1)
mx = F.Mix(kinds, amps)
gm = mx.f(Rb, Vb)
oldmap = dict((nm, o) for nm, o, _ in pairs)
for i, k in enumerate(kinds):
    chk("Mix[" + k + "]", gm[i], oldmap[{"quartic_r": "quartic"}.get(k, k)].force(R, Vv))

Xb = torch.tensor(sc_o["X0"])[None].repeat(3, 1, 1)
tp = F.Triplet([0.0, 0.02, 0.05])
gt = tp.f(Xb)
for i, lam in enumerate([0.0, 0.02, 0.05]):
    chk("Triplet lam=%.2f" % lam, gt[i], o6.f3(sc_o["X0"], lam, 0.15))

print()
print("FAILED:", len(FAIL))
for f in FAIL:
    print("   ", f)
