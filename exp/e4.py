"""E4 corrected: known-term ablation (A0/A1/A2 x full/mid/hard) plus a collinearity diagnostic.

The old E4 had two defects: the design matrix omitted the external field while the target
contained it (an identifiability clash), and the STLSQ threshold used 1e-3·max|c|, which is
not scale-invariant. Both are done the corrected way here.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tl import bases, est, fit, obs, rt, scenes, sim  # noqa: E402

OUT = []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


def known_accel(terms, ext, X, V, step=768):
    out = torch.empty_like(X)
    for i in range(0, len(X), step):
        sl = slice(i, min(i + step, len(X)))
        out[sl] = sim.pair_accel(terms, X[sl], V[sl]) + sim.accel([], ext, X[sl], V[sl])
    return out


def main():
    cfg = scenes.make_cfg("full")
    sc = scenes.build_scene("magnus", n=40)
    known, ext, N = sc["known"], sc["ext"], sc["N"]
    adj = rt.t(sc["adj"])
    terms = [*known.terms, *sc["unknown"].terms]
    log("truth = magnus (velocity-dependent)   N=%d  T=6s  σ_x=0  external = %s(w2=%.3g)"
        % (N, type(ext[0]).__name__, ext[0].w2))

    Xs, Vs = sim.rk4(terms, ext, sc["X0"], sc["V0"], 6.0, cfg["dt"], every=4, graph=False)
    dt_obs = cfg["dt"] * 4
    a_est = est.est_sg(Xs, dt_obs, 4, 4)
    a_ext = sim.accel([], ext, Xs, Vs)
    log("estimator = local polynomial (m=4,p=4)   observed frames %d" % Xs.shape[0])

    lo, hi = 12, Xs.shape[0] - 12
    sub = np.arange(lo, hi, 16)
    Rsub = Xs[sub][:, :, None, :] - Xs[sub][:, None, :, :]
    Vsub = Vs[sub][:, :, None, :] - Vs[sub][:, None, :, :]
    a_known_sub = known_accel(known.terms, [], Xs[sub], Vs[sub])

    A_unk = bases.pair_sum_design(Rsub, Vsub, bases.VEL)
    A_bond_nomask = bases.pair_sum_design(Rsub, Vsub, bases.POS[:2])
    A_bond_mask = bases.pair_sum_design(Rsub, Vsub, bases.POS[:2], mask=adj)
    del Rsub, Vsub

    xcol = Xs[sub].reshape(-1).to(torch.float64)
    rr = A_unk[:, bases.IDX["rR̂"]].to(torch.float64)
    cc = float(np.corrcoef(obs.np_(rr), obs.np_(xcol))[0, 1])
    proj = float(((xcol - rr * (xcol @ rr) / (rr @ rr)).norm() / xcol.norm()))
    log()
    log("1. collinearity diagnostic: the `rR̂` column of unk vs the x_i column of the self-field block")
    log("    corr = %.6f ; relative norm left after projecting x_i onto rR̂ = %.4g" % (cc, proj))
    log("    identity Σ_j r R̂_ij = N·x_i − Σ_j x_j — the two are the same quantity.")

    idx = np.arange(0, Xs.shape[0], 32)
    Re = Xs[idx][:, :, None, :] - Xs[idx][:, None, :, :]
    Ve = Vs[idx][:, :, None, :] - Vs[idx][:, None, :, :]
    f_unk_true = sc["unknown"].f(Re, Ve)
    offd = ~torch.eye(N, dtype=torch.bool, device=Re.device)
    den = float((f_unk_true[:, offd, :] ** 2).sum())

    y_ext = (a_est - a_ext)[sub].reshape(-1)
    y_ext_old = a_est[sub].reshape(-1)
    y_bond = (a_est - a_ext)[sub].reshape(-1) - a_known_sub.reshape(-1)

    vis = {}
    for reg in ("full", "mid", "hard"):
        vis[reg] = obs.observe(Xs, scenes.make_cfg(reg), np.random.default_rng(scenes.RNG_SEED))[1]

    def lawerr(c, off):
        p = bases.eval_law(Re, Ve, c[off:off + bases.M])
        return float(((p - f_unk_true)[:, offd, :] ** 2).sum() / den)

    log()
    log("2. known-term ablation (external field pre-subtracted + contribution-thresholded STLSQ)")
    log("  %-5s %-6s %-9s %-16s %-16s %-10s" %
        ("known level", "regime", "rows", "pair-sum residual", "pairwise law error", "degeneracy ratio"))
    log("  " + "-" * 78)
    res = {}
    for level in ("A0", "A1", "A2"):
        for reg in ("full", "mid", "hard"):
            g = np.zeros(tuple(Xs.shape[:2]), bool)
            g[lo:hi] = vis[reg][lo:hi]
            sel = np.repeat(g[sub].reshape(-1), 2)
            if level == "A0":
                A = torch.cat([A_bond_nomask, A_unk], dim=1)
                off, y = 2, y_ext
            elif level == "A1":
                A = torch.cat([A_bond_mask, A_unk], dim=1)
                off, y = 2, y_ext
            else:
                A, off, y = A_unk, 0, y_bond
            Ad, yd, sd = obs.np_(A)[sel], obs.np_(y)[sel], sel
            c = fit.stlsq(Ad, yd)
            ins = float(np.mean((Ad @ c - yd) ** 2) / np.var(yd))
            e = lawerr(torch.as_tensor(c, device=Re.device, dtype=torch.float64), off)
            res[(level, reg)] = e
            log("  %-5s %-6s %-9d %-16.4g %-16.4g %-10s"
                % (level, reg, int(sd.sum()), ins, e,
                   "%.3f" % (e / res[(level, "full")]) if reg != "full" else "1.000"))

    log()
    log("criterion check (scaffold hypothesis: the gain from the bond graph should grow as observations thin out)")
    for reg in ("full", "mid", "hard"):
        log("  regime %-5s: A0 %.4g -> A1 %.4g -> A2 %.4g    bond-graph gain %.3g x ; extra gain from parameters %.3g x"
            % (reg, res[("A0", reg)], res[("A1", reg)], res[("A2", reg)],
               res[("A0", reg)] / max(res[("A1", reg)], 1e-300),
               res[("A1", reg)] / max(res[("A2", reg)], 1e-300)))

    g = np.zeros(tuple(Xs.shape[:2]), bool)
    g[lo:hi] = vis["full"][lo:hi]
    sel = np.repeat(g[sub].reshape(-1), 2)
    Aold = torch.cat([A_bond_mask, A_unk], dim=1)
    c_old = fit.stlsq(obs.np_(Aold)[sel], obs.np_(y_ext_old)[sel])
    e_old = lawerr(torch.as_tensor(c_old, device=Re.device, dtype=torch.float64), 2)
    log()
    log("3. old-criterion reference (external field not subtracted from the target, A1/full only):")
    log("    old criterion %.4g -> after pre-subtracting the field %.4g    field leakage %.1f%%"
        % (e_old, res[("A1", "full")],
           100 * (e_old - res[("A1", "full")]) / max(e_old, 1e-300)))
    log("    peak GPU memory %.3g GB (torch side)" % rt.peak_gb())

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "E4.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
