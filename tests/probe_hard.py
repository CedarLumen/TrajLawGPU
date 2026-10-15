"""Why the hard-regime A1/A2 law error drops to 1e-16: print every intermediate of the design matrix and STLSQ."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tl import bases, est, fit, obs, rt, scenes, sim  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = []


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
    Xs, Vs = sim.rk4(terms, ext, sc["X0"], sc["V0"], 6.0, cfg["dt"], every=4, graph=False)
    dt_obs = cfg["dt"] * 4
    a_est = est.est_sg(Xs, dt_obs, 4, 4)
    a_ext = sim.accel([], ext, Xs, Vs)
    a_true = sim.accel(terms, ext, Xs, Vs)
    lo, hi = 12, Xs.shape[0] - 12
    sub = np.arange(lo, hi, 16)
    Rsub = Xs[sub][:, :, None, :] - Xs[sub][:, None, :, :]
    Vsub = Vs[sub][:, :, None, :] - Vs[sub][:, None, :, :]
    a_known_sub = known_accel(known.terms, [], Xs[sub], Vs[sub])
    A_unk = bases.pair_sum_design(Rsub, Vsub, bases.VEL)
    A_bond = bases.pair_sum_design(Rsub, Vsub, bases.POS[:2], mask=adj)

    idx = np.arange(0, Xs.shape[0], 32)
    Re = Xs[idx][:, :, None, :] - Xs[idx][:, None, :, :]
    Ve = Vs[idx][:, :, None, :] - Vs[idx][:, None, :, :]
    f_true = sc["unknown"].f(Re, Ve)
    offd = ~torch.eye(N, dtype=torch.bool, device=Re.device)
    den = float((f_true[:, offd, :] ** 2).sum())

    def lawerr(c, off):
        p = bases.eval_law(Re, Ve, c[off:off + bases.M])
        return float(((p - f_true)[:, offd, :] ** 2).sum() / den)

    log("truth = magnus. Every regime, with all design-matrix and fit intermediates printed.")
    log("A: target after pre-subtracting the external field (what E4 uses)   B: pre-subtracted true acceleration (control, no estimator bias)")
    log()
    log("%-6s %-7s %-6s %-8s %-10s %-10s %-10s %-10s %s"
        % ("regime", "rows", "rank", "cond", "selected |v|", "selected r", "bias RMS", "A law error", "B law error"))
    vel = list(bases.VEL)
    for reg in ("full", "mid", "hard"):
        vis = obs.observe(Xs, scenes.make_cfg(reg), np.random.default_rng(scenes.RNG_SEED))[1]
        g = np.zeros(tuple(Xs.shape[:2]), bool)
        g[lo:hi] = vis[lo:hi]
        sel = np.repeat(g[sub].reshape(-1), 2)
        A = torch.cat([A_bond, A_unk], dim=1)
        Ad = obs.np_(A)[sel]
        yA = obs.np_((a_est - a_ext)[sub]).reshape(-1)[sel]
        yB = obs.np_((a_true - a_ext)[sub]).reshape(-1)[sel]
        sv = np.linalg.svd(Ad, compute_uv=False)
        rank = int((sv > sv[0] * 1e-12).sum())
        cA = fit.stlsq(Ad, yA)
        cB = fit.stlsq(Ad, yB)
        ins = float(np.mean((Ad @ cA - yA) ** 2) / np.var(yA))
        gsub = g[sub]
        vv = Vs[sub][gsub].norm(dim=-1)
        rr = Xs[sub][gsub].norm(dim=-1)
        bias = yA - yB
        log("%-6s %-7d %-6d %-8.3g %-10.4g %-10.4g %-10.3g %-10.4g %-10.4g"
            % (reg, int(sel.sum()), rank, float(sv[0] / sv[-1]),
               float(vv.mean()), float(rr.mean()), float(np.sqrt(np.mean(bias ** 2))),
               lawerr(torch.as_tensor(cA, device=Re.device, dtype=torch.float64), 2),
               lawerr(torch.as_tensor(cB, device=Re.device, dtype=torch.float64), 2)))
        cN = cA[2:]
        log("        regime-A coefficients (truth v⊥R̂=+0.07, v∥n̂=−0.07, rest 0):"
            + "  ".join("%s=%+.3g" % (bases.NAMES[i], cN[i]) for i in np.flatnonzero(np.abs(cN) > 1e-9)))

    log()
    log("control: the same three regimes with only the unknown-term basis (no bond block)")
    for reg in ("full", "mid", "hard"):
        vis = obs.observe(Xs, scenes.make_cfg(reg), np.random.default_rng(scenes.RNG_SEED))[1]
        g = np.zeros(tuple(Xs.shape[:2]), bool)
        g[lo:hi] = vis[lo:hi]
        sel = np.repeat(g[sub].reshape(-1), 2)
        yB = (obs.np_((a_true - a_ext)[sub]).reshape(-1)
              - obs.np_(a_known_sub).reshape(-1))[sel]
        Ad = obs.np_(A_unk)[sel]
        sv = np.linalg.svd(Ad, compute_uv=False)
        c = fit.stlsq(Ad, yB)
        log("  %-6s rows %-7d cond %-10.3g law error under the true target %.4g"
            % (reg, int(sel.sum()), float(sv[0] / sv[-1]),
               lawerr(torch.as_tensor(c, device=Re.device, dtype=torch.float64), 0)))

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "HARD.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
