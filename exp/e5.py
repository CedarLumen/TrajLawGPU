import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tl import bases, est, fit, obs, rt, scenes, sim  # noqa: E402
from tl.forces import Mix  # noqa: E402

OUT = []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


def main():
    cfg = scenes.make_cfg("full")
    dt = cfg["dt"]
    sc = scenes.build_scene("magnus", n=40)
    N = sc["N"]
    known, ext = sc["known"], sc["ext"]

    # 4 ensembles stacked on the batch dim: magnus s=1,2 and quartic_r s=1,2
    kinds = ["magnus", "magnus", "quartic_r", "quartic_r"]
    amps = [0.07, 0.07, 0.18, 0.18]
    vscale = [1.0, 2.0, 1.0, 2.0]
    X0 = sc["X0"][None].repeat(4, 1, 1)
    V0 = sc["V0"][None] * rt.t(np.array(vscale))[:, None, None]
    terms = [*known.terms, Mix(kinds, amps)]

    log("=" * 104)
    log("E5 migration of spurious laws under velocity dependence    N=40  T=6s  obs Δt=1ms  σ_x=0  regime=mid")
    log("=" * 104)

    t0 = time.time()
    Xs, Vs = sim.rk4(terms, ext, X0, V0, 6.0, dt, every=1)
    rt.sync()
    log("4 ensembles in one batched integration: %d frames x 4, %.1fs" % (Xs.shape[0], time.time() - t0))

    Xd, Vd = Xs[::4], Vs[::4]
    dt_obs = dt * 4
    cfg_mid = scenes.make_cfg("mid")

    data = []
    for b in range(4):
        Xb, Vb = Xd[:, b], Vd[:, b]
        _, vis = obs.observe(Xb, cfg_mid, np.random.default_rng(scenes.RNG_SEED))
        a_known = sim.pair_accel(known.terms, Xb, Vb) + sim.accel([], ext, Xb, Vb)
        resid = est.est_sg(Xb, dt_obs, 4, 4) - a_known
        lo, hi = 12, Xb.shape[0] - 12
        sub = np.arange(lo, hi, 8)
        good = np.zeros(tuple(Xb.shape[:2]), bool)
        good[lo:hi] = vis[lo:hi]
        R = Xb[sub][:, :, None, :] - Xb[sub][:, None, :, :]
        V = Vb[sub][:, :, None, :] - Vb[sub][:, None, :, :]
        data.append(dict(R=R, V=V, y=obs.np_(resid[sub]).reshape(-1),
                         ok=np.repeat(good[sub].reshape(-1), 2),
                         vmax=float(Vs[:, b].abs().max())))

    for variant, ia, ib in (("magnus", 0, 1), ("quartic_r", 2, 3)):
        A, B = data[ia], data[ib]
        tag = "velocity-dependent" if variant in scenes.VELOCITY_DEPENDENT else "position-dependent (control)"
        unk = sc["unknown"] if variant == "magnus" else scenes.make_unknown("quartic_r", 0.18)
        log()
        log("[truth = %s (%s)] |v|max of the two ensembles: s=1.0 -> %.3f, s=2.0 -> %.3f"
            % (variant, tag, A["vmax"], B["vmax"]))
        log()
        log("  %-12s %-6s %-16s %-16s %-16s %s"
            % ("ansatz", "terms", "pair-sum residual (in-sample)", "pairwise law error", "migration ‖Δf‖²/‖f‖²",
               "velocity coefficients (v⊥R̂ / v∥n̂)"))
        log("  " + "-" * 100)

        R, V = A["R"], A["V"]
        f_true = unk.f(R, V)
        offd = ~torch.eye(N, dtype=torch.bool, device=R.device)
        den = float((f_true[:, offd, :] ** 2).sum())

        for name, cols in (("position ansatz", bases.POS), ("velocity ansatz", bases.VEL)):
            fA = fB = None
            resA = coefA = err = None
            for D, which in ((A, "A"), (B, "B")):
                Mo = obs.np_(bases.pair_sum_design(D["R"], D["V"], cols))[D["ok"]]
                yo = D["y"][D["ok"]]
                c = fit.stlsq(Mo, yo)
                ins = float(np.mean((Mo @ c - yo) ** 2) / np.var(yo))
                ev = bases.eval_law(R, V, c, cols)
                err = float(((ev - f_true)[:, offd, :] ** 2).sum() / den)
                if which == "A":
                    fA, resA, coefA = ev, ins, c
                else:
                    fB = ev
            mig = float(((fA - fB)[:, offd, :] ** 2).sum() / (fA[:, offd, :] ** 2).sum())
            cv = "%+.4f / %+.4f" % (coefA[6], coefA[7]) if len(cols) > 4 else ""
            log("  %-12s %-6d %-16.4g %-16.4g %-16.4g %s"
                % (name, len(cols), resA, max(err, 0), mig, cv))
        log("  reference: the true magnus is +0.070 / −0.070 on (v⊥R̂, v∥n̂); quartic_r has no velocity term.")

    log()
    log("=" * 104)
    log("criterion check:")
    log("  - velocity-dependent truth + position ansatz  -> migration should be well above 0 (the law drifts with the velocity distribution)")
    log("  - velocity-dependent truth + velocity ansatz  -> migration ≈ 0, with velocity coefficients hitting ±0.070")
    log("  - position-dependent truth (control)          -> both ansatz should give migration ≈ 0")
    log("    If the control fails, this experiment is measuring something other than velocity dependence.")

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "E5.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
