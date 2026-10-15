import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tl import bases, est, fit, obs, rt, scenes, sim  # noqa: E402
from tl.forces import Triplet  # noqa: E402

OUT = []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


def main():
    cfg = scenes.make_cfg("full")
    dt = cfg["dt"]
    sc = scenes.build_scene("magnus", n=40)
    known, ext, N = sc["known"], sc["ext"], sc["N"]
    pair_terms = [*known.terms, *sc["unknown"].terms]
    A3 = 0.15
    lam_list = [0.0, 0.008, 0.02, 0.05]

    log("truth = magnus + triplet term λ3·Σφ(r_ij)φ(r_ik)φ(r_jk),  φ(s)=exp(−s/%.2f)" % A3)
    log("criteria (pre-registered): λ3=0 should give ≲1e-6; λ3>0 should match the triplet share and give a residual cosine ≳0.7")

    # Calibrate λ3 so the triplet force reaches a set fraction of the unknown term. Triplet takes f(X), so it goes in ext.
    Xs0, Vs0 = sim.rk4(pair_terms, [*ext, Triplet(0.0, a=A3)], sc["X0"], sc["V0"],
                       4 * dt, dt, every=1, graph=False)
    Xc = Xs0[-1]
    f_unk_ref = sim.pair_accel(sc["unknown"].terms, Xc, Vs0[-1])
    una = float((f_unk_ref ** 2).sum(-1).mean().sqrt())
    log()
    log("calibration: per-particle RMS of the unknown term = %.4g   triplet per-particle RMS vs λ3:" % una)
    for lam in lam_list:
        f3v = Triplet(lam, a=A3).f(Xc)
        rr = float((f3v ** 2).sum(-1).mean().sqrt())
        log("    λ3=%-6.3f RMS(f3) = %-10.4g  -> %.1f%% of the unknown term" % (lam, rr, 100 * rr / una))

    X0 = sc["X0"][None].repeat(len(lam_list), 1, 1)
    V0 = sc["V0"][None].repeat(len(lam_list), 1, 1)
    t0 = time.time()
    Xs, Vs = sim.rk4(pair_terms, [*ext, Triplet(lam_list, a=A3)], X0, V0, 6.0, dt, every=4)
    rt.sync()
    log()
    log("4 values of λ3 in one batched integration: %d frames x 4, %.1fs" % (Xs.shape[0], time.time() - t0))

    log()
    log("%-7s %-14s %-16s %-16s %-12s %s"
        % ("λ3", "triplet share of unknown force", "pair-sum residual", "pairwise law error", "residual / floor", "cosine with true triplet force"))
    log("-" * 104)
    cfg_mid = scenes.make_cfg("mid")
    floor = None
    res = {}
    for b, lam in enumerate(lam_list):
        Xd, Vd = Xs[:, b], Vs[:, b]
        a_est = est.est_sg(Xd, dt * 4, 4, 4)
        a_known = sim.pair_accel(known.terms, Xd, Vd) + sim.accel([], ext, Xd, Vd)
        lo, hi = 12, Xd.shape[0] - 12
        sub = np.arange(lo, hi, 16)
        Rsub = Xd[sub][:, :, None, :] - Xd[sub][:, None, :, :]
        Vsub = Vd[sub][:, :, None, :] - Vd[sub][:, None, :, :]
        A_unk = obs.np_(bases.pair_sum_design(Rsub, Vsub, bases.VEL))
        y = obs.np_((a_est - a_known)[sub]).reshape(-1)
        _, vis = obs.observe(Xd, cfg_mid, np.random.default_rng(scenes.RNG_SEED))
        good = np.zeros(tuple(Xd.shape[:2]), bool)
        good[lo:hi] = vis[lo:hi]
        sel = np.repeat(good[sub].reshape(-1), 2)

        c = fit.stlsq(A_unk[sel], y[sel])
        ins = float(np.mean((A_unk[sel] @ c - y[sel]) ** 2) / np.var(y[sel]))

        idx = np.arange(0, Xd.shape[0], 32)
        Re = Xd[idx][:, :, None, :] - Xd[idx][:, None, :, :]
        Ve = Vd[idx][:, :, None, :] - Vd[idx][:, None, :, :]
        f_unk_true = sc["unknown"].f(Re, Ve)
        f3p = Triplet(lam, a=A3).f(Xd[idx])
        # The pair residual resid[t,i,j] is the pair force on i from j, so the triplet force has to be
        # matched along rows (i). The old script wrote f3_true[j] = f3(...), broadcasting (N,d) onto
        # the columns of (N,N,d) — rows and columns swapped.
        f3_row = f3p[:, :, None, :].expand(-1, -1, N, -1)
        f3_col = f3p[:, None, :, :].expand(-1, N, -1, -1)
        offd = ~torch.eye(N, dtype=torch.bool, device=Re.device)
        pred = bases.eval_law(Re, Ve, c)
        resid = (pred - f_unk_true)[:, offd, :]
        den = float((f_unk_true[:, offd, :] ** 2).sum())
        err = float((resid ** 2).sum() / den)
        rn = float(resid.norm())

        def cos_with(f3a):
            r3 = f3a[:, offd, :]
            n3 = float(r3.norm())
            return (float((resid * r3).sum() / max(rn * n3, 1e-30)) if n3 > 0 else 0.0), n3
        cosm, n3 = cos_with(f3_row)
        cosm_col, _ = cos_with(f3_col)
        frac = n3 / float(f_unk_true[:, offd, :].norm()) if n3 > 0 else 0.0
        if lam == 0.0:
            floor = err
        res[lam] = dict(err=err, ins=ins, cosm=cosm, frac=frac, col=cosm_col)
        ratio = err / floor if floor is not None else float("nan")
        log("%-7.3f %-14s %-16.4g %-16.4g %-12s %+.4f" %
            (lam, ("%.1f%%" % (100 * frac)) if frac else "0 (baseline)", ins, err,
             "1.00" if lam == 0.0 else "%.3g" % ratio, cosm))
        if frac:
            log("        (matching the same residual along columns is the old convention, cosine %+.4f — opposite sign to the row above, see the verdict section)" % cosm_col)
        del A_unk, y, c, pred, f3_row, f3_col

    log()
    log("=" * 104)
    log("verdict (against the pre-registered criteria; the two conditions are reported separately, never merged into one sentence)")
    log("=" * 104)
    log("  condition A (power): λ3>0 law error / λ3=0 floor > 1e4")
    log("  condition B (specificity): cosine between the residual and the true triplet force > 0.7")
    log("  λ3=0 floor = %.4g (pre-registered: ≲1e-6 -> %s)"
        % (floor, "met" if floor < 1e-6 else "NOT met"))
    dmax = max(abs(abs(res[l]["cosm"]) - abs(res[l]["col"])) for l in lam_list[1:])
    log("  pairing check: max |cosine| difference between the two pairings (residual rows vs columns against the triplet force) = %.3g" % dmax)
    log("    The pair residual is antisymmetric in (i,j) (f_ij = −f_ji), so column pairing is exactly row")
    log("    pairing negated: same angle, opposite sign. The −0.15 in the old report is the column pairing;")
    log("    row pairing (correct) gives %+.4f." % res[lam_list[-1]]["cosm"])
    log()
    okA = okB = False
    for lam in lam_list[1:]:
        d = res[lam]
        r = d["err"] / max(floor, 1e-300)
        a = r > 1e4
        bb = d["cosm"] > 0.7
        okA = okA or a
        okB = okB or bb
        log("  λ3=%-6.3f triplet share %-9.1f%% law error %-10.4g (%.3g x floor) cosine %+.4f  A:%s B:%s"
            % (lam, 100 * d["frac"], d["err"], r, d["cosm"],
               "pass" if a else "FAIL", "pass" if bb else "FAIL"))
    log()
    log("  => condition A (power): %s" % ("holds — the residual is orders of magnitude above the floor, so the test has power" if okA else "does not hold"))
    log("  => condition B (specificity): %s" % ("holds" if okB else
        "does NOT hold — the residual does not align with the true triplet force, so it cannot be called the triplet term"))
    log()
    if okA and okB:
        log("  Both conditions pass => pairwise separability is falsifiable and the residual is attributable to the triplet term. Safe to write up.")
    elif okA and not okB:
        log("  Only A passes => the conclusion has to stay narrow: the irreducible triplet term can be detected (residual clearly above the floor),")
        log("  but the residual cannot be identified with the triplet term (direction does not align, cosine %+.2f)."
            % max(res[l]["cosm"] for l in lam_list[1:]))
        log("  Per the pre-registration this claim drops to an appendix, and the failed specificity criterion must be stated.")
    else:
        log("  A fails too => the test has no power; this claim drops to an appendix, with \"cannot be falsified\" written out.")

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "E6.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
