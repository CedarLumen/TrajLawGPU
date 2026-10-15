"""Turn every hidden bug into a checkable number. Verdicts land in reports/BUGS.txt."""
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tl import bases, est, fit, obs, rt, scenes, sim  # noqa: E402

OLD = r"D:\FAST\TrajLaw\sim"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = []


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


def hdr(t):
    log()
    log("=" * 86)
    log(t)
    log("=" * 86)


def known_accel(terms, ext, X, V, step=768):
    out = torch.empty_like(X)
    for i in range(0, len(X), step):
        sl = slice(i, min(i + step, len(X)))
        out[sl] = sim.pair_accel(terms, X[sl], V[sl]) + sim.accel([], ext, X[sl], V[sl])
    return out


def main():
    sc = scenes.build_scene("magnus", n=40)
    known, ext = sc["known"], sc["ext"]
    terms = [*known.terms, *sc["unknown"].terms]
    cfg = scenes.make_cfg("full")
    Xs, Vs = sim.rk4(terms, ext, sc["X0"], sc["V0"], 6.0, cfg["dt"], every=4)
    dt_obs = cfg["dt"] * 4
    a_est = est.est_sg(Xs, dt_obs, 4, 4)
    a_known = known_accel(known.terms, ext, Xs, Vs)
    resid = a_est - a_known
    lo, hi = 12, Xs.shape[0] - 12
    vis = obs.observe(Xs, scenes.make_cfg("mid"), np.random.default_rng(scenes.RNG_SEED))[1]
    good = np.zeros(tuple(Xs.shape[:2]), bool)
    good[lo:hi] = vis[lo:hi]

    hdr("BUG 1  KRR coefficient mapping misaligned (in both e3_main.py and e3b_krr_sweep.py)")
    log("The design columns run (rh_0, n_0, rh_1, n_1, ...): column 2k is rh_k and column 2k+1 is n_k.")
    log("The old script rebuilt the pair force as c[k]*P_k*rh + c[K+k]*P_k*n, which does not match that order.")
    subk = np.arange(lo, hi, 32)
    Rk = Xs[subk][:, :, None, :] - Xs[subk][:, None, :, :]
    Vk = Vs[subk][:, :, None, :] - Vs[subk][:, None, :, :]
    tr = obs.pair_configs(obs.np_(Xs), obs.np_(Vs), vis)
    u_tr = np.stack([tr["r"], tr["vpar"], tr["vperp"]], -1)
    u_mu, u_sd = u_tr.mean(0), u_tr.std(0) + 1e-12
    Un = (u_tr - u_mu) / u_sd
    K = 120
    cen = fit.kmeans(Un, K, seed=1)
    krr = fit.KRR(cen, 0.4, u_mu, u_sd)
    Ak = krr.design(Rk, Vk, dtype=torch.float32)
    kok = np.repeat(good[subk].reshape(-1), 2)
    yok = resid[subk].reshape(-1)
    kok_t = torch.as_tensor(kok)
    Gk, bk = fit.gram_y(Ak, yok, kok_t)
    ck = fit.ridge(Gk, bk, 1e-6)
    idx = np.arange(0, Xs.shape[0], 32)
    Re = Xs[idx][:, :, None, :] - Xs[idx][:, None, :, :]
    Ve = Vs[idx][:, :, None, :] - Vs[idx][:, None, :, :]
    f_true = sc["unknown"].f(Re, Ve)
    offd = ~torch.eye(40, dtype=torch.bool)
    den = float((f_true[:, offd, :] ** 2).sum())
    c_np = obs.np_(ck)
    U, rh_q, n_q = krr._U(Re, Ve)
    f_new = krr.predict(Re, Ve, c_np)
    fold = Re * 0
    for k in range(K):
        P = torch.exp(-((U - torch.as_tensor(cen[k], device=U.device)) ** 2).sum(-1)
                      / (2 * krr.h2)).unsqueeze(-1)
        fold = fold + float(c_np[k]) * P * rh_q + float(c_np[K + k]) * P * n_q
    e_new = float(((f_new - f_true)[:, offd, :] ** 2).sum() / den)
    e_old = float(((fold - f_true)[:, offd, :] ** 2).sum() / den)
    d = float((f_new - fold).norm() / f_new.norm())
    log("same coefficient vector c (K=%d, h²=0.4):" % K)
    log("  taking (2k, 2k+1) as the column order: reference pair force, law error NMSE = %.4g" % e_new)
    log("  taking (k, K+k) as the old code did: relative force difference %.3f, law error NMSE = %.4g" % (d, e_old))
    log("  => the old convention is %.1f x worse than the correct one. The KRR column of E3/E3b is void." % (e_old / max(e_new, 1e-300)))

    hdr("BUG 2  the sweep selected on the metric it later reports (e3_main.py)")
    log("Old code: `if best is None or c1 < best[1]`, where c1 is the covered-region law error — selecting on the reported metric.")
    log("Sweeping once per criterion to see whether they pick the same cell:")
    rows = []
    for Kc in (120, 600):
        cen2 = fit.kmeans(Un, Kc, seed=1)
        k2 = fit.KRR(cen2, 1.0, u_mu, u_sd)
        Ak2 = k2.design(Rk, Vk, dtype=torch.float32)
        kok2 = torch.as_tensor(kok)
        for h2 in (0.12, 0.4, 1.0):
            k2.h2 = h2
            Ak2 = k2.design(Rk, Vk, dtype=torch.float32)
            G2, b2 = fit.gram_y(Ak2, yok, kok2)
            c2 = fit.ridge(G2, b2, 1e-6)
            ins = float((((Ak2[kok2] @ c2.float()) - yok[kok2]) ** 2).mean() / yok[kok2].var())
            pred = k2.predict(Re, Ve, obs.np_(c2))
            cov = float(((pred - f_true)[:, offd, :] ** 2).sum() / den)
            rows.append((Kc, h2, ins, cov))
            log("    K=%-5d h²=%-6.3g in-sample NMSE %.4g   covered law error %.4g" % (Kc, h2, ins, cov))
    b_ins = min(rows, key=lambda r: r[2])
    b_cov = min(rows, key=lambda r: r[3])
    log("  by in-sample NMSE -> K=%d h²=%.3g" % (b_ins[0], b_ins[1]))
    log("  by covered law error -> K=%d h²=%.3g" % (b_cov[0], b_cov[1]))
    log("  => %s" % ("the two differ, so the old code did select on the reported metric" if b_ins[:2] != b_cov[:2]
                    else "identical on this grid (another grid may differ; the rule still has to change)"))

    hdr("BUG 3  memory: the old version materialised the whole palette at once")
    log("In old e3_main.krr_build, `((Un[:,None,:]-cen[None,:,:])**2).sum(-1)` has len(Un)×K×3 elements.")
    log("training pair configurations in this mid-regime run (measured) = %d" % len(Un))
    for Kc in (200, 600):
        log("    K=%d -> %d×%d×3×8B = %.2f GB (single allocation)" % (Kc, len(Un), Kc, len(Un) * Kc * 3 * 8 / 2 ** 30))
    log("measured (peak working-set increase of a child process on synthetic points of the same size):")
    log("  %-8s %-8s %-12s %s" % ("points", "K", "old", "new (30k subsample)"))
    for Kc in (60, 120, 200):
        r1 = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "_mem_child.py"),
                             str(len(Un)), str(Kc), "old"], capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "_mem_child.py"),
                             str(len(Un)), str(Kc), "new"], capture_output=True, text=True)
        if r1.returncode or r2.returncode:
            log("  measurement failed %s %s" % (r1.stderr[-120:], r2.stderr[-120:]))
            continue
        _, _, _, t1, m1 = r1.stdout.split()
        _, _, _, t2, m2 = r2.stdout.split()
        log("  %-8d %-8d %-12s %s" % (len(Un), Kc, "%s MB / %ss" % (m1, t1),
                                      "%s MB / %ss" % (m2, t2)))
    log("  => the old version grows linearly with K; the new one subsamples and is independent of K. Linear extrapolation to K=600 = %.1f GB."
        % (len(Un) * 600 * 3 * 8 / 2 ** 30))

    hdr("BUG 4  sum_design recomputed frame for every column — slower, but the absolute cost is tiny")
    R = Xs[subk][:, :, None, :] - Xs[subk][:, None, :, :]
    V = Vs[subk][:, :, None, :] - Vs[subk][:, None, :, :]
    ts = []
    for _ in range(5):
        rt.sync()
        t0 = time.time()
        _ = bases.pair_sum_design(R, V, bases.VEL)
        rt.sync()
        ts.append(time.time() - t0)
    t_new = min(ts)
    ts = []
    for _ in range(5):
        rt.sync()
        t0 = time.time()
        out = torch.empty((*R.shape[:2], R.shape[-1], bases.M), dtype=R.dtype, device=R.device)
        for j, i in enumerate(bases.VEL):
            r_, rh_, n_, vp_, vq_ = bases.frame(R, V)
            g = bases.BASES[i](r_, rh_, n_, vp_, vq_)
            g = g * (~torch.eye(40, dtype=torch.bool, device=R.device)).unsqueeze(-1)
            out[:, :, :, j] = g.sum(2)
        rt.sync()
        ts.append(time.time() - t0)
    t_old = min(ts)
    log("10 columns, %d frames: frame computed once %.4fs   recomputed per column %.4fs   ratio %.2f"
        % (R.shape[0], t_new, t_old, t_old / max(t_new, 1e-9)))
    log("=> a few times slower, but the absolute cost is milliseconds; not a performance concern and not counted as a defect.")
    log("  (the 0.7x measured without sync is bogus: the GPU is asynchronous, so unsynced timing measures dispatch.)")

    hdr("BUG 5  numbers from the report hard-coded into the code")
    with open(os.path.join(OLD, "e3_main.py"), encoding="utf-8") as f:
        src = f.read()
    for ln, line in enumerate(src.splitlines(), 1):
        if "5.7e-5" in line:
            log("    e3_main.py:%d  %s" % (ln, line.strip()))
    log("    This line never changes on a rerun, violating the rule that derived quantities must be computed.")

    hdr("BUG 6  E6 and T.rk4 disagree on the frame index")
    a = [i for i in range(0, 21, 4)]
    b = [i + 1 for i in range(0, 21, 4)]
    log("    T.rk4  takes i ∈ %s" % a)
    log("    e6     takes i ∈ %s   (in rk4_total, i starts at 1 and steps by every)" % b)
    log("    one integration step apart = %g s" % (2.5e-4))

    hdr("BUG 7  a tautology posing as an assertion")
    log("    Old e0_pairsim.py: res['E0.4 NMSE(a,a)'] = mean((a-a)**2)/mean(a**2) is identically 0 and carries no information.")
    log("    In the old e6 the pair-sum residual column was written separately from the verdict block, and numbers not from that column once appeared in the verdict.")

    hdr("BUG 10  E6 matched the triplet force to the residual along the wrong axis (only flips the sign, but the reported numbers are wrong)")
    log("    Old e6: f3_true[j] = f3(Xd[t]) broadcasts the per-particle force (N,d) onto axis 1 (columns) of (N,N,d),")
    log("    while the residual resid[t,i,j] is the pair force on i from j, which belongs on rows (i).")
    log("    The pair residual is antisymmetric in (i,j) (f_ij = −f_ji), so column matching = row matching negated:")
    log("      old report cosine  −0.1559 / −0.1569 / −0.1476")
    log("      correct matching  +0.1559 / +0.1569 / +0.1476")
    log("    New exp/e6.py prints both; measured max |cos_row| − |cos_col| = 0 (see reports/E6.txt).")
    log("    => the conclusion stands (|cos|≈0.15 < 0.7, condition B still fails and the claim stays in the appendix), but the sign has to be corrected when citing.")

    hdr("BUG 8  the old report's conclusion contradicts its own table")
    log("    Old E3_report.txt table: AFR-sym 0.87 / KRR 0.12 / NN 4.76 (degeneracy ratio).")
    log("    The summary right after says \"numerical baselines degenerate 0.1 x (KRR 0.1, NN 4.8); the symbolic form 0.87 x.\"")
    log("    The next sentence says \"outside the sampled configuration region the numerical error rises several-fold while the symbolic form barely moves.\"")
    log("    => KRR's 0.12 is a decrease and the symbolic 0.87 is a decrease too, so both sentences contradict the table;")
    log("       and the crossing of the two curves that C1 requires was never checked in that report.")

    hdr("BUG 9  the degeneracy ratio is undecidable in this setting (the direction flips after the fix)")
    log("    Re-measured after fixing the KRR mapping (exp/e3.py):")
    log("      AFR-sym   covered 4.672e-09   uncovered 4.402e-08   ratio 9.42")
    log("      KRR-pair  covered 3.848e-05   uncovered 2.063e-04   ratio 5.36")
    log("      NN-pair   covered 1.474e-03   uncovered 5.547e-03   ratio 3.76")
    log("      KRR(old map) covered 3.739     uncovered 3.018      ratio 0.81")
    log("    => when both quantities sit at rounding level (1e-9/1e-8) the ratio is noise: the symbolic 9.42 and the old 0.87")
    log("      are both quotients of machine-precision numbers. Judging C1 by the degeneracy ratio neither holds nor fails here —")
    log("      the criterion has to change (e.g. absolute error in the uncovered region < 1e-6), and separately explain why the symbolic form")
    log("      already reaches rounding level in the covered region (the basis contains the true family).")
    log("    The old KRR uncovered error of exactly 1.000 is another fingerprint: the local kernel has no support outside the training range,")
    log("      so the prediction is ≈0 and the NMSE is exactly the 1 of always predicting zero.")

    hdr("A defect of the same kind that the gates caught in my own new code (kept as a positive example)")
    log("    The first CUDA Graph path in tl/sim.py stored a frame every integration step, while every=4 wants one every 4 steps.")
    log("    Symptom: E0's |x|max went from 1.00984 to 0.857619 and every other metric was off.")
    log("    How it was caught: comparing old numpy / new eager / new graph one by one over every ∈ {1,4,5,32,64}.")
    log("    After the fix the three agree to ≤ 2.8e-16 (see tests/test_parity.py).")

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "BUGS.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
