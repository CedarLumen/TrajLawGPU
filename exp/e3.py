"""E3 main experiment: closed-form symbolic law vs numerical competitors, covered / uncovered.

Three changes from the old version:
 1 KRR coefficient mapping fixed (the design columns alternate (rh_k, n_k); the old code took (k, K+k))
 2 (K, h²) is now chosen by in-sample NMSE, not by the covered-region law error we report
 3 design matrix and AᵀA are built in blocks instead of stacking 2K (Ts,N,N) arrays into a list
"""
import os
import sys
import time

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
    terms = [*known.terms, *sc["unknown"].terms]
    T_END, SIGMA = 6.0, 0.0

    Xs, Vs = sim.rk4(terms, ext, sc["X0"], sc["V0"], T_END, cfg["dt"], every=1, graph=False)
    every = 4
    Xd, Vd = Xs[::every], Vs[::every]
    dt_obs = cfg["dt"] * every
    log("truth = magnus (κ=0.07)    N=%d  d=2  T=%gs   position noise σ_x=%g   obs Δt=%gs"
        % (N, T_END, SIGMA, dt_obs))
    log("dense %d frames -> observed %d frames   device %s" % (Xs.shape[0], Xd.shape[0], rt.DEV))

    cfg_obs = scenes.make_cfg("mid")
    _, vis = obs.observe(Xd, cfg_obs, np.random.default_rng(scenes.RNG_SEED))
    log("regime mid: visible trajectory points %.3f, seen at least once %d/%d"
        % (float(vis.mean()), int(vis.any(axis=0).sum()), N))

    a_known = known_accel(known.terms, ext, Xd, Vd)
    Xn = Xd + rt.t(np.random.default_rng(7).normal(0.0, SIGMA, size=tuple(Xd.shape)))
    resid_est = est.est_sg(Xn, dt_obs, 4, 4) - a_known

    lo, hi = 12, Xd.shape[0] - 12
    good = np.zeros(tuple(Xd.shape[:2]), bool)
    good[lo:hi] = vis[lo:hi]
    log("training rows (observed (t,i) pairs): %d, fraction %.3f" % (good.sum(), good.sum() / (Xd.shape[0] * N)))

    # ---- AFR-sym ----------------------------------------------------------
    sub = np.arange(lo, hi, 8)
    Rsub = Xd[sub][:, :, None, :] - Xd[sub][:, None, :, :]
    Vsub = Vd[sub][:, :, None, :] - Vd[sub][:, None, :, :]
    A = bases.pair_sum_design(Rsub, Vsub, bases.VEL)
    y = resid_est[sub].reshape(-1)
    okey = np.repeat(good[sub].reshape(-1), 2)
    t0 = time.time()
    c_sym = fit.stlsq(obs.np_(A[okey]), obs.np_(y[okey]))
    log("AFR-sym solve %.2fs   design matrix %s   AᵀA size %s" % (time.time() - t0, tuple(A.shape), (bases.M, bases.M)))
    log("coefficients (truth: v⊥R̂ = +0.070, v∥n̂ = −0.070, rest 0):")
    for nm, v in zip(bases.NAMES, c_sym):
        if abs(v) > 1e-4:
            log("    %-6s %+.5f" % (nm, v))

    # ---- coverage grid (bins taken from the fully observed reference) --------
    ref = obs.pair_configs(obs.np_(Xd), obs.np_(Vd), np.ones(tuple(Xd.shape[:2]), bool))
    edges = obs.make_edges(ref, nb=16)
    tr = obs.pair_configs(obs.np_(Xd), obs.np_(Vd), vis)
    ncell = tuple(len(e) - 1 for e in edges)

    def cells(pc):
        X = np.stack([pc["r"], pc["vpar"], pc["vperp"]], axis=-1)
        return np.array([np.clip(np.searchsorted(edges[k], X[:, k]) - 1, 0, len(edges[k]) - 2)
                         for k in range(3)]).T
    occ = np.zeros(ncell, bool)
    occ[tuple(cells(tr).T)] = True
    dil = np.zeros_like(occ)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for oz in (-1, 0, 1):
                off = (ox, oy, oz)
                src = np.roll(occ, off, axis=(0, 1, 2))
                sl = tuple(slice(max(0, -o), n - max(0, o)) for o, n in zip(off, ncell))
                dst = tuple(slice(max(0, o), n - max(0, -o)) for o, n in zip(off, ncell))
                dil[dst] |= src[sl]
    log("configuration space: %d training pairs / %d fully observed; grid %s, uncovered bin fraction %.3f"
        % (tr["n"], ref["n"], str(ncell), 1 - dil.mean()))

    # ---- evaluation set (shared by the three methods) ------------------------
    idx = np.arange(0, Xd.shape[0], 32)
    R = Xd[idx][:, :, None, :] - Xd[idx][:, None, :, :]
    V = Vd[idx][:, :, None, :] - Vd[idx][:, None, :, :]
    f_true = sc["unknown"].f(R, V)
    offd = ~np.eye(N, dtype=bool)
    r_e, _, _, vp_e, vq_e = bases.frame(R, V)
    cidx = [np.clip(np.searchsorted(edges[k], Xk) - 1, 0, len(edges[k]) - 2)
            for k, Xk in enumerate((obs.np_(r_e), obs.np_(vp_e), obs.np_(vq_e)))]
    lab3 = dil[cidx[0], cidx[1], cidx[2]][:, offd]
    den_all = (f_true[:, offd, :] ** 2).sum(-1).reshape(-1)

    def law_err(fpred):
        e = (fpred - f_true)[:, offd, :]
        num = (e ** 2).sum(-1).reshape(-1)
        cov = lab3.reshape(-1)
        unc = ~cov
        return (float(num[cov].sum() / max(float(den_all[cov].sum()), 1e-300)),
                float(num[unc].sum() / max(float(den_all[unc].sum()), 1e-300)),
                int(cov.sum()), int(unc.sum()))
    log("evaluation frames: %d (shared by the three methods)" % len(idx))

    log()
    log("%-12s %-18s %-18s %s" % ("method", "covered law error NMSE", "uncovered law error NMSE", "rows (cov/uncov)"))
    log("-" * 82)
    ac, au, nc, nu = law_err(bases.eval_law(R, V, c_sym))
    log("%-12s %-18.4g %-18.4g %d / %d" % ("AFR-sym", ac, au, nc, nu))

    # ---- NN-pair ----------------------------------------------------------
    dev = rt.DEV
    Xt = Xd[:, :, None, :] - Xd[:, None, :, :]
    Vt = Vd[:, :, None, :] - Vd[:, None, :, :]
    r_t, rh_t, n_t, vp_t, vq_t = bases.frame(Xt, Vt)
    m_t = (~torch.eye(N, dtype=torch.bool, device=r_t.device)).unsqueeze(-1)
    valid = (r_t > 1e-9)
    feat = torch.stack([r_t[valid], vp_t[valid], vq_t[valid]], -1)
    mu = feat.mean(0)
    sd = feat.std(0) + 1e-9
    del Xt, Vt, feat
    Y = resid_est.to(torch.float32)
    G = torch.as_tensor(good, device=Y.device)
    net = torch.nn.Sequential(torch.nn.Linear(3, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 2)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000)

    def fwd(i):
        x = torch.stack([(r_t[i] - mu[0]) / sd[0], (vp_t[i] - mu[1]) / sd[1],
                         (vq_t[i] - mu[2]) / sd[2]], -1).to(torch.float32)
        return net(x)

    def loss_on(i):
        o = fwd(i)
        fv = (o[..., 0:1] * rh_t[i] + o[..., 1:2] * n_t[i]) * m_t
        return ((fv.sum(2) - Y[i]) ** 2)[G[i]].mean()

    tr_fr = np.arange(lo, hi)
    va_fr = tr_fr[::10]
    tr_fr = np.setdiff1d(tr_fr, va_fr)
    rng = np.random.default_rng(0)
    t0 = time.time()
    curve = []
    for it in range(4000):
        b = torch.as_tensor(rng.choice(tr_fr, size=256, replace=False), device=dev)
        opt.zero_grad()
        loss_on(b).backward()
        opt.step()
        sched.step()
        if it % 500 == 0 or it == 3999:
            with torch.no_grad():
                curve.append((it, float(loss_on(b)),
                              float(loss_on(torch.as_tensor(va_fr, device=dev)))))
    log()
    log("NN-pair learning curve (step / train loss / val loss), 4000 steps / %.1fs / %s" % (time.time() - t0, dev))
    log("    " + "   ".join("%d:%.4g/%.4g" % (a, b, c) for a, b, c in curve))
    log("    reference: variance of the residual itself = %.4g (a loss stuck at this level means the net learned nothing)"
        % float(Y[G].var()))
    with torch.no_grad():
        rq, rhq, nq, vpq, vqq = bases.frame(R, V)
        o = net(torch.stack([(rq - mu[0]) / sd[0], (vpq - mu[1]) / sd[1],
                             (vqq - mu[2]) / sd[2]], -1).to(torch.float32))
        f_nn = o.to(torch.float64)[..., 0:1] * rhq + o.to(torch.float64)[..., 1:2] * nq
    nnc, nnu, _, _ = law_err(f_nn)
    log("%-12s %-18.4g %-18.4g" % ("NN-pair", nnc, nnu))

    # ---- KRR-pair: blocked design + in-sample model selection ---------------
    u_tr = np.stack([tr["r"], tr["vpar"], tr["vperp"]], axis=-1)
    u_mu, u_sd = u_tr.mean(0), u_tr.std(0) + 1e-12
    Un = (u_tr - u_mu) / u_sd
    subk = np.arange(lo, hi, 32)
    Rk = Xd[subk][:, :, None, :] - Xd[subk][:, None, :, :]
    Vk = Vd[subk][:, :, None, :] - Vd[subk][:, None, :, :]
    kok = np.repeat(good[subk].reshape(-1), 2)
    yok = resid_est[subk].reshape(-1)
    kok_t = torch.as_tensor(kok, device=dev)
    yok_t = yok

    log()
    log("KRR-pair grid (chosen by in-sample NMSE; the old code used the covered law error, which is selecting on the reported metric)")
    log("  %-6s %-8s %-11s %-10s %-14s %-16s %-16s" %
        ("K", "h²", "cond(AᵀA)", "in-sample NMSE", "chosen by", "covered law error", "uncovered law error"))
    t0 = time.time()
    rows = []
    for Kc in (120, 200, 600):
        cen = fit.kmeans(Un, Kc, seed=1)
        krr = fit.KRR(cen, 1.0, u_mu, u_sd)
        for h2 in (0.05, 0.12, 0.15, 0.4, 1.0):
            krr.h2 = h2
            Ak = krr.design(Rk, Vk, dtype=torch.float32)
            Gk, bk = fit.gram_y(Ak, yok_t, kok_t)
            ck = fit.ridge(Gk, bk, 1e-6)
            ins = float((((Ak[kok_t] @ ck.to(torch.float32)) - yok_t[kok_t]) ** 2).mean()
                        / yok_t[kok_t].var())
            pred = krr.predict(R, V, obs.np_(ck))
            c1, u1, _, _ = law_err(pred)
            rows.append(dict(K=Kc, h2=h2, ins=ins, c=c1, u=u1, cond=float(torch.linalg.cond(Gk.double())),
                             cen=cen))
            log("  %-6d %-8.3g %-11.3g %-10.4g %-14s %-16.4g %-16.4g"
                % (Kc, h2, rows[-1]["cond"], ins, "", c1, u1))
            del Ak, Gk, bk, pred
    best = min(rows, key=lambda r: r["ins"])
    log("  -> in-sample NMSE picks K = %d, h² = %.3g (in-sample %.4g)" % (best["K"], best["h2"], best["ins"]))
    old_pick = min(rows, key=lambda r: r["c"])
    log("  -> the old criterion (covered law error) would pick K = %d, h² = %.3g — different rows mean tuning leakage" %
        (old_pick["K"], old_pick["h2"]))
    log("  sweep %.1fs (at K=600 the design is %d×%d, built in blocks, no 2K (Ts,N,N) intermediates)"
        % (time.time() - t0, len(kok), 2 * 600))

    cen = best["cen"]
    krr = fit.KRR(cen, best["h2"], u_mu, u_sd)
    Ak = krr.design(Rk, Vk, dtype=torch.float32)
    Gk, bk = fit.gram_y(Ak, yok_t, kok_t)
    ck = fit.ridge(Gk, bk, 1e-6)
    kc, ku, _, _ = law_err(krr.predict(R, V, obs.np_(ck)))
    log("%-12s %-18.4g %-18.4g" % ("KRR-pair", kc, ku))

    # old mapping: the design columns alternate (rh_k, n_k); the old script took (k, K+k)
    U, rh_q, n_q = krr._U(R, V)
    c_np = obs.np_(ck)
    fold = R * 0
    for k in range(len(cen)):
        P = torch.exp(-((U - torch.as_tensor(cen[k], device=U.device)) ** 2).sum(-1) / (2 * krr.h2)).unsqueeze(-1)
        fold = fold + float(c_np[k]) * P * rh_q + float(c_np[len(cen) + k]) * P * n_q
    oc, ou, _, _ = law_err(fold)
    log("%-12s %-18.4g %-18.4g   <- the old coefficient mapping, same c" % ("KRR(old map)", oc, ou))

    log()
    log("peak GPU memory %.3g GB (torch side)" % rt.peak_gb())
    log()
    log("%-14s %-20s %-20s %-14s %s" % ("method", "covered law error NMSE", "uncovered law error NMSE", "degeneracy uncov/cov", "representation"))
    log("-" * 100)
    log("%-14s %-20.4g %-20.4g %-14.2f %s" % ("AFR-sym", ac, au, au / max(ac, 1e-300),
                                              "global closed form (10 power/velocity terms)"))
    log("%-14s %-20.4g %-20.4g %-14.2f %s" % ("KRR-pair", kc, ku, ku / max(kc, 1e-300),
                                              "local radial basis (%d centers)" % len(cen)))
    log("%-14s %-20.4g %-20.4g %-14.2f %s" % ("NN-pair", nnc, nnu, nnu / max(nnc, 1e-300), "MLP"))
    log("%-14s %-20.4g %-20.4g %-14.2f %s" % ("KRR(old map)", oc, ou, ou / max(oc, 1e-300),
                                              "same coefficients, misaligned mapping"))
    log()
    log("C1 is judged by the degeneracy ratio (uncovered / covered), not by which method is lower in both regions:")
    log("  numerical baselines degenerate %.1f x (KRR %.1f, NN %.1f); the symbolic form %.2f x."
        % (max(ku, nnu) / max(kc, nnc), ku / max(kc, 1e-300), nnu / max(nnc, 1e-300),
           au / max(ac, 1e-300)))
    log("  Note: the AFR-sym basis happens to contain the true family, so it is machine-precision")
    log("        in both regions — an upper bound from picking the right representation, not a general result.")

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "E3.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
