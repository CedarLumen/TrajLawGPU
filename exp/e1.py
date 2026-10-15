"""E1 identifiability gates: angular, sampling range, and integration entanglement."""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tl import bases, obs, rt, scenes, sim  # noqa: E402
from tl.forces import Bond, Composite, Trap  # noqa: E402

OUT = []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = bases.NAMES
M = bases.M


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


def frame(R, V):
    r, rh, n, vp, vq = bases.frame(rt.t(R), rt.t(V))
    return tuple(obs.np_(x) for x in (r, rh, n, vp, vq))


def basis_matrix(R, V):
    r, rh, n, vp, vq = bases.frame(rt.t(R), rt.t(V))
    return obs.np_(torch.stack([f(r, rh, n, vp, vq) for f in bases.BASES], dim=-2))


def to_A(B):
    # Do not write B.reshape(-1, M): C order scrambles the column index; move d up first.
    return B.transpose(0, 2, 1).reshape(-1, M)


def posinfo(R, V):
    _, _, _, vp, _ = frame(R, V)
    r = np.linalg.norm(R, axis=-1)
    sp = np.linalg.norm(V, axis=-1)
    return r, sp, np.where(sp > 1e-12, vp / np.maximum(sp, 1e-300), 0.0)


def two_body(x_terms, v0rel, r0rel=(0.20, 0.00), T_end=1.5, sigma_x=0.0, seed=20261014):
    cfg = scenes.DEFAULT_CFG
    rng = np.random.default_rng(seed)
    adj = np.array([[0.0, 1.0], [1.0, 0.0]])
    known = Composite([Bond(k=scenes.BOND_K, r0=scenes.BOND_R0, adj=adj)])
    ext = [Trap(w2=scenes.TRAP_W2)]
    X0 = np.array([list(r0rel), [-r0rel[0], -r0rel[1]]])
    V0 = np.array([list(v0rel), [-v0rel[0], -v0rel[1]]])
    terms = [*known.terms, *x_terms]
    Xs, Vs = sim.rk4(terms, ext, rt.t(X0), rt.t(V0), T_end, cfg["dt"], every=cfg["every"])
    Xs, Vs = obs.np_(Xs), obs.np_(Vs)
    Xo = Xs + rng.normal(0.0, sigma_x, size=Xs.shape) if sigma_x > 0 else Xs.copy()
    return known, ext, Xs, Vs, Xo


def collect(n_traj=60, T_end=1.5, radial=False, seed=11, x_terms=None):
    # Stack the 60 trajectories as (60,2,2) and do them in one batched rk4 call.
    rng = np.random.default_rng(seed)
    if x_terms is None:
        x_terms = [scenes.UNK["magnus"]()]
    X0 = np.empty((n_traj, 2, 2))
    V0 = np.empty((n_traj, 2, 2))
    for k in range(n_traj):
        if radial:
            # Central force plus radial launch keeps the motion on the line: v⊥ is exactly zero.
            phi = rng.uniform(0, 2 * np.pi)
            e = np.array([np.cos(phi), np.sin(phi)])
            r0 = rng.uniform(0.15, 0.45)
            R0 = r0 * e
            Vrel = rng.uniform(-0.6, 0.6) * e
        else:
            alpha = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, 2 * np.pi)
            speed = rng.uniform(0.05, 0.70)
            r0 = rng.uniform(0.15, 0.45)
            R0 = r0 * np.array([np.cos(phi), np.sin(phi)])
            Vrel = speed * np.array([np.cos(alpha), np.sin(alpha)])
        X0[k] = [R0, -R0]
        V0[k] = [Vrel, -Vrel]
    adj = np.array([[0.0, 1.0], [1.0, 0.0]])
    known = Composite([Bond(k=scenes.BOND_K, r0=scenes.BOND_R0, adj=adj)])
    ext = [Trap(w2=scenes.TRAP_W2)]
    terms = [*known.terms] if radial else [*known.terms, *x_terms]
    cfg = scenes.DEFAULT_CFG
    Xs, Vs = sim.rk4(terms, ext, rt.t(X0), rt.t(V0), T_end, cfg["dt"], every=cfg["every"])
    R = obs.np_(Xs[:, :, 0, :] - Xs[:, :, 1, :]).reshape(-1, 2)
    V = obs.np_(Vs[:, :, 0, :] - Vs[:, :, 1, :]).reshape(-1, 2)
    return R, V


def se_from(A, sigma_a=1.0):
    """Coefficient standard error: with noise σ_a, se = σ_a sqrt(diag(pinv(AᵀA))).

    A column that is identically zero gives inf.
    """
    AtA = A.T @ A
    coln = np.linalg.norm(A, axis=0)
    se = np.full(M, np.inf)
    ok = coln > 1e-9
    if ok.sum() > 0:
        sub = np.linalg.pinv(AtA[np.ix_(ok, ok)])
        se[ok] = sigma_a * np.sqrt(np.clip(np.diag(sub), 0, None))
    return se, coln


def e1a(pool, pool_radial):
    log("=" * 108)
    log("E1a angular identifiability   observation = analytic torques (isolates identifiability itself)")
    log("    criterion = coefficient standard error se (force noise σ_a=1); se > 1 means unidentifiable (|true coefficient| = 1)")
    log("=" * 108)
    for tag, (R, V) in (("generic configuration pool (60 trajectories, random initial relative velocity)", pool),
                        ("exactly radial pool (central force + radial launch ⇒ v⊥ ≡ 0)", pool_radial)):
        r, sp, cos = posinfo(R, V)
        vqmax = float(np.max(np.abs(frame(R, V)[4])))
        log()
        log("【%s】" % tag)
        log("  samples %d ; r ∈ [%.3f, %.3f] ; |v| ∈ [%.3f, %.3f] ; |cos| median %.3f ; max|v⊥| = %.3g"
            % (len(r), r.min(), np.percentile(r, 99.5), sp.min(), np.percentile(sp, 99.5),
               np.median(np.abs(cos)), vqmax))
        log("  %-9s %-8s %-9s | %s" % ("|cos|>=", "kept", "cond(A)",
                                       " ".join("%-6s" % nm for nm in NAMES)))
        log("  " + "-" * 104)
        for th in (0.0, 0.5, 0.9, 0.98, 0.995):
            keep = np.abs(cos) >= th
            if keep.sum() < 200:
                log("  %-9.4g too few samples kept (%d)" % (th, keep.sum()))
                continue
            A = to_A(basis_matrix(R, V)[keep])
            se, coln = se_from(A)
            log("  %-9.4g %-8.3f %-9.3g | " % (th, keep.mean(), np.linalg.cond(A))
                + " ".join("%-6.2g" % v for v in se))
            log("  %-9s %-8s %-9s | " % ("", "", "column norm")
                + " ".join("%-6.2g" % v for v in coln))
            log("  %-9s %-8s %-9s | " % ("", "", "verdict")
                + " ".join(("%-6s" % "!!!") if se[m] > 1.0 or coln[m] <= 1e-9 else "      "
                           for m in range(M)))
    log()
    log("expectation check (against the measurements; both lines have to be written down as they are):")
    log("  - generic pool: as the |cos| threshold rises the first thing to break is the position")
    log("    family (rR̂ / r²R̂ / r³R̂). The cause is not angular: energy conservation ties r and |v|")
    log("    onto one curve (the energy shell), so high-order r polynomials become indistinguishable.")
    log("    At |cos|>=0.995 cond(A) reaches 1.3e3.")
    log("  - exactly radial (v⊥≡0): the three V⊥ columns have norm exactly 0 (1.3e-10 / 1.3e-10 / 1.4e-20),")
    log("    se = inf and cond(A) = 1e23, i.e. genuinely rank-deficient and unidentifiable.")
    log("  - key counterexample: in the same radial data v∥n̂ has se = 0.0043, the best of any basis;")
    log("    the Magnus transverse part is ∝ v∥ and fully visible in head-on collisions.")
    log("    => the initial guess \"the transverse part is unidentifiable in head-on collisions\" is falsified;")
    log("       the corrected claim \"what is unidentifiable is the basis ∝ v⊥\" holds.")


def e1b(R, V):
    log()
    log("=" * 108)
    log("E1b sampling-range identifiability   restricting the sampled (r, |v|) range")
    log("=" * 108)
    r, sp, cos = posinfo(R, V)
    log()
    log("  %-26s %-8s %-9s | %s" % ("restriction", "kept", "cond(A)",
                                    " ".join("%-6s" % nm for nm in NAMES)))
    log("  " + "-" * 104)
    cases = [
        ("none", lambda r, sp, c: np.ones(len(r), bool)),
        ("keep r <= median", lambda r, sp, c: r <= np.median(r)),
        ("keep r >= median", lambda r, sp, c: r >= np.median(r)),
        ("keep |v| <= median", lambda r, sp, c: sp <= np.median(sp)),
        ("keep |v| >= median", lambda r, sp, c: sp >= np.median(sp)),
        ("keep near head-on |cos|>=0.8", lambda r, sp, c: np.abs(c) >= 0.8),
    ]
    for name, f in cases:
        keep = f(r, sp, cos)
        if keep.sum() < 200:
            log("  %-26s too few samples kept (%d)" % (name, keep.sum()))
            continue
        A = to_A(basis_matrix(R, V)[keep])
        se, coln = se_from(A)
        log("  %-26s %-8.3f %-9.3g | " % (name, keep.mean(), np.linalg.cond(A))
            + " ".join("%-6.2g" % v for v in se))
    log()
    log("reading: once the samples are cut to half the range, the bases that live in the")
    log("         unsampled range see se rise — that is what the unsampled range means.")


def e1c():
    log()
    log("=" * 108)
    log("E1c integration entanglement + second-derivative noise: the observation moves from")
    log("    analytic torques to trajectory second differences. If the known terms can be")
    log("    subtracted pointwise, axis 1 is just preprocessing; this measures the cost of")
    log("    having position trajectories only.")
    log("=" * 108)
    dt_obs = scenes.DEFAULT_CFG["dt"] * scenes.DEFAULT_CFG["every"]
    xt = [scenes.UNK["magnus"]()]
    head = ("  %-12s %-11s %-16s %-15s %-15s %s"
            % ("position noise σ", "obs ΔT", "force residual ‖Δa‖/‖a‖", "v⊥R̂ (true 0.07)", "v∥n̂ (true -0.07)", "v∥R̂ (true 0)"))
    old_rows, new_rows, diffs = [], [], []
    for sigma in (0.0, 1e-6, 1e-5, 1e-4, 1e-3):
        known, ext, Xs, Vs, Xo = two_body(xt, (0.02, 0.42), (0.20, 0.0), 4.0, sigma)
        a_tot = (Xo[2:] - 2 * Xo[1:-1] + Xo[:-2]) / dt_obs ** 2
        R = Xo[1:-1, 0, :] - Xo[1:-1, 1, :]
        V = (Xo[2:, 0, :] - Xo[:-2, 0, :] - Xo[2:, 1, :] + Xo[:-2, 1, :]) / (2 * dt_obs)
        Vest = (Xo[2:] - Xo[:-2]) / (2 * dt_obs)
        Xc = rt.t(Xo[1:-1])
        ak_true = obs.np_(sim.pair_accel(known.terms, Xc, rt.t(Vs[1:-1]))[:, 0, :]
                          + sim.accel([], ext, Xc, rt.t(Vs[1:-1]))[:, 0, :])
        ak_est = obs.np_(sim.pair_accel(known.terms, Xc, rt.t(Vest))[:, 0, :]
                         + sim.accel([], ext, Xc, rt.t(Vest))[:, 0, :])
        diffs.append(float(np.max(np.abs(ak_true - ak_est))))
        f_true = obs.np_(sim.pair_accel(xt, rt.t(Xs[1:-1]), rt.t(Vs[1:-1]))[:, 0, :])
        A = to_A(basis_matrix(R, V))
        for ak, store in ((ak_true, old_rows), (ak_est, new_rows)):
            f_est = a_tot[:, 0, :] - ak
            c, *_ = np.linalg.lstsq(A, f_est.reshape(-1), rcond=None)
            resid = np.linalg.norm(f_est - f_true) / np.linalg.norm(f_true)
            store.append("  %-12.1e %-11.1e %-16.4g %-15.3f %-15.3f %s"
                         % (sigma, dt_obs, resid, c[bases.IDX["v⊥R̂"]],
                            c[bases.IDX["v∥n̂"]], "%.3f" % c[bases.IDX["v∥R̂"]]))
    log()
    log(head + "   (before: a_known used the true velocity Vs, oracle mixed in)")
    log("  " + "-" * 108)
    for row in old_rows:
        log(row)
    log()
    log(head + "   (after: a_known uses the estimated velocity Vest)")
    log("  " + "-" * 108)
    for row in new_rows:
        log(row)
    log()
    log("  max pointwise difference between the two a_known versions: %s" % "  ".join("%.3g" % v for v in diffs))
    log()
    log("reference: the truth is Magnus(κ=0.07), i.e. 0.07·(v⊥R̂) − 0.07·(v∥n̂) + 0·(v∥R̂).")
    log("      At σ=0 the fit hits (0.070, −0.070, −0.000) exactly.")
    log("      At σ=1e-6 the v⊥R̂ coefficient jumps from 0.070 to 49.95: position noise σ_x picked")
    log("      up a factor 1/Δt² = 1e6, so the acceleration noise is ~1.0, already 14x the signal (~0.07).")
    log("      => hard threshold of the trajectory route: σ_x/Δt² must stay far below the force")
    log("         scale. This degeneracy is a separate matter from identifiability and has to be")
    log("         reported separately.")


def main():
    t0 = time.time()
    cfg = scenes.DEFAULT_CFG
    log("device %s   dtype %s   numpy %s   torch %s" % (rt.DEV, rt.DT, np.__version__, torch.__version__))
    log("config dt=%gs every=%d obs ΔT=%gs" % (cfg["dt"], cfg["every"], cfg["dt"] * cfg["every"]))
    log()
    t1 = time.time()
    pool_main = collect(radial=False)
    pool_rad = collect(radial=True, seed=77)
    rt.sync()
    log("pooled 60+60 two-body trajectories (one batched rk4 per pool): %.1fs   generic %d   radial %d"
        % (time.time() - t1, len(pool_main[0]), len(pool_rad[0])))
    e1a(pool_main, pool_rad)
    e1b(*pool_main)
    e1c()
    log()
    log("total %.1fs" % (time.time() - t0))
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "E1.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
