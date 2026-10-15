"""E7 real material: recover the pair potential from liquid-argon trajectories (3-D, PBC, physical units).

Truth = Lennard-Jones with literature parameters ε/kB = 119.8 K, σ = 3.405 Å; the state point sits
near the triple point of real argon. The inversion only uses two bases:
  A a generic basis {r^-p r̂, p = 4…16} (13 terms) — the functional form is unknown
  B a physics-constrained two-term basis {p = 7, p = 13}, i.e. r^-6 and r^-12 — form given, parameters not
"""
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tl import est, fit, obs, rt, sim, thermo  # noqa: E402
from tl.forces import LJ  # noqa: E402

OUT = []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


SIG, EPS, MASS = 3.405, 119.8, 39.948
RHO_STAR, T_RUN = 0.708, 120.0
N = 108
BOX = SIG * (N / RHO_STAR) ** (1.0 / 3.0)
DT = 0.002
TU_PS = math.sqrt(1.66053906660e-27 * (1e-10) ** 2 / 1.380649e-23) / 1e-12
PS_GEN = list(range(4, 17))
PS_2 = [7, 13]
EVERY = 1
NSTEP = int(os.environ.get("NSTEP", "30000"))
NEQ = int(os.environ.get("NEQ", "48"))


def fcc_box(n, box):
    a = box / n
    b = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * a
    pts = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                pts.append(b + np.array([i, j, k]) * a)
    return np.concatenate(pts, 0)[:N]


def chunk_accel(terms, X, V, box, step=192):
    out = torch.empty_like(X)
    for i in range(0, len(X), step):
        sl = slice(i, min(i + step, len(X)))
        out[sl] = sim.accel(terms, [], X[sl], V[sl], box=box)
    return out


def chunk_epot(terms, X, box, step=192):
    out = torch.empty(len(X), dtype=X.dtype, device=X.device)
    for i in range(0, len(X), step):
        sl = slice(i, min(i + step, len(X)))
        out[sl] = sim.total_potential(terms, [], X[sl], box=box).sum(-1)
    return out


def rescale(V, T):
    ke = 0.5 * MASS * float((V ** 2).sum())
    return V * math.sqrt(T / max(2.0 * ke / (3.0 * N), 1e-12))


def langevin(terms, X, V, T, n, dt, box, gamma=0.5, chunk=1000, tag=""):
    """NVT equilibration: stochastic Euler–Verlet (dv = (a − γv)dt + sqrt(2γT/m dt)·ξ, dx = v dt).

    One force evaluation per step, 4x cheaper than RK4; γ = 0.5 /tu (relaxation time ≈ 2.2 ps).
    """
    gen = torch.Generator(device=X.device).manual_seed(7)
    sig = math.sqrt(2.0 * gamma * T / MASS * dt)
    n = int(n)
    for c in range(max(1, n // chunk)):
        for _ in range(chunk):
            a = sim.accel(terms, [], X, V, box=box)
            V = V + (a - gamma * V) * dt + sig * torch.randn(
                X.shape, generator=gen, device=X.device, dtype=X.dtype)
            X = X + V * dt
        if tag:
            Tx = float(MASS * (V ** 2).sum() / (3.0 * N))
            gx_, gy_ = thermo.rdf(X, box, nb=120)
            log("    %s %5d/%d  T = %7.2f K  RDF first peak %.2f Å  g_max %.2f  g(7.5Å) %.3f"
                % (tag, (c + 1) * chunk, n, Tx, gx_[int(np.argmax(gy_))], gy_.max(),
                   float(gy_[np.argmin(np.abs(gx_ - 7.5))])))
    return X, V


class RecLaw:
    """Wrap {c_p r^-p r̂} as a force-law object so energy and pressure can reuse it."""

    conservative = True

    def __init__(self, c, ps):
        self.c = np.asarray(c, dtype=np.float64)
        self.ps = list(ps)
        self.u_k = thermo.power_u_over_k(self.c, self.ps)

    def f(self, R, V):
        return thermo.radial_law(R, torch.tensor(self.c, device=R.device, dtype=R.dtype), self.ps)

    def u(self, R):
        r = R.norm(dim=-1)
        return torch.tensor(self.u_k(obs.np_(r)), device=R.device, dtype=R.dtype)


def main():
    log("units: length Å  mass amu  energy K·kB  time %.4f ps" % TU_PS)
    log("system: %d argon atoms, 3-D periodic boundary, box %.4f Å (minimum image, cutoff %.2f Å)" % (N, BOX, BOX / 2))
    log("state point: T = %.2f K (= %.3f ε/kB), ρ* = %.3f — mid liquid range, away from the triple point"
        % (T_RUN, T_RUN / EPS, RHO_STAR))
    rho_mass = (RHO_STAR / SIG ** 3) * 1e24 * MASS / 6.02214076e23
    log("  mass density %.4f g/cm³ (the order of saturated liquid argon near 120 K)" % rho_mass)
    log("truth (used to generate the data only): LJ, ε/kB = %.1f K, σ = %.3f Å, m = %.3f amu" % (EPS, SIG, MASS))
    log("integration step %.4f ps; all positions recorded every %d steps (Δt_obs = %.1f fs); forces were never observed"
        % (DT * TU_PS, EVERY, DT * EVERY * TU_PS * 1000))

    truth = LJ(EPS, SIG)
    rng = np.random.default_rng(20261015)
    X = torch.tensor(fcc_box(3, BOX))
    V0 = torch.tensor(rng.normal(0, 1.0, (N, 3)))
    V0 = rescale(V0 - V0.mean(0), 3.0 * EPS)
    nn = BOX / 3.0 / math.sqrt(2.0)
    log("start: 3x3x3 FCC lattice (nearest neighbour %.3f Å = %.3f σ), Langevin melting from T* = 3" % (nn, nn / SIG))
    X, V = X, V0

    t0 = time.time()
    X, V = langevin([truth], X, V, 3.0 * EPS, 5000, DT, BOX, chunk=1000, tag="melt")
    X, V = langevin([truth], X, V, T_RUN, NEQ * 1000, DT, BOX, chunk=2000, tag="cool+equilibrate")
    rt.sync()
    gx, gy = thermo.rdf(X, BOX, nb=120)
    log()
    log("Langevin equilibration %d steps = %.1f ps, %.1fs (NVT, γ = 0.5 /tu)" %
        (5000 + NEQ * 1000, (5000 + NEQ * 1000) * DT * TU_PS, time.time() - t0))

    t0 = time.time()
    Xs, Vs = sim.rk4([truth], [], X, V, None, DT, steps=NSTEP, every=EVERY, box=BOX)
    rt.sync()
    burn = NSTEP // 5
    Xs, Vs = Xs[burn:], Vs[burn:]
    Tsim = 2.0 * (0.5 * MASS * (Vs ** 2).sum((-1, -2))) / (3.0 * N)
    E = 0.5 * MASS * (Vs ** 2).sum((-1, -2)) + chunk_epot([truth], Xs, BOX)
    log()
    log("production %d steps x %.2f fs = %.1f ps; %d frames kept after dropping %d steps, %.1fs" %
        (NSTEP, DT * TU_PS * 1000, NSTEP * DT * TU_PS, burn, Xs.shape[0], time.time() - t0))
    log("  temperature %.2f K (mean) ± %.2f K; relative energy drift %.3g" %
        (float(Tsim.mean()), float(Tsim.std()), float((E - E[0]).abs().max() / E.abs().mean())))
    ms = thermo.msd(Xs)
    tt = np.arange(len(ms)) * EVERY * DT * TU_PS
    h = len(ms) // 2
    D = float(ms[-1] - ms[h]) / (6.0 * (tt[-1] - tt[h]))
    log("  liquid/solid MSD test: at %.1f ps %.2f Å², at %.1f ps %.2f Å²; diffusion coefficient from the linear part D = %.3g×10⁻⁹ m²/s" %
        (tt[h], ms[h], tt[-1], ms[-1], D * 1e-8 / 1e-9))
    log("  final RDF: first peak %.2f Å, g_max %.2f, g(7.5 Å) %.3f (liquid: g_max ≈ 3 and g→1; crystal: larger g_max and g(7.5) off 1)"
        % (gx[int(np.argmax(gy))], gy.max(), float(gy[np.argmin(np.abs(gx - 7.5))])))

    lo0, hi0 = 12, Xs.shape[0] - 12
    idx = np.arange(0, Xs.shape[0], 64)
    Re = Xs[idx][:, :, None, :] - Xs[idx][:, None, :, :]
    Re = Re - BOX * torch.round(Re / BOX)
    ft = truth.f(Re, Re * 0)
    offd = ~torch.eye(N, dtype=torch.bool, device=Re.device)
    den = float((ft[:, offd, :] ** 2).sum())
    rs = obs.np_(Re.norm(dim=-1))
    rs = rs[np.isfinite(rs) & (rs > 1e-6)]
    log()
    log("evaluation frames %d; sampled r range %.3f ~ %.3f Å (median %.3f, minimum within 2.3σ)" %
        (len(idx), rs.min(), np.percentile(rs, 99.9), np.median(rs)))

    def invert(Xk, dtk, ps):
        ak = est.est_sg(Xk, dtk, 4, 4)
        hik = Xk.shape[0] - 12
        sub = np.arange(12, hik, max(1, len(range(12, hik)) // 400))
        R = Xk[sub][:, :, None, :] - Xk[sub][:, None, :, :]
        R = R - BOX * torch.round(R / BOX)
        Vsub = Xk[sub] * 0
        A = thermo.pair_sum_radial(R, Vsub, ps)
        y = ak[sub].reshape(-1)
        Ad, yd = obs.np_(A), obs.np_(y)
        cn = np.maximum(np.linalg.norm(Ad, axis=0), 1e-300)
        c = fit.stlsq(Ad, yd)
        ins = float(np.mean((Ad @ c - yd) ** 2) / np.var(yd))
        condn = float(np.linalg.cond((Ad / cn).T @ (Ad / cn)))
        cc = torch.tensor(c, device=Re.device)
        fr = thermo.radial_law(Re, cc, ps)
        err = float(((fr - ft)[:, offd, :] ** 2).sum() / den)
        return dict(c=c, ins=ins, cond=condn, err=err, nrow=len(yd))

    log()
    log("observation-interval sweep (same trajectory resampled at Δt_obs; basis B = {p=7, p=13})")
    log("  %-9s %-8s %-7s %-11s %-11s %-8s %-10s %s" %
        ("Δt_obs/fs", "frames", "rows", "normalized cond", "pair-sum residual", "σ/Å", "ε/kB", "pairwise law error"))
    ksweep = []
    for k in (1, 2, 4, 8):
        Xk = Xs[::k]
        r = invert(Xk, DT * k, PS_2)
        c7, c13 = r["c"]
        s6 = -c13 / (2.0 * c7)
        s_rec = s6 ** (1.0 / 6.0) if s6 > 0 else float("nan")
        e_rec = c13 / (48.0 * s_rec ** 12) if s_rec == s_rec else float("nan")
        r.update(k=k, sig=s_rec, eps=e_rec)
        ksweep.append(r)
        log("  %-9.1f %-8d %-7d %-11.3g %-11.4g %-8.4f %-10.3f %.4g" %
            (DT * k * TU_PS * 1000, Xk.shape[0], r["nrow"], r["cond"], r["ins"],
             s_rec, e_rec, r["err"]))
    best = min(ksweep, key=lambda r: abs(r["eps"] / EPS - 1))
    log("  -> best recovery: Δt_obs = %.1f fs, σ = %.4f Å (%+.3f%%), ε/kB = %.2f K (%+.3f%%)"
        % (best["k"] * DT * TU_PS * 1000, best["sig"], 100 * (best["sig"] / SIG - 1),
           best["eps"], 100 * (best["eps"] / EPS - 1)))

    k = best["k"]
    Xk = Xs[::k]
    gen = invert(Xk, DT * k, PS_GEN)
    log()
    log("the generic 13-term basis (basis A) on the same setting, for reference:")
    log("  normalized cond %.3g (basis B %.3g); pair-sum residual %.4g (basis B %.4g); pairwise law error %.4g (basis B %.4g)"
        % (gen["cond"], best["cond"], gen["ins"], best["ins"], gen["err"], best["err"]))
    rg = np.linspace(2.6, 6.0, 200)
    utru = thermo.lj_u_over_k(EPS, SIG)(rg)
    uA = thermo.power_u_over_k(gen["c"], PS_GEN)(rg)
    uB = thermo.power_u_over_k(best["c"], PS_2)(rg)
    m = np.max(np.abs(utru))
    log("  potential curve max|ΔU|/max|U|: basis A %.4g (U_min = %+.3f K·kB), basis B %.4g (U_min = %+.3f K·kB); true U_min = %.3f"
        % (float(np.max(np.abs(uA - utru)) / m), float(np.min(uA)),
           float(np.max(np.abs(uB - utru)) / m), float(np.min(uB)),
           float(np.min(utru))))

    log()
    log("using the true acceleration as the target (isolates estimator bias):")
    hik = Xk.shape[0] - 12
    sub = np.arange(12, hik, max(1, len(range(12, hik)) // 400))
    at = chunk_accel([truth], Xk[sub], Vs[::k][sub], BOX)
    R = Xk[sub][:, :, None, :] - Xk[sub][:, None, :, :]
    R = R - BOX * torch.round(R / BOX)
    for ps, tag in ((PS_2, "B"), (PS_GEN, "A")):
        A = obs.np_(thermo.pair_sum_radial(R, Xk[sub] * 0, ps))
        yd = obs.np_(at.reshape(-1))
        c = fit.stlsq(A, yd)
        cc = torch.tensor(c, device=Re.device)
        fr = thermo.radial_law(Re, cc, ps)
        err = float(((fr - ft)[:, offd, :] ** 2).sum() / den)
        extra = ""
        if tag == "B":
            s6 = -c[1] / (2.0 * c[0])
            sr = s6 ** (1.0 / 6.0)
            extra = "  σ = %.4f Å（%+.4f%%）ε/kB = %.2f K（%+.4f%%）" % (
                sr, 100 * (sr / SIG - 1), c[1] / (48 * sr ** 12), 100 * (c[1] / (48 * sr ** 12) / EPS - 1))
        log("  %s pairwise law error %.4g (upper bound on fit accuracy)%s" % (tag, err, extra))

    log()
    log("macroscopic closure (recovered potential and true potential, same frame, same velocities)")
    law_rec = RecLaw(best["c"], PS_2)
    Pref = Uref = None
    for tag, lw in (("true LJ", truth), ("recovered (B)", law_rec)):
        P = thermo.virial_pressure(Xs[-1], Vs[-1], [lw], BOX, float(Tsim[-1]))
        U = float(chunk_epot([lw], Xs[-1:], BOX)[0] / N)
        log("  %-12s potential/atom %+9.4f K·kB   pressure %+9.1f bar" %
            (tag, U, P * thermo.BAR_PER_UNIT))
        if lw is truth:
            Pref, Uref = P, U
        else:
            log("  difference: potential/atom %.4g K·kB (%.3g%%), pressure %.4g bar (%.3g%%)" %
                (abs(U - Uref), 100 * abs(U - Uref) / abs(Uref),
                 abs(P - Pref) * thermo.BAR_PER_UNIT, 100 * abs(P - Pref) / abs(Pref)))

    log()
    log("second virial coefficient B2(T) (cm³/mol, integrated from U(r))")
    log("  %-8s %-12s %-12s %s" % ("T/K", "true LJ", "recovered (B)", "rel. diff"))
    urec_f = thermo.power_u_over_k(best["c"], PS_2)
    for T in (100.0, 150.0, 200.0, 300.0, 400.0, 600.0):
        b0 = thermo.second_virial(T, thermo.lj_u_over_k(EPS, SIG))
        b1 = thermo.second_virial(T, urec_f)
        log("  %-8.0f %-12.4f %-12.4f %.3g" % (T, b0, b1, abs(b1 - b0) / abs(b0)))

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "E7.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()

