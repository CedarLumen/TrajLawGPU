"""Radial power-law bases and thermodynamic quantities, for 3-D real materials."""
import math

import numpy as np
import torch

from .sim import offdiag


def pair_sum_radial(R, V, ps, mask=None, dtype=None):
    """Design matrix of pair sums for the basis g_p = r^-p · r̂. Row order matches pair_sum_design."""
    ts, N, _, d = R.shape
    dtype = dtype or R.dtype
    r = R.norm(dim=-1).clamp_min(1e-12)
    rh = R / r.unsqueeze(-1)
    m = offdiag(N, R.dtype, R.device).unsqueeze(-1)
    out = torch.empty((ts, N, d, len(ps)), dtype=dtype, device=R.device)
    for j, p in enumerate(ps):
        g = (r ** (-p)).unsqueeze(-1) * rh
        if mask is not None:
            g = g * mask.unsqueeze(-1)
        out[:, :, :, j] = (g * m).sum(2)
    return out.reshape(-1, len(ps))


def radial_law(R, c, ps):
    r = R.norm(dim=-1).clamp_min(1e-12)
    rh = R / r.unsqueeze(-1)
    out = R * 0
    for j, p in enumerate(ps):
        out = out + float(c[j]) * (r ** (-p)).unsqueeze(-1) * rh
    return out


def rdf(X, box, rmax=None, nb=120):
    """Radial distribution function g(r). X is one frame, shape (N, d)."""
    n = X.shape[0]
    rmax = rmax if rmax is not None else 0.5 * float(box)
    R = X[:, None, :] - X[None, :, :]
    R = R - box * torch.round(R / box)
    r = R.norm(dim=-1)
    i = torch.arange(n, device=X.device)
    r[i, i] = float("inf")
    h = torch.histc(r[r <= rmax], bins=nb, min=0.0, max=rmax)
    edges = torch.linspace(0, rmax, nb + 1, device=X.device, dtype=r.dtype)
    dv = 4.0 / 3.0 * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho = n / float(box) ** 3
    return ((edges[:-1] + edges[1:]) / 2).cpu().numpy(), (h / (n * rho * dv)).cpu().numpy()


def msd(Xs, ref=None):
    X0 = Xs[0] if ref is None else ref
    return ((Xs - X0) ** 2).sum(-1).mean(-1).cpu().numpy()


def virial_pressure(X, V, terms, box, T):
    """P = (N kT + W/3) / V with W = Σ_{i<j} f_ij·r_ij = ½ Σ_{i≠j} F_ij·R_ij. Units: K·kB/Å³."""
    from .sim import diffs
    R, Vv = diffs(X, V, box)
    F = R * 0
    for t in terms:
        F = F + t.f(R, Vv)
    n = X.shape[0]
    W = 0.5 * float((F * R).sum())
    return (n * T + W / 3.0) / float(box) ** 3


BAR_PER_UNIT = 1.380649e-23 / 1e-30 / 1e5


def second_virial(T, u_over_k, rmin=0.30, rmax=12.0, n=40000):
    """B2(T) = -2π N_A ∫ (e^{-U/kT} - 1) r² dr, in cm³/mol.

    u_over_k(r) returns U(r)/k_B in K, with r in Å.
    """
    r = np.linspace(rmin, rmax, n)
    u = np.asarray(u_over_k(r), dtype=np.float64)
    f = np.expm1(-u / T)
    b = -2.0 * np.pi * 6.02214076e23 * np.trapezoid(f * r ** 2, r)
    return b * 1e-24


def lj_u_over_k(eps, sigma):
    return lambda r: 4.0 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)


def power_u_over_k(c, ps):
    """Recover U(r)/k_B from a force written as {C_p r^-p r̂}.

    f = -dU/dr r̂ with f = Σ C_p r^-p r̂ implies U(r) = Σ C_p r^(1-p)/(p-1), with U(∞) = 0.
    """
    def u(r):
        r = np.maximum(np.asarray(r, dtype=np.float64), 1e-6)
        out = np.zeros_like(r)
        for p, cp in zip(ps, c):
            if abs(cp) < 1e-300:
                continue
            out += float(cp) * r ** (1 - p) / (p - 1)
        return out
    return u
