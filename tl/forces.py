import numpy as np
import torch

from . import rt


class Pair:
    conservative = False

    def f(self, R, V):
        raise NotImplementedError

    def u(self, R):
        raise NotImplementedError


class Bond(Pair):
    conservative = True

    def __init__(self, k, r0, adj):
        self.k, self.r0 = float(k), float(r0)
        self._adj = np.asarray(adj, dtype=np.float64)
        self._cache = {}

    def _A(self, R):
        key = (R.dtype, str(R.device))
        a = self._cache.get(key)
        if a is None:
            a = torch.as_tensor(self._adj, dtype=R.dtype, device=R.device)
            self._cache[key] = a
        return a

    def _r(self, R):
        return (R * R).sum(-1).add(1e-24).sqrt()

    def f(self, R, V):
        r = self._r(R)
        return (-self.k * (r - self.r0) / r).unsqueeze(-1) * R * self._A(R).unsqueeze(-1)

    def u(self, R):
        r = self._r(R)
        return 0.5 * self.k * (r - self.r0) ** 2 * self._A(R)


class SoftCoulomb(Pair):
    conservative = True

    def __init__(self, C, eps=0.06):
        self.C, self.eps = float(C), float(eps)

    def _rs(self, R):
        return (R * R).sum(-1).add(self.eps ** 2).sqrt()

    def f(self, R, V):
        return (-self.C / self._rs(R) ** 3).unsqueeze(-1) * R

    def u(self, R):
        return -self.C / self._rs(R)


class SoftYukawa(Pair):
    conservative = True

    def __init__(self, A, lam, eps=0.06):
        self.A, self.lam, self.eps = float(A), float(lam), float(eps)

    def _rs(self, R):
        return (R * R).sum(-1).add(self.eps ** 2).sqrt()

    def f(self, R, V):
        rs = self._rs(R)
        mag = -self.A * (-rs / self.lam).exp() * (1.0 / rs ** 2 + 1.0 / (self.lam * rs))
        return (mag / rs).unsqueeze(-1) * R

    def u(self, R):
        rs = self._rs(R)
        return -self.A * (-rs / self.lam).exp() / rs


class QuarticR(Pair):
    conservative = True

    def __init__(self, lam):
        self.lam = float(lam)

    def f(self, R, V):
        return (-self.lam * (R * R).sum(-1)).unsqueeze(-1) * R

    def u(self, R):
        return 0.25 * self.lam * (R * R).sum(-1) ** 2


class Drag(Pair):
    def __init__(self, g):
        self.g = float(g)

    def f(self, R, V):
        return -self.g * V


class VLongitudinal(Pair):
    def __init__(self, b, eps=1e-12):
        self.b, self.eps = float(b), float(eps)

    def f(self, R, V):
        r = (R * R).sum(-1).add(self.eps ** 2).sqrt()
        rh = R / r.unsqueeze(-1)
        return (-self.b * (V * rh).sum(-1)).unsqueeze(-1) * rh


class Magnus(Pair):
    def __init__(self, k):
        self.k = float(k)

    def f(self, R, V):
        out = V * 0
        out[..., 0] = self.k * V[..., 1]
        out[..., 1] = -self.k * V[..., 0]
        return out


class Composite(Pair):
    def __init__(self, terms, name=""):
        self.terms = list(terms)
        self.name = name
        self.conservative = all(getattr(x, "conservative", False) for x in self.terms)

    def f(self, R, V):
        out = R * 0
        for x in self.terms:
            out = out + x.f(R, V)
        return out

    def u(self, R):
        out = R[..., 0] * 0
        for x in self.terms:
            out = out + x.u(R)
        return out


def _magnus(R, V, c, lam, eps):
    out = R * 0
    out[..., 0] = c * V[..., 1]
    out[..., 1] = -c * V[..., 0]
    return out


def _drag(R, V, c, lam, eps):
    return -(c.unsqueeze(-1)) * V


def _vlong(R, V, c, lam, eps):
    rh = R / (R * R).sum(-1).add(1e-24).sqrt().unsqueeze(-1)
    return (-c * (V * rh).sum(-1)).unsqueeze(-1) * rh


def _quartic(R, V, c, lam, eps):
    return (-c * (R * R).sum(-1)).unsqueeze(-1) * R


def _yukawa(R, V, c, lam, eps):
    rs = (R * R).sum(-1).add(eps ** 2).sqrt()
    return ((-c * (-rs / lam).exp() * (1.0 / rs ** 2 + 1.0 / (lam * rs))) / rs).unsqueeze(-1) * R


def _coulomb(R, V, c, lam, eps):
    rs = (R * R).sum(-1).add(eps ** 2).sqrt()
    return (-c / rs ** 3).unsqueeze(-1) * R


KINDS = {"magnus": _magnus, "drag": _drag, "vlong": _vlong,
         "quartic_r": _quartic, "yukawa": _yukawa, "coulomb": _coulomb}


class _Bcast:
    """Broadcast length-B coefficients to a shape that fits (..., N, N)."""

    def __init__(self, vals, dtype=None):
        a = np.asarray(vals)
        if dtype is not None:
            a = a.astype(dtype)
        self.vals = a.reshape(-1)
        self._c = {}

    def get(self, ref, tag=""):
        key = (tag, ref.dtype, str(ref.device), ref.ndim)
        v = self._c.get(key)
        if v is None:
            if tag:
                val = np.array([1.0 if s == tag else 0.0 for s in self.vals])
            else:
                val = self.vals
            v = torch.as_tensor(val, dtype=ref.dtype, device=ref.device)
            v = v.reshape((-1,) + (1,) * (ref.ndim - 2))
            self._c[key] = v
        return v

    def tags(self):
        return sorted(set(self.vals.tolist()))


class Mix(Pair):
    """A set of unknown laws stacked along the batch dim: one integration covers several
    truths at once, so the GPU does not sit idle."""

    def __init__(self, kinds, coefs, lam=0.30, eps=0.06):
        self.kinds = list(kinds)
        self.ab = _Bcast(np.asarray(coefs, dtype=np.float64).reshape(-1))
        self.kb = _Bcast(np.array(self.kinds))
        self.lam, self.eps = float(lam), float(eps)
        self.active = [k for k in KINDS if k in set(self.kinds)]

    def f(self, R, V):
        c = self.ab.get(R)
        out = R * 0
        for k in self.active:
            cm = c * self.kb.get(R, tag=k)
            out = out + KINDS[k](R, V, cm, self.lam, self.eps)
        return out


class Triplet:
    """Irreducible triplet term; the force is a single matmul. λ may vary along the batch dim."""

    def __init__(self, lam, a=0.15):
        self.lb = _Bcast(np.asarray(lam, dtype=np.float64).reshape(-1))
        self.a = float(a)

    def f(self, X):
        lam = self.lb.get(X)
        eye = torch.eye(X.shape[-2], dtype=X.dtype, device=X.device)
        ban = eye.to(torch.bool)
        R = X.unsqueeze(-2) - X.unsqueeze(-3)
        r = torch.where(ban, torch.full_like(R.norm(dim=-1), float("inf")), R.norm(dim=-1))
        P = torch.where(ban, torch.zeros_like(r), (-r / self.a).exp())
        M = P @ P.transpose(-1, -2)
        return (lam / self.a).unsqueeze(-1) * ((P * M).unsqueeze(-1) * (R / r.unsqueeze(-1))).sum(-2)


class LJ(Pair):
    """Lennard-Jones. With ε in K·kB and σ in Å, f comes out in K·kB/Å and a = f/m in amu.

    fmax is for the equilibration stage: it clamps the pair-force magnitude so that overlaps
    in a random start do not blow the system apart. Pass None for production.
    """

    conservative = True

    def __init__(self, eps, sigma, fmax=None):
        self.eps, self.sigma = float(eps), float(sigma)
        self.fmax = None if fmax is None else float(fmax)

    def f(self, R, V):
        r2 = (R * R).sum(-1).clamp_min(1e-12)
        a = 48.0 * self.eps * self.sigma ** 12 / r2 ** 7
        b = -24.0 * self.eps * self.sigma ** 6 / r2 ** 4
        mag = a + b
        if self.fmax is not None:
            mag = mag.clamp(-self.fmax, self.fmax)
        return mag.unsqueeze(-1) * R

    def u(self, R):
        r2 = (R * R).sum(-1).clamp_min(1e-12)
        x = self.sigma ** 2 / r2
        return 4.0 * self.eps * (x ** 6 - x ** 3)


class Trap:
    conservative = True

    def __init__(self, w2):
        self.w2 = float(w2)

    def f(self, X):
        return -self.w2 * X

    def u(self, X):
        return 0.5 * self.w2 * (X * X).sum(-1)
