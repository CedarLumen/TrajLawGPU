import numpy as np

from . import rt
from .forces import (Bond, Composite, Drag, Magnus, QuarticR, SoftCoulomb,
                     SoftYukawa, Trap, VLongitudinal)

RNG_SEED = 20261014
BOND_K = 2.0
BOND_R0 = 0.30
TRAP_W2 = 0.12

UNK = {
    "magnus": lambda: Magnus(k=0.07),
    "drag": lambda: Drag(g=0.55),
    "vlong": lambda: VLongitudinal(b=0.35),
    "quartic_r": lambda: QuarticR(lam=0.18),
    "yukawa": lambda: SoftYukawa(A=0.0024, lam=0.30),
    "coulomb": lambda: SoftCoulomb(C=0.0019),
}
VELOCITY_DEPENDENT = ("magnus", "drag", "vlong")
POSITION_ONLY = ("quartic_r", "yukawa", "coulomb")


def make_unknown(variant, amp):
    if variant == "magnus":
        return Composite([Magnus(k=amp)])
    if variant == "drag":
        return Composite([Drag(g=amp)])
    if variant == "vlong":
        return Composite([VLongitudinal(b=amp)])
    if variant == "quartic_r":
        return Composite([QuarticR(lam=amp)])
    if variant == "yukawa":
        return Composite([SoftYukawa(A=amp, lam=0.30)])
    if variant == "coulomb":
        return Composite([SoftCoulomb(C=amp)])
    raise ValueError(variant)


def build_scene(variant="magnus", d=2, n=40, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    N = int(n)
    X0 = rng.uniform(-0.45, 0.45, size=(N, d))
    if d > 2:
        X0 = np.concatenate([X0, rng.uniform(-0.2, 0.2, size=(N, d - 2))], axis=-1)
    D = np.linalg.norm(X0[:, None, :] - X0[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    adj = np.zeros((N, N))
    for i in range(N):
        for j in np.argsort(D[i])[:3]:
            adj[i, j] = adj[j, i] = 1.0
    known = Composite([Bond(k=BOND_K, r0=BOND_R0, adj=adj)], name="bonded")
    ext = [Trap(w2=TRAP_W2)]
    unknown = Composite([UNK[variant]()], name="unknown")
    V0 = rng.normal(0.0, 0.40, size=(N, d))
    return dict(known=known, unknown=unknown, ext=ext, adj=adj,
                X0=rt.t(X0), V0=rt.t(V0), d=d, N=N, variant=variant, rng=rng)


OBS_REGIMES = {
    "full": dict(obs_mode="full"),
    "easy": dict(obs_mode="window", win_lo=[-0.75, -0.75], win_hi=[0.75, 0.15]),
    "mid": dict(obs_mode="window", win_lo=[-0.30, -0.30], win_hi=[0.30, 0.30]),
    "hard": dict(obs_mode="window", win_lo=[-0.20, 0.05], win_hi=[0.20, 0.45]),
    "random": dict(obs_mode="random", n_obs=14),
}

BASE_CFG = dict(
    T=6.0, dt=2.5e-4, every=4, sigma_x=0.0, n_obs=14, regime="mid",
    box_lo=[-1.0, -1.0], box_hi=[1.0, 1.0],
)


def make_cfg(regime=None, **over):
    cfg = dict(BASE_CFG)
    name = regime or cfg["regime"]
    cfg.update(OBS_REGIMES[name])
    cfg["regime"] = name
    cfg.update(over)
    return cfg


DEFAULT_CFG = make_cfg()
