"""Fill in the two report-backed numbers the paper uses: visible rate and coverage for the three observation regimes, and the per-truth RK4 order."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tl import obs, rt, scenes, sim  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINDS = ["magnus", "drag", "vlong", "quartic_r", "yukawa", "coulomb"]
AMPS = dict(zip(KINDS, [0.07, 0.55, 0.35, 0.18, 0.0024, 0.0019]))
OUT = []


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


def main():
    sc = scenes.build_scene("magnus", n=40)
    known, ext = sc["known"], sc["ext"]
    terms = [*known.terms, *sc["unknown"].terms]
    Xb, Vb = sc["X0"], sc["V0"]

    log("visible rate and configuration coverage by regime (truth = magnus, N=40)")
    log("  %-8s %-12s %-14s %-12s %s" %
        ("regime", "visible trajectory points", "seen at least once", "grid coverage", "uncovered"))
    Xs, Vs = sim.rk4(terms, ext, sc["X0"], sc["V0"], 6.0, 2.5e-4, every=4)
    ones = np.ones((Xs.shape[0], 40), bool)
    ref_pc = obs.pair_configs(obs.np_(Xs), obs.np_(Vs), ones)
    edges = obs.make_edges(ref_pc, nb=12)
    for regime in ("full", "mid", "hard"):
        cfg = scenes.make_cfg(regime)
        _, vis = obs.observe(Xs, cfg, np.random.default_rng(scenes.RNG_SEED))
        cov = obs.coverage(obs.pair_configs(obs.np_(Xs), obs.np_(Vs), vis), edges, nb=12)
        log("  %-8s %-12.6g %-14s %-12.6g %.6g" %
            (regime, float(vis.mean()), "%d of 40" % int(vis.any(axis=0).sum()),
             cov["grid_covered"], cov["grid_outside"]))

    log()
    log("per-truth RK4 convergence order (T=0.5, dt = 1e-3 / 5e-4 / 2.5e-4, reference dt=6.25e-5)")
    log("  %-12s %-14s %-14s %-14s %s" % ("truth", "err(dt=1e-3)", "err(5e-4)", "err(2.5e-4)", "measured order"))
    orders = []
    for k in KINDS:
        s1 = scenes.build_scene(k, n=40)
        t1 = [*s1["known"].terms, *s1["unknown"].terms]
        e1 = s1["ext"]
        ref = sim.rk4(t1, e1, Xb, Vb, 0.5, 6.25e-5)[0][-1]
        es = [float((sim.rk4(t1, e1, Xb, Vb, 0.5, dt)[0][-1] - ref).abs().max())
              for dt in (1e-3, 5e-4, 2.5e-4)]
        o = float(np.log2(es[0] / es[2]) / 2)
        orders.append(o)
        log("  %-12s %-14.6g %-14.6g %-14.6g %.4f" % (k, es[0], es[1], es[2], o))
    log("  -> range %.4f ~ %.4f; worst of the six %.4f" % (min(orders), max(orders), min(orders)))

    log()
    log("total-energy drift of the conservative truths (same convention as E0.3, per truth)")
    for k in scenes.POSITION_ONLY:
        s1 = scenes.build_scene(k, n=40)
        Xs, Vs = sim.rk4([*s1["known"].terms, *s1["unknown"].terms], s1["ext"],
                         Xb, Vb, 6.0, 2.5e-4, every=4)
        E = sim.energy([*s1["known"].terms, *s1["unknown"].terms], s1["ext"], Xs, Vs)
        log("  %-12s %.6g" % (k, float((E - E[0]).abs().max() / E.abs().mean())))

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "OBS_CONV.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
