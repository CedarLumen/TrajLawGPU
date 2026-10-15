import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tl import obs, rt, scenes, sim  # noqa: E402
from tl.forces import Mix  # noqa: E402

OUT = []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


KINDS = ["magnus", "drag", "vlong", "quartic_r", "yukawa", "coulomb"]
AMPS = dict(zip(KINDS, [0.07, 0.55, 0.35, 0.18, 0.0024, 0.0019]))


def grad_check(terms, ext, X, eps=1e-6):
    F = sim.accel(terms, ext, X, X * 0)
    G = torch.zeros_like(X)
    for i in range(X.shape[0]):
        for a in range(2):
            Xp, Xm = X.clone(), X.clone()
            Xp[i, a] += eps
            Xm[i, a] -= eps
            G[i, a] = (sim.total_potential(terms, ext, Xp).sum()
                       - sim.total_potential(terms, ext, Xm).sum()) / (2 * eps)
    return float((F + G).abs().max() / max(float(F.abs().max()), 1e-12))


def main():
    cfg = scenes.make_cfg()
    sc = scenes.build_scene("magnus", n=40)
    nb = len(KINDS)
    X0 = sc["X0"][None].repeat(nb, 1, 1)
    V0 = sc["V0"][None].repeat(nb, 1, 1)
    known, ext = sc["known"], sc["ext"]
    mix = Mix(KINDS, [AMPS[k] for k in KINDS])
    terms = [*known.terms, mix]

    log("device %s   dtype %s   numpy %s   torch %s" % (rt.DEV, rt.DT, np.__version__, torch.__version__))
    log("config N=40 d=2 T=%gs dt=%gs every=%d regime=%s" % (cfg["T"], cfg["dt"], cfg["every"], cfg["regime"]))

    log()
    log("E0.1  per truth")
    R = torch.tensor(np.random.default_rng(0).normal(0, 0.3, (40, 40, 2)))
    Vv = torch.tensor(np.random.default_rng(9).normal(0, 0.3, (40, 40, 2)))
    log("  %-12s %-22s %s" % ("truth", "Newton3 max|f(-R,-V)+f(R,V)|", "force split max|F-Fk-Fu|"))
    for k in KINDS:
        sk = scenes.build_scene(k, n=40)
        tk = [*sk["known"].terms, *sk["unknown"].terms]
        n3 = float(max(float((x.f(-R, -Vv) + x.f(R, Vv)).abs().max())
                       for x in [*sk["known"].terms, *sk["unknown"].terms]))
        Fk = sim.pair_accel(sk["known"].terms, R, Vv)
        Fu = sim.pair_accel(sk["unknown"].terms, R, Vv)
        dec = float((sim.pair_accel(tk, R, Vv) - Fk - Fu).abs().max())
        log("  %-12s %-22.3g %.3g" % (k, n3, dec))

    log()
    log("E0.1b Mix / Composite batching: per-batch agreement (max|Δa|)")
    d = float((sim.accel([known, mix], ext, X0, V0)
               - torch.stack([sim.accel([known, scenes.make_unknown(k, AMPS[k])], ext, X0[b], V0[b])
                              for b, k in enumerate(KINDS)])).abs().max())
    log("  %.3g" % d)

    T = 0.5
    ref = sim.rk4(terms, ext, X0, V0, T, 6.25e-5)[0][-1]
    errs = [float((sim.rk4(terms, ext, X0, V0, T, dt)[0][-1] - ref).abs().max())
            for dt in (1e-3, 5e-4, 2.5e-4)]
    log("E0.2 position error dt=1e-3 / 5e-4 / 2.5e-4 (max over 6 truths)  %s"
        % "  ".join("%.6g" % e for e in errs))
    log("     measured order (RK4 expects ~4)  %.4f" % (np.log2(errs[0] / errs[2]) / 2))

    t0 = time.time()
    Xs, Vs = sim.rk4(terms, ext, X0, V0, cfg["T"], cfg["dt"], every=cfg["every"])
    rt.sync()
    log("     batched integration: %d truths x %d frames, %.1fs" % (nb, Xs.shape[0], time.time() - t0))

    log()
    log("E0.3 per truth: total-energy drift / |v|max / |x|max")
    for b, k in enumerate(KINDS):
        Xb, Vb = Xs[:, b], Vs[:, b]
        if k in scenes.POSITION_ONLY:
            E = sim.energy([*known.terms, scenes.make_unknown(k, AMPS[k])], ext, Xb, Vb)
            e = "%.6g" % float((E - E[0]).abs().max() / E.abs().mean())
        else:
            e = "n/a (non-conservative terms)"
        log("  %-12s %-16s |v|max %.6g  |x|max %.6g"
            % (k, e, float(Vb.abs().max()), float(Xb.abs().max())))

    log("E0.3b gradient check max|F + grad U| / |F| (force = -grad U)")
    for k in scenes.POSITION_ONLY:
        s1 = scenes.build_scene(k, n=40)
        X = s1["X0"] + rt.t(np.random.default_rng(7).normal(0, 0.02, (40, 2)))
        log("  %-12s %.6g" % (k, grad_check(s1["known"].terms, s1["ext"], X)))

    log()
    log("E0.5 observation regime / configuration coverage / degeneracy gate")
    ones = np.ones((Xs.shape[0], Xs.shape[2]), bool)
    for b, k in enumerate(KINDS):
        Xb, Vb = Xs[:, b], Vs[:, b]
        ref_pc = obs.pair_configs(obs.np_(Xb), obs.np_(Vb), ones)
        edges = obs.make_edges(ref_pc, nb=12)
        _, vis = obs.observe(Xb, cfg, np.random.default_rng(scenes.RNG_SEED))
        cov = obs.coverage(obs.pair_configs(obs.np_(Xb), obs.np_(Vb), vis), edges, nb=12)
        unk = scenes.make_unknown(k, AMPS[k])
        sr = obs.sample_ratio(known, unk, Xb, Vb)
        sr = sr[np.isfinite(sr)]
        log()
        log("=" * 78)
        log("E0  truth=%-11s (%s)" % (k, "velocity-dependent" if k in scenes.VELOCITY_DEPENDENT else "position-dependent"))
        log("=" * 78)
        log("  %-44s %s" % ("visible trajectory points / particles seen at least once",
                            "%.3f / %d of 40" % (float(vis.mean()), int(vis.any(axis=0).sum()))))
        log("  %-44s %.6g  %.6g" % ("(r, v∥, v⊥) grid coverage / outside", cov["grid_covered"], cov["grid_outside"]))
        log("  %-44s %.6g  %.6g  %.6g" % ("|cos(x,v)| median / oblique / head-on",
                                          cov["cabs_median"], cov["frac_oblique"], cov["frac_head_on"]))
        log("  %-44s %d" % ("pairs used in sampling", cov["n_pairs"]))
        log("  %-44s %.6g" % ("frac_unknown (primary gate, target 0.25~0.6)",
                              obs.frac_unknown(known, unk, Xb, Vb)))
        log("  %-44s %.6g" % ("|Fu|/|Fk| median (secondary diagnostic, not a gate)",
                              float(np.median(sr)) if sr.size else float("nan")))

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "E0.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
