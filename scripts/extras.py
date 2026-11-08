"""Robustness analysis of Delta_DAPT. Reads runs/ (main) and runs_extra/ (DAPT replicates, LR variants).

  PYTHONPATH=src python scripts/extras.py

One line per (arch, DAPT checkpoint, learning-rate multiplier): Delta F1 of dapt vs the base runs
at the SAME lr multiplier, paired by seed, with a sentence-level bootstrap 95% CI (test rows of one
sentence are correlated, so sentences are resampled, not rows). The bootstrap covers test-set noise,
the t-test covers seed noise; neither covers DAPT-run noise except across the replicate lines.
Holm-adjusted p is given for the main per-architecture Deltas (original DAPT, lr x1) only.
"""
import glob, json
import numpy as np
from scipy import stats
from config import ARCHS, PATHS
from data_mams import load_split

B = 2000


def macro_f1(g, p):
    cm = np.bincount(g * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm)
    pr, rc = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    return np.mean(2 * pr * rc / np.maximum(pr + rc, 1e-12))


def load():
    cells = {}
    for d in (PATHS["runs"], PATHS["runs"] + "_extra"):
        for f in glob.glob(f"{d}/*.json"):
            r = json.load(open(f))
            lr0 = ARCHS[r["arch"]].lr
            mult = round(r.get("lr", lr0) / lr0, 3)            # main runs predate the "lr" field
            cells.setdefault((r["arch"], r["init"], r.get("dapt_seed"), mult), {})[r["seed"]] = r
    return cells


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    cells = load()
    rows = load_split("test")
    gold = np.array([r["label"] for r in rows])
    by_sent = {}
    for i, r in enumerate(rows):
        by_sent.setdefault(r["sentence"], []).append(i)
    groups = [np.array(v) for v in by_sent.values()]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
             for _ in range(B)]

    main_p = []
    for (arch, init, ds, mult), dr in sorted(cells.items(), key=str):
        br = cells.get((arch, "base", None, mult))
        if init != "dapt" or not br:
            continue
        seeds = sorted(set(dr) & set(br))
        if not seeds:
            continue
        for s in seeds:
            assert np.array_equal(dr[s]["gold"], gold) and np.array_equal(br[s]["gold"], gold), \
                f"{arch} s{s}: stored gold differs from MAMS test order"
        d = np.array([(dr[s]["test_f1"] - br[s]["test_f1"]) * 100 for s in seeds])
        p = stats.ttest_rel([dr[s]["test_f1"] for s in seeds], [br[s]["test_f1"] for s in seeds])[1] \
            if len(seeds) > 1 else float("nan")
        dp = [np.array(dr[s]["preds"]) for s in seeds]
        bp = [np.array(br[s]["preds"]) for s in seeds]
        boot = np.array([np.mean([macro_f1(gold[ix], a[ix]) - macro_f1(gold[ix], b[ix])
                                  for a, b in zip(dp, bp)]) for ix in draws]) * 100
        lo, hi = np.percentile(boot, [2.5, 97.5])
        name = "orig" if ds is None else f"d{ds}"
        sd = d.std(ddof=1) if len(d) > 1 else float("nan")
        print(f"{arch:8s} dapt={name:4s} lr x{mult:<4g} n={len(seeds)}  Delta F1 {d.mean():+.2f} +/- {sd:.2f}"
              f"  boot95 [{lo:+.2f}, {hi:+.2f}]  paired p={p:.3f}")
        print(f"{'':30s}per-seed: {', '.join(f'{x:+.2f}' for x in d)}   (seeds {seeds})")
        if ds is None and mult == 1.0 and not np.isnan(p):
            main_p.append((arch, p))

    if main_p:
        adj = holm(np.array([p for _, p in main_p]))
        print("\nHolm-adjusted p for the main Deltas (original DAPT, lr x1):")
        for (arch, p), a in zip(main_p, adj):
            print(f"  {arch:8s} raw p={p:.3f}  Holm p={a:.3f}")


if __name__ == "__main__":
    main()
