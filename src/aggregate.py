"""Turn 18 run-jsons into the paper's main table + the significance test.

The headline number is NOT accuracy. It is Delta_DAPT, and a Delta without error
bars is a rumour. With 3 seeds we report mean +/- std, and we test whether the
Deltas differ ACROSS architectures -- because "which topology benefits most" is a
claim about differences of differences.
"""
import glob, json, itertools
import numpy as np
from scipy import stats
from config import PATHS, ARCHS


def load():
    runs = {}
    for f in glob.glob(f"{PATHS['runs']}/*.json"):
        r = json.load(open(f))
        runs.setdefault((r["arch"], r["init"]), []).append(r)
    return runs


def main():
    runs = load()
    deltas = {}
    print(f"{'arch':9s} {'init':5s} {'acc':>16s} {'macro-F1':>16s}")
    for arch in ARCHS:
        for init in ["base", "dapt"]:
            rs = runs.get((arch, init), [])
            if not rs:
                continue
            acc = np.array([r["test_acc"] for r in rs]) * 100
            f1 = np.array([r["test_f1"] for r in rs]) * 100
            print(f"{arch:9s} {init:5s} {acc.mean():7.2f} +/- {acc.std():4.2f} "
                  f"{f1.mean():7.2f} +/- {f1.std():4.2f}")
        b, d = runs.get((arch, "base")), runs.get((arch, "dapt"))
        if b and d:
            # pair by seed -- same seed, same data order, only init differs
            bm = {r["seed"]: r for r in b}
            dm = {r["seed"]: r for r in d}
            common = sorted(set(bm) & set(dm))
            dd = np.array([(dm[s]["test_f1"] - bm[s]["test_f1"]) * 100 for s in common])
            deltas[arch] = dd
            t, p = stats.ttest_rel([dm[s]["test_f1"] for s in common],
                                   [bm[s]["test_f1"] for s in common])
            print(f"  --> Delta_DAPT(F1) = {dd.mean():+.2f} +/- {dd.std():.2f} "
                  f"(paired t={t:.2f}, p={p:.3f}, n={len(common)})\n")

    print("\n=== does topology modulate the DAPT benefit? ===")
    for a, b in itertools.combinations(deltas, 2):
        t, p = stats.ttest_ind(deltas[a], deltas[b])
        verdict = "DIFFERENT" if p < 0.05 else "not distinguishable"
        print(f"{a:9s} ({deltas[a].mean():+.2f}) vs {b:9s} ({deltas[b].mean():+.2f})"
              f"  p={p:.3f}  -> {verdict}")

    print("\nNOTE: with n=3 seeds this test is badly underpowered. If two Deltas are "
          "close, the honest conclusion is 'not distinguishable', NOT 'X wins'.")


if __name__ == "__main__":
    main()
