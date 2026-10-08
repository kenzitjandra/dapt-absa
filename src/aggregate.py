"""Turn 18 run-jsons into the paper's main table + the significance test.

The headline number is NOT accuracy. It is Delta_DAPT, and a Delta without error
bars is a rumour. With 3 seeds we report mean +/- std, and we test whether the
Deltas differ ACROSS architectures -- because "which topology benefits most" is a
claim about differences of differences.
"""
import glob, json, itertools
import numpy as np
from scipy import stats
from config import PATHS, ARCHS, DAPT_TOKEN_BUDGET


def load():
    runs = {}
    for f in glob.glob(f"{PATHS['runs']}/*.json"):
        r = json.load(open(f))
        runs.setdefault((r["arch"], r["init"]), []).append(r)
    return runs


def check_provenance(runs):
    """Warn loudly if the table mixes code versions, dirty trees, GPUs or short DAPT runs."""
    rs = [r for v in runs.values() for r in v]
    commit = lambda r: r.get("git_commit", r.get("commit"))
    gpu = lambda r: r.get("gpu_name", r.get("gpu"))

    commits = sorted({str(commit(r)) for r in rs})
    print(f"commits in this table: {commits}")
    if len(commits) > 1:
        print(f"WARNING: runs mix commits: {commits}")
    gpus = sorted({str(gpu(r)) for r in rs})
    if len(gpus) > 1:
        print(f"WARNING: runs mix gpus: {gpus}")
    if len({bool(r.get("smoke")) for r in rs}) > 1:
        print("WARNING: table mixes smoke and real runs")

    for r in rs:
        tag = f"{r['arch']}_{r['init']}_s{r['seed']}"
        c = commit(r)
        if c is None or "dirty" in str(c):
            print(f"WARNING: {tag} from uncommitted/unknown code ({c})")
        if r["init"] == "dapt" and r.get("dapt_tokens", 0) < 0.99 * DAPT_TOKEN_BUDGET:
            print(f"WARNING: {tag} fine-tuned from a {r.get('dapt_tokens', 0):,}-token DAPT run")


def sd(x):
    """Sample std (ddof=1); nan instead of a numpy warning when there is only one run."""
    return x.std(ddof=1) if len(x) > 1 else float("nan")


def main():
    runs = load()
    check_provenance(runs)
    deltas = {}
    print(f"{'arch':9s} {'init':5s} {'acc':>16s} {'macro-F1':>16s}")
    for arch in ARCHS:
        for init in ["base", "dapt"]:
            rs = runs.get((arch, init), [])
            if not rs:
                continue
            acc = np.array([r["test_acc"] for r in rs]) * 100
            f1 = np.array([r["test_f1"] for r in rs]) * 100
            print(f"{arch:9s} {init:5s} {acc.mean():7.2f} +/- {sd(acc):4.2f} "
                  f"{f1.mean():7.2f} +/- {sd(f1):4.2f}")
        b, d = runs.get((arch, "base")), runs.get((arch, "dapt"))
        if b and d:
            # pair by seed -- same seed, same data order, only init differs
            bm = {r["seed"]: r for r in b}
            dm = {r["seed"]: r for r in d}
            common = sorted(set(bm) & set(dm))
            if not common:
                print(f"  --> {arch}: no seed has both a base and a dapt run, skipping Delta\n")
                continue
            dd = np.array([(dm[s]["test_f1"] - bm[s]["test_f1"]) * 100 for s in common])
            deltas[arch] = dd
            if len(common) > 1:
                t, p = stats.ttest_rel([dm[s]["test_f1"] for s in common],
                                       [bm[s]["test_f1"] for s in common])
            else:
                t, p = float("nan"), float("nan")
            print(f"  --> Delta_DAPT(F1) = {dd.mean():+.2f} +/- {sd(dd):.2f} "
                  f"(paired t={t:.2f}, p={p:.3f}, n={len(common)})\n")

    print("\n=== does topology modulate the DAPT benefit? ===")
    for a, b in itertools.combinations(deltas, 2):
        t, p = stats.ttest_ind(deltas[a], deltas[b])
        verdict = "DIFFERENT" if p < 0.05 else "not distinguishable"
        print(f"{a:9s} ({deltas[a].mean():+.2f}) vs {b:9s} ({deltas[b].mean():+.2f})"
              f"  p={p:.3f}  -> {verdict}")

    print("\nstd is sample std (ddof=1).")
    print("NOTE: with n=3 seeds this test is badly underpowered. If two Deltas are "
          "close, the honest conclusion is 'not distinguishable', NOT 'X wins'.")


if __name__ == "__main__":
    main()
