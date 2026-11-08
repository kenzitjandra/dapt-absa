"""Per-class F1 and confusion matrices for the main runs (runs/, original DAPT checkpoint, lr x1).

Reads only the stored preds/gold, so no GPU and no model loading.
Run from the repo root:  python -I scripts/per_class.py
"""
import glob, json
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

LABELS = ["negative", "neutral", "positive"]          # same order as config.LABELS
ARCHS = ["roberta", "gpt2", "t5", "t5small"]
SEEDS = [7, 13, 42, 1337, 2024]

runs = {}
for f in glob.glob("runs/*.json"):
    r = json.load(open(f))
    runs[(r["arch"], r["init"], r["seed"])] = r

print(f"{'arch':8s} {'class':9s} {'base F1':>8s} {'DAPT F1':>8s} {'Delta':>7s}   per-seed Delta (seeds in order)")
for a in ARCHS:
    seeds = [s for s in SEEDS if (a, "base", s) in runs and (a, "dapt", s) in runs]
    if not seeds:
        continue
    pc = {}
    for init in ["base", "dapt"]:
        rows = []
        for s in seeds:
            r = runs[(a, init, s)]
            assert r["gold"] == runs[(a, "base", seeds[0])]["gold"], "gold order differs between runs"
            f = f1_score(r["gold"], r["preds"], labels=[0, 1, 2], average=None)
            # sanity: the mean of the per-class F1s must equal the stored macro-F1
            assert abs(f.mean() - r["test_f1"]) < 1e-6, (a, init, s)
            rows.append(f * 100)
        pc[init] = np.array(rows)                       # seeds x 3
    d = pc["dapt"] - pc["base"]
    for k, lab in enumerate(LABELS):
        per = ", ".join(f"{x:+.2f}" for x in d[:, k])
        print(f"{a:8s} {lab:9s} {pc['base'][:, k].mean():8.2f} {pc['dapt'][:, k].mean():8.2f} "
              f"{d[:, k].mean():+7.2f}   {per}  (seeds {seeds})")
    for init in ["base", "dapt"]:
        cm = sum(confusion_matrix(runs[(a, init, s)]["gold"], runs[(a, init, s)]["preds"],
                                  labels=[0, 1, 2]) for s in seeds)
        print(f"  {a} {init}: confusion matrix summed over {len(seeds)} seeds "
              f"(rows = gold neg/neu/pos, columns = predicted)")
        for k, lab in enumerate(LABELS):
            print(f"    {lab:9s} {cm[k]}")
    print()
