"""The ONE fine-tuning script. Every cell in the results table comes from here.

  python -m src.train_absa --arch roberta --init base --seed 42
  python -m src.train_absa --arch roberta --init dapt --seed 42

`--init base` loads the stock HF checkpoint. `--init dapt` loads our adapted one.
NOTHING ELSE CHANGES between them. That is the experiment.

Model selection is on VALIDATION macro-F1, and the test set is touched exactly
once per run, at the end. If anyone starts tuning against test, the Deltas become
meaningless.
"""
import argparse, json, os, random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup
from config import ARCHS, PATHS, LABELS
from data_mams import load_split
import absa_models


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def batches(rows, bs, shuffle):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [rows[j] for j in idx[i:i + bs]]


def evaluate(m, rows, bs):
    preds, gold = [], []
    for b in batches(rows, bs, False):
        enc = m.encode(b)
        preds += m.predict(enc).cpu().tolist()
        gold += [x["label"] for x in b]
    return (accuracy_score(gold, preds),
            f1_score(gold, preds, average="macro"),
            preds, gold)


def main(a):
    cfg = ARCHS[a.arch]
    set_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    path = cfg.hf_id if a.init == "base" else f"{PATHS['ckpt']}/{a.arch}_dapt"
    m = absa_models.build(a.arch, path, dev, cfg.max_len)

    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    print(f"[{a.arch}/{a.init}/s{a.seed}] train={len(tr)} val={len(va)} test={len(te)}")

    steps = (len(tr) // cfg.train_bs) * cfg.epochs
    opt = torch.optim.AdamW(m.model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = get_linear_schedule_with_warmup(
        opt, int(cfg.warmup_ratio * steps), steps)

    best_f1, best_state = -1, None
    for ep in range(cfg.epochs):
        m.model.train()
        for b in batches(tr, cfg.train_bs, True):
            loss = m.loss(m.encode(b))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.model.parameters(), cfg.grad_clip)
            opt.step(); sched.step(); opt.zero_grad()

        m.model.eval()
        acc, f1, _, _ = evaluate(m, va, cfg.eval_bs)
        print(f"  epoch {ep+1}  val_acc={acc:.4f}  val_f1={f1:.4f}")
        if f1 > best_f1:                       # select on VAL, never on test
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in m.model.state_dict().items()}

    m.model.load_state_dict(best_state)
    m.model.eval()
    acc, f1, preds, gold = evaluate(m, te, cfg.eval_bs)
    print(f"[{a.arch}/{a.init}/s{a.seed}] TEST acc={acc:.4f}  macro_f1={f1:.4f}")

    os.makedirs(PATHS["runs"], exist_ok=True)
    tag = f"{a.arch}_{a.init}_s{a.seed}"
    json.dump({"arch": a.arch, "init": a.init, "seed": a.seed,
               "val_f1": best_f1, "test_acc": acc, "test_f1": f1,
               "preds": preds, "gold": gold,          # kept for error analysis
               "n_params": sum(p.numel() for p in m.model.parameters())},
              open(f"{PATHS['runs']}/{tag}.json", "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=list(ARCHS))
    p.add_argument("--init", required=True, choices=["base", "dapt"])
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
