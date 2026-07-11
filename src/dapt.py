"""Domain-Adaptive Pretraining. One script, three objectives.

python -m src.dapt --arch roberta   # ~35 min on an L4 at 50M tokens
python -m src.dapt --arch gpt2
python -m src.dapt --arch t5

GATE: this script prints held-out Yelp LM loss BEFORE and AFTER. If the loss did
not drop, DAPT did not happen and every downstream number is worthless. Do not
proceed to fine-tuning until you have seen that drop, and put it in the paper --
it is the evidence that our independent variable was actually applied.
"""
import argparse, math, os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup)
from config import ARCHS, PATHS, DAPT_BLOCK_LEN
from collators import get_collator

LOADER = {"mlm": AutoModelForMaskedLM,
          "clm": AutoModelForCausalLM,
          "span": AutoModelForSeq2SeqLM}


def blocks_loader(path, bs, collate, shuffle):
    arr = np.load(path)
    ds = TensorDataset(torch.from_numpy(arr.astype(np.int64)))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=True,
                      collate_fn=lambda b: collate([x[0].numpy() for x in b]))


@torch.no_grad()
def heldout_loss(model, loader, dev):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        tot += model(**batch).loss.item()
        n += 1
    model.train()
    return tot / max(n, 1)


def main(arch, resume=True):
    cfg = ARCHS[arch]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = LOADER[cfg.dapt_objective].from_pretrained(cfg.hf_id).to(dev)
    collate = get_collator(cfg.dapt_objective, tok)

    tr = blocks_loader(f"{PATHS['packed']}/{arch}_train.npy",   cfg.dapt_bs, collate, True)
    ho = blocks_loader(f"{PATHS['packed']}/{arch}_heldout.npy", cfg.dapt_bs, collate, False)

    out = f"{PATHS['ckpt']}/{arch}_dapt"
    os.makedirs(out, exist_ok=True)
    state_f = f"{out}/state.json"

    before = heldout_loss(model, ho, dev)
    print(f"[{arch}] held-out loss BEFORE dapt = {before:.4f}  (ppl {math.exp(before):.1f})")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.dapt_lr, weight_decay=0.01)
    total = len(tr)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.cuda.amp.GradScaler()

    start = 0
    if resume and os.path.exists(state_f):
        st = json.load(open(state_f))
        start = st["step"]
        model = LOADER[cfg.dapt_objective].from_pretrained(out).to(dev)
        print(f"[{arch}] RESUMING from step {start}")

    model.train()
    step = 0
    for batch in tr:
        if step < start:                      # cheap skip on resume
            step += 1
            continue
        batch = {k: v.to(dev) for k, v in batch.items()}
        with torch.cuda.amp.autocast():
            loss = model(**batch).loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        step += 1

        if step % 100 == 0:
            seen = step * cfg.dapt_bs * DAPT_BLOCK_LEN
            print(f"[{arch}] step {step}/{total} loss {loss.item():.4f} tokens {seen:,}")
        if step % 1000 == 0:                  # <-- mirror `out` to Drive!
            model.save_pretrained(out); tok.save_pretrained(out)
            json.dump({"step": step}, open(state_f, "w"))

    model.save_pretrained(out); tok.save_pretrained(out)
    after = heldout_loss(model, ho, dev)
    print(f"[{arch}] held-out loss AFTER  dapt = {after:.4f}  (ppl {math.exp(after):.1f})")
    print(f"[{arch}] LM loss drop = {before - after:+.4f}   <-- MUST be positive")
    json.dump({"step": step, "loss_before": before, "loss_after": after,
               "tokens": step * cfg.dapt_bs * DAPT_BLOCK_LEN},
              open(f"{out}/dapt_report.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=list(ARCHS))
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()
    main(a.arch, resume=not a.no_resume)
