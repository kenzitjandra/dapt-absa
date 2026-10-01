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
from config import ARCHS, PATHS, DAPT_BLOCK_LEN, DAPT_SAVE_EVERY
from collators import get_collator

LOADER = {"mlm": AutoModelForMaskedLM,
          "clm": AutoModelForCausalLM,
          "span": AutoModelForSeq2SeqLM}


def blocks_loader(path, bs, collate, shuffle, skip=0):
    """Returns (loader, total_batches). Shuffle order is fixed (seed 0) so a resume
    can drop exactly the `skip` batches already trained on, without collating them."""
    arr = np.load(path)
    total = len(arr) // bs
    if shuffle:
        arr = arr[np.random.default_rng(0).permutation(len(arr))]
    ds = TensorDataset(torch.from_numpy(arr[skip * bs:].astype(np.int64)))
    return DataLoader(ds, batch_size=bs, shuffle=False, drop_last=True,
                      collate_fn=lambda b: collate([x[0].numpy() for x in b])), total


@torch.no_grad()
def heldout_loss(model, loader, dev):
    """Fixed mask seed so BEFORE and AFTER score the exact same corruptions."""
    model.eval()
    tot, n = 0.0, 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
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

    collate = get_collator(cfg.dapt_objective, tok)

    out = f"{PATHS['ckpt']}/{arch}_dapt"
    os.makedirs(out, exist_ok=True)
    state_f, trainer_f = f"{out}/state.json", f"{out}/trainer_state.pt"
    report_f = f"{out}/dapt_report.json"
    if os.path.exists(report_f):              # a report must only ever describe a finished run
        os.remove(report_f)

    st = json.load(open(state_f)) if resume and os.path.exists(state_f) else None
    start = st["step"] if st else 0
    # Model is loaded BEFORE the optimizer is built, so the optimizer owns these params.
    model = LOADER[cfg.dapt_objective].from_pretrained(out if st else cfg.hf_id).to(dev)

    tr, total = blocks_loader(f"{PATHS['packed']}/{arch}_train.npy", cfg.dapt_bs, collate,
                              True, skip=start)
    ho, _ = blocks_loader(f"{PATHS['packed']}/{arch}_heldout.npy", cfg.dapt_bs, collate, False)

    if st:
        before = st["loss_before"]            # measured on the base model at the original start
        print(f"[{arch}] RESUMING from step {start}/{total}")
    else:
        before = heldout_loss(model, ho, dev)
    print(f"[{arch}] held-out loss BEFORE dapt = {before:.4f}  (ppl {math.exp(before):.1f})")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.dapt_lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    if st:
        ts = torch.load(trainer_f, map_location=dev)
        if ts["step"] != start:               # crash mid-save: model/trainer/state disagree
            raise SystemExit(f"[{arch}] checkpoint torn (trainer step {ts['step']} vs state "
                             f"{start}). Rerun with --no-resume.")
        opt.load_state_dict(ts["opt"]); sched.load_state_dict(ts["sched"])

    model.train()
    step = start
    for batch in tr:
        batch = {k: v.to(dev) for k, v in batch.items()}
        # bf16 for every arch: T5 overflows in fp16, and one recipe keeps DAPT uniform.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(); sched.step()
        step += 1

        if step % 100 == 0:
            seen = step * cfg.dapt_bs * DAPT_BLOCK_LEN
            print(f"[{arch}] step {step}/{total} loss {loss.item():.4f} tokens {seen:,}")
        if step % DAPT_SAVE_EVERY == 0:                  # <-- mirror `out` to Drive!
            # order matters: trainer, model, state. Any crash in between leaves
            # trainer step != state step, which resume refuses instead of mixing.
            torch.save({"step": step, "opt": opt.state_dict(), "sched": sched.state_dict()},
                       trainer_f)
            model.save_pretrained(out); tok.save_pretrained(out)
            json.dump({"step": step, "loss_before": before}, open(state_f, "w"))  # written last

    model.save_pretrained(out); tok.save_pretrained(out)
    after = heldout_loss(model, ho, dev)
    print(f"[{arch}] held-out loss AFTER  dapt = {after:.4f}  (ppl {math.exp(after):.1f})")
    print(f"[{arch}] LM loss drop = {before - after:+.4f}   <-- MUST be positive")
    json.dump({"step": step, "loss_before": before, "loss_after": after,
               "tokens": step * cfg.dapt_bs * DAPT_BLOCK_LEN},
              open(report_f, "w"), indent=2)
    for f in (state_f, trainer_f):            # finished: a rerun must start fresh, not "resume"
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=list(ARCHS))
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()
    main(a.arch, resume=not a.no_resume)
