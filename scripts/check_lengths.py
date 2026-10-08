"""Count MAMS examples whose prompt is longer than the truncation limit.

Uses the real prompt templates from absa_models.py (GPT2ABSA.PROMPT, T5ABSA.TEMPLATE),
so this cannot drift from what the models actually see. Loads tokenizers only.

Run from the repo root with PYTHONPATH=src:
    python scripts/check_lengths.py
"""
import argparse
from transformers import AutoTokenizer
from absa_models import GPT2ABSA, T5ABSA
from data_mams import load_split

LIMIT = 124   # GPT-2 truncates the prompt at max_len - 4 = 124 (see GPT2ABSA._build)


def lengths(tok, fmt, rows):
    # no truncation here: we want the TRUE length, then compare to the limit
    return [len(tok(fmt(r))["input_ids"]) for r in rows]


def report(name, ls, limits):
    ls_sorted = sorted(ls)
    parts = "  ".join(f">{lim}: {sum(l > lim for l in ls):4d}" for lim in limits)
    print(f"  {name:6s} n={len(ls):6d}  max={ls_sorted[-1]:4d}  "
          f"p99={ls_sorted[int(0.99 * (len(ls) - 1))]:4d}  {parts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt2", default="gpt2")
    ap.add_argument("--t5", default="t5-base")   # t5-small shares the same vocab
    a = ap.parse_args()

    gtok = AutoTokenizer.from_pretrained(a.gpt2)
    ttok = AutoTokenizer.from_pretrained(a.t5)
    gfmt = lambda r: GPT2ABSA.PROMPT.format(s=r["sentence"], a=r["aspect"])
    tfmt = lambda r: T5ABSA.TEMPLATE.format(s=r["sentence"], a=r["aspect"])

    total = {"gpt2": 0, "t5": 0}
    for split in ["train", "val", "test"]:
        rows = load_split(split)
        print(f"{split}:")
        gl = lengths(gtok, gfmt, rows)
        tl = lengths(ttok, tfmt, rows)
        report("gpt2", gl, [LIMIT])
        report("t5", tl, [LIMIT, 128])
        total["gpt2"] += sum(l > LIMIT for l in gl)
        total["t5"] += sum(l > LIMIT for l in tl)

    print(f"\nTOTAL examples over {LIMIT} tokens:  GPT-2 = {total['gpt2']}   T5 = {total['t5']}")
    print("GPT-2 over 124 = prompt gets truncated. T5 is cut at 128 (see the >128 column).")


if __name__ == "__main__":
    main()
