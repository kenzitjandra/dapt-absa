"""Yelp DAPT corpus: clean -> dedupe -> tokenize -> PACK to a fixed token budget.

Two sources:
  --source hf    : HuggingFace `yelp_review_full`. Zero friction. Use this to get
                   the pipeline running THIS WEEK.
  --source json  : official Yelp Open Dataset. Streams review.json and keeps only
                   reviews whose business is categorised Restaurants/Food. This is
                   the one that actually matches MAMS (Citysearch restaurant reviews)
                   and it is what goes in the paper.

PACKING is not an optimisation, it is a correctness issue. If you pad each review
to 256 you burn ~half your 50M token budget on <pad>, and "50M tokens" in the paper
becomes a lie. We concatenate the review stream with EOS separators and chop into
contiguous blocks.
"""
import argparse, json, os, re, hashlib
import numpy as np
from config import DAPT_BLOCK_LEN, DAPT_TOKEN_BUDGET, ARCHS, PATHS

URL = re.compile(r"https?://\S+|www\.\S+")
WS  = re.compile(r"\s+")


def clean(text):
    if not text:
        return None
    t = URL.sub(" ", text)
    t = t.replace("\\n", " ")
    t = WS.sub(" ", t).strip()
    if len(t) < 40:                       # "Great!" teaches the model nothing
        return None
    if len(t) > 4000:
        t = t[:4000]
    # kill reviews that are mostly non-alpha (menus, phone numbers, star spam)
    alpha = sum(c.isalpha() or c.isspace() for c in t) / len(t)
    if alpha < 0.85:
        return None
    return t


def iter_hf():
    from datasets import load_dataset
    ds = load_dataset("yelp_review_full", split="train", streaming=True)
    for ex in ds:
        yield ex["text"]


def iter_official(root):
    """root = dir containing yelp_academic_dataset_{business,review}.json"""
    keep = set()
    bpath = os.path.join(root, "yelp_academic_dataset_business.json")
    with open(bpath, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            cats = b.get("categories") or ""
            if "Restaurants" in cats or "Food" in cats:
                keep.add(b["business_id"])
    print(f"[corpus] restaurant/food businesses: {len(keep):,}")

    rpath = os.path.join(root, "yelp_academic_dataset_review.json")
    with open(rpath, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["business_id"] in keep:
                yield r["text"]


def build_text(source, root, out):
    seen, kept = set(), 0
    it = iter_hf() if source == "hf" else iter_official(root)
    # 50M tokens ~= 400k reviews. Take 600k raw to survive cleaning + dedupe,
    # plus a held-out tail for the perplexity check.
    target_reviews = 650_000
    with open(out, "w", encoding="utf-8") as f:
        for text in it:
            t = clean(text)
            if t is None:
                continue
            h = hashlib.md5(t[:200].encode()).hexdigest()   # near-dup on prefix
            if h in seen:
                continue
            seen.add(h)
            f.write(t.replace("\n", " ") + "\n")
            kept += 1
            if kept % 50_000 == 0:
                print(f"[corpus] kept {kept:,}")
            if kept >= target_reviews:
                break
    print(f"[corpus] DONE {kept:,} reviews -> {out}")


def pack(arch, corpus_path, out_dir, budget=None, heldout_blocks=2000):
    if budget is None:
        budget = DAPT_TOKEN_BUDGET      # read the CURRENT value, not a definition-time snapshot
    """Tokenise the review stream and chop into contiguous DAPT_BLOCK_LEN blocks.

    Every architecture gets the SAME budget of *real* tokens. Different tokenizers
    will consume a different NUMBER OF REVIEWS to reach 50M tokens -- that is fine
    and expected; what we hold constant is tokens seen, not reviews seen.
    """
    from transformers import AutoTokenizer
    cfg = ARCHS[arch]
    tok = AutoTokenizer.from_pretrained(cfg.hf_id)
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id

    need = budget + heldout_blocks * DAPT_BLOCK_LEN
    buf, blocks = [], []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            ids = tok(line.strip(), add_special_tokens=False)["input_ids"]
            buf.extend(ids)
            if eos is not None:
                buf.append(eos)
            while len(buf) >= DAPT_BLOCK_LEN:
                blocks.append(buf[:DAPT_BLOCK_LEN])
                buf = buf[DAPT_BLOCK_LEN:]
            if len(blocks) * DAPT_BLOCK_LEN >= need:
                break

        arr = np.array(blocks, dtype=np.int32)
    total_blocks = len(arr)
    # if the corpus is smaller than the budget (smoke test), carve out a small
    # held-out slice and use the rest for training instead of going negative.
    want_train = budget // DAPT_BLOCK_LEN
    if total_blocks <= heldout_blocks + 1:
        raise SystemExit(f"[pack:{arch}] corpus too small: only {total_blocks} blocks. "
                         f"Use more reviews.")
    if want_train + heldout_blocks > total_blocks:
        heldout_blocks = max(1, min(heldout_blocks, total_blocks // 10))
        n_train = total_blocks - heldout_blocks
        print(f"[pack:{arch}] corpus smaller than budget; using {n_train} train blocks "
              f"({n_train*DAPT_BLOCK_LEN:,} tokens) instead of the full {budget:,}")
    else:
        n_train = want_train
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{arch}_train.npy"), arr[:n_train])
    np.save(os.path.join(out_dir, f"{arch}_heldout.npy"), arr[n_train:n_train + heldout_blocks])
    print(f"[pack:{arch}] train blocks={n_train:,} tokens={n_train*DAPT_BLOCK_LEN:,} "
          f"heldout={heldout_blocks:,} blocks  (total {total_blocks:,})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["text", "pack"], required=True)
    p.add_argument("--source", choices=["hf", "json"], default="hf")
    p.add_argument("--root", default="data/yelp_raw")
    p.add_argument("--arch", choices=list(ARCHS))
    a = p.parse_args()

    if a.stage == "text":
        build_text(a.source, a.root, PATHS["corpus"])
    else:
        pack(a.arch, PATHS["corpus"], PATHS["packed"])
