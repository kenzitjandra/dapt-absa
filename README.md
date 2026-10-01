# DAPT × Transformer Topology for ABSA

Four models (RoBERTa, GPT-2, T5-base, T5-small scale control) × {baseline, DAPT} × 3 seeds
= **24 fine-tuning runs** + 4 DAPT runs.
The result of this paper is not accuracy. It is **Δ_DAPT**, and a Δ without error
bars is a rumour.

## Setup (Colab Pro, L4)

```python
!git clone https://github.com/<team>/dapt-absa.git && cd dapt-absa
!pip install -q -r dapt-absa/requirements.txt     # pinned. Then Runtime > Restart (numpy<2.0).
```

After the restart, run this cell at the top of every notebook:

```python
%cd /content/dapt-absa
%env PYTHONPATH=src                     # modules use `from config import ...`
from google.colab import drive; drive.mount('/content/drive')
# ONE shared Drive folder for all three of us. Everything off Colab disk survives disconnects.
# Owner shares the folder; the other two use "Organize > Add shortcut" into My Drive so
# this same path resolves on every account.
import os
S = "/content/drive/MyDrive/dapt-absa-shared"
for d in ("data", "checkpoints", "runs"):
    os.makedirs(f"{S}/{d}", exist_ok=True)
    if not os.path.exists(d):
        os.symlink(f"{S}/{d}", d)
    assert os.path.islink(d), f"{d} is a real dir on Colab disk; move it to {S}/{d}"
import torch, transformers; print(torch.__version__, transformers.__version__,
                                  torch.cuda.get_device_name())
!git describe --always --dirty          # must be a clean, tagged commit for real runs
```

Smoke tests: never edit `config.py` for them. Run results from a dirty tree are flagged by
`aggregate.py`; move smoke run JSONs out of `runs/` before aggregating.

## Run order

```bash
export PYTHONPATH=src           # from the repo root; `data/` paths are relative to it

# 0. MAMS. Download the ATSA xml from github.com/siat-nlp/MAMS-for-ABSA -> data/mams/
python -m src.data_mams          # sanity report. paste these numbers in the paper.

# 1. Corpus (once, ~30 min)
python -m src.build_corpus --stage text --source hf        # week 1: unblock the pipeline
python -m src.build_corpus --stage text --source json --root data/yelp_raw   # real runs
for A in roberta gpt2 t5 t5small; do python -m src.build_corpus --stage pack --arch $A; done

# 2. Baselines FIRST. 12 runs, ~15 min each.
for A in roberta gpt2 t5 t5small; do for S in 13 42 1337; do
  python -m src.train_absa --arch $A --init base --seed $S; done; done

# 3. DAPT. 50M tokens each. ~35 min (roberta/gpt2), ~90 min (t5) on L4.
for A in roberta gpt2 t5 t5small; do python -m src.dapt --arch $A; done

# 4. DAPT fine-tunes. 12 runs. Refuses to start unless dapt_report.json passes Gate 2.
for A in roberta gpt2 t5 t5small; do for S in 13 42 1337; do
  python -m src.train_absa --arch $A --init dapt --seed $S; done; done

# 5. The table + the significance test
python -m src.aggregate
```

## Two gates. Do not skip them.

**Gate 1 — baseline sanity.** RoBERTa-base must land around **82–84% accuracy** on
MAMS-ATSA test. If it doesn't, our fine-tuning is broken and every Δ computed
downstream is measuring our bug, not domain adaptation. Fix this before touching DAPT.

**Gate 2 — did DAPT actually happen?** `src/dapt.py` prints held-out Yelp LM loss
before and after. **If the loss didn't drop, the independent variable was never
applied.** Report this drop in the paper; it is the evidence the treatment was real.

## Decisions this repo bakes in (and the paper must defend)

**Corpus is Yelp, not Amazon.** MAMS is built from Citysearch *restaurant* reviews.
Gururangan et al. — the paper we cite *for* DAPT — is the paper showing gains come
from domain *proximity*. Amazon reviews talk about shipping and battery life;
restaurant reviews talk about ambience and portions. "Reviews" is a genre, not a
domain. **The methodology section still says Amazon. Fix it.**

**Parameter matching is dropped.** RoBERTa-base 125M, GPT-2 small 124M, T5-base
**220M**. That is a 76% gap and "approximately constant" is indefensible. Instead we
match on **DAPT tokens seen (50M)**, report parameter counts honestly, and run
**t5small** (60M) as a scale control so nobody can say the T5 result is just size.

**T5, not "BART/T5".** The intro and related work say "BART/T5" in three places.
Pick one, say T5 everywhere. Span corruption is native to T5 and clean to reimplement.

**Constrained scoring.** T5 and GPT-2 never free-generate. We score the three label
strings and take the argmax. Otherwise we'd be measuring decoding strategy, not
architecture, and a format error would penalise the generative models for something
that has nothing to do with domain adaptation.

**Learning rate differs per architecture, and that is deliberate.** T5 needs ~1e-4;
RoBERTa needs ~2e-5. Forcing one LR on all three would cripple somebody and we'd
report the crippling as an architecture finding. What must be *identical* is the
recipe **between `base` and `dapt` for a given architecture** — because that's the
contrast Δ is computed over. Say this explicitly in the paper; a reviewer will ask.

## The limitation we have to state out loud

MLM supervises ~15% of tokens per pass; CLM supervises ~100%. The three objectives
are not equally dense, and we cannot normalise that away — the objective *is* part
of the architecture. We control the knob we can (tokens processed) and name the rest
as a confound rather than pretending it doesn't exist.

## The result we might get

All three Δs land within noise of each other. That is a real possibility. It is
publishable as a negative result — "topology does not meaningfully modulate DAPT
benefit at base scale" — **but only if the experiment is tight enough that the null
is believable.** That is exactly what the multi-seed / matched-budget / matched-decoding
discipline buys us. With one seed per cell, a +0.8 Δ means nothing and we'd have no
paper either way.
