# DAPT × Transformer Topology for ABSA

Three architectures × {baseline, DAPT} × 3 seeds = **18 fine-tuning runs**.
The result of this paper is not accuracy. It is **Δ_DAPT**, and a Δ without error
bars is a rumour.

## Setup (Colab Pro, L4)

Before running anything: **Runtime → Change runtime type → Runtime version: 2025.07**
(Python 3.11). The default "Latest" runtime uses Python 3.13, which has no prebuilt
wheels for the pinned packages, so `pip install` fails with "Failed building wheel
for tokenizers". Also pick a GPU (L4 on Colab Pro).

One-time: create a GitHub token (Settings → Developer settings → Personal access
tokens, **Contents: Read and write**). In Colab, open the key icon (Secrets), add it
as `GITHUB_TOKEN` and enable notebook access. Never paste the token into a cell.

Keep your notebook in your **own** Drive, not in the shared folder and not in Git.
Run Cell 1 and Cell 2 at the start of every session.

**Cell 1: install, then restart**

```python
from google.colab import drive, userdata
drive.mount('/content/drive')

token = userdata.get('GITHUB_TOKEN')
!rm -rf /content/dapt-absa
!git clone -q https://{token}@github.com/<team>/dapt-absa.git /content/dapt-absa
%cd /content/dapt-absa
!git checkout main                      # your branch while developing
!pip install -q -r requirements.txt
```

Then click **Runtime → Restart session** (`numpy<2.0` only takes effect after a
restart; the cloned files stay on disk).

The install ends with a pip "dependency resolver" error about gcsfs, opencv, thinc
and umap-learn. This is harmless: the project does not use those packages. A real
failure contains "Failed building wheel".

**Cell 2: connect folders, set identity, check everything**

```python
%cd /content/dapt-absa
%env PYTHONPATH=src
SHARED = '/content/drive/MyDrive/dapt-absa-shared'

for d in ['data', 'checkpoints', 'checkpoints_smoke', 'runs', 'runs_smoke', 'logs']:
    !rm -rf {d} && ln -s {SHARED}/{d} {d}

!git config user.name  "Your Name"
!git config user.email "you@example.com"

import torch, transformers, numpy
print('GPU         :', torch.cuda.get_device_name(0))
print('torch       :', torch.__version__)
print('transformers:', transformers.__version__)
print('numpy       :', numpy.__version__)
!echo "commit      : $(git rev-parse --short HEAD)"
```

Check the printout every time: **L4**, versions matching `requirements.txt`, and the
commit you expected. If `import torch` fails with "module functions cannot set
METH_CLASS or METH_STATIC", delete the runtime, run Cell 1, restart the session,
and run Cell 2 first without importing torch anywhere else.

**Running anything:** always save a log to Drive.

```python
%env SMOKE=1
!python -m src.train_absa --arch roberta --init base --seed 42 2>&1 | tee logs/smoke_roberta_base_s42.log
```

Git holds the code. The shared Drive folder holds everything the code produces.
Colab deletes its local disk when a session ends, so push code early and often.
Do not run `git remote -v` in a shared notebook: the token is in the URL.

## Setup (local, Windows / PowerShell)

```powershell
git clone https://github.com/<team>/dapt-absa.git; cd dapt-absa
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # pinned. Do not use unpinned installs.
$env:PYTHONPATH = "src"                  # REQUIRED every new shell: modules do `from config import ...`
```

Download the MAMS-ATSA XML files into `data/mams/` (`train.xml`, `val.xml`, `test.xml`),
then `python -m src.data_mams` to check them. `data/`, `checkpoints/` and `runs/` are
git-ignored; never commit them.

## Smoke test (minutes, any GPU)

Checks the whole pipeline (pack, DAPT, resume, fine-tune) on a tiny budget. Results go
to `*_smoke` paths and never touch real outputs. **Do not edit `config.py` for this**;
set `SMOKE=1` in the shell.

```powershell
$env:SMOKE = "1"
$env:DAPT_BS = "8"        # only for GPUs under ~8 GB; 32 OOMs on 4 GB (CUBLAS_STATUS_EXECUTION_FAILED)
Copy-Item data\yelp_clean.txt data\yelp_clean_smoke.txt       # or build it with build_corpus --stage text
python -m src.build_corpus --stage pack --arch roberta
python -m src.dapt --arch roberta
python -m src.train_absa --arch roberta --init dapt --seed 42
```

Expect: `LM loss drop` is positive, and `runs_smoke/roberta_dapt_s42.json` has non-zero
`dapt_tokens`. Test accuracy is **not** expected to beat base at this scale.

Test resume: Ctrl+C after `checkpoints_smoke/roberta_dapt/state.json` appears (a
checkpoint is written every 50 steps in smoke mode), rerun the same `dapt` command, and
look for `RESUMING from step N`. `--no-resume` forces a fresh start. To leave smoke
mode: `Remove-Item Env:SMOKE, Env:DAPT_BS`.

For real runs leave `DAPT_BS` unset (default 32). Move smoke JSONs out of `runs/`
before running `aggregate.py`.

## Run order

```bash
# 0. MAMS. Download the ATSA xml from github.com/siat-nlp/MAMS-for-ABSA -> data/mams/
python -m src.data_mams          # sanity report. paste these numbers in the paper.

# 1. Corpus (once, ~30 min)
python -m src.build_corpus --stage text --source hf        # week 1: unblock the pipeline
python -m src.build_corpus --stage text --source json --root data/yelp_raw   # real runs
for A in roberta gpt2 t5; do python -m src.build_corpus --stage pack --arch $A; done

# 2. Baselines FIRST. 9 runs, ~15 min each.
for A in roberta gpt2 t5; do for S in 13 42 1337; do
  python -m src.train_absa --arch $A --init base --seed $S; done; done

# 3. DAPT. 50M tokens each. ~35 min (roberta/gpt2), ~90 min (t5) on L4.
for A in roberta gpt2 t5; do python -m src.dapt --arch $A; done

# 4. DAPT fine-tunes. 9 runs.
for A in roberta gpt2 t5; do for S in 13 42 1337; do
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
