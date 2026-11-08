# Week 3: what we ran, what changed in the code, and what we found

Written 2026-10-15. All numbers below were copied from the printed logs and the output of
`src.aggregate` and `scripts/extras.py` on Colab. The run JSONs themselves live on the shared Drive
(`runs/`, `runs_extra/`); this document does not replace them. For the background on the pipeline
(data, DAPT objectives, fine-tuning, the statistics), read `docs/PIPELINE.md` first.

**Verified on 2026-10-15** by recomputing everything from copies of all 80 run JSONs and the six
`dapt_report.json` files: file sets, provenance, metrics recomputed from the stored predictions, the
n = 3 and n = 5 aggregate tables, the `extras.py` output and the Holm values all match this document.

**New to the extra steps?** The terms and methods we added beyond the original plan (replicates,
post-hoc, bootstrap intervals, Holm correction, the learning-rate check, the new flags) are explained in
plain language in Appendix A at the end of this file.

## 1. Summary

- All four DAPT runs (RoBERTa, GPT-2, T5-base, T5-small) completed and passed the DAPT gate: held-out
  language-modelling loss dropped for every model, on exactly 49,995,776 tokens.
- With the planned 3 seeds, GPT-2 looked like the one model that benefits from DAPT (Delta macro-F1 of
  +0.83, p = 0.067). We then added 2 seeds, trained 2 more GPT-2 DAPT checkpoints, and re-ran the
  fine-tuning at a lower learning rate to test that result.
- **The GPT-2 gain did not replicate.** Two fresh GPT-2 DAPT checkpoints gave +0.07 and +0.30, against
  +0.84 for the original. The original checkpoint's interval touches zero, and after Holm correction its
  p-value is 0.100.
- At a learning rate halved (x0.5), the original GPT-2 checkpoint shows +1.28 with an interval that
  excludes zero. This has not yet been tested on the replicate checkpoints, so it may come from the same
  lucky checkpoint.
- RoBERTa, T5-base and T5-small show no clear effect, and for T5 and T5-small the interval rules out an
  effect larger than about one F1 point either way.
- Honest headline: **no robust DAPT benefit at the base fine-tuning recipe, and DAPT run-to-run variation
  (standard deviation about 0.4 F1 for GPT-2) is as large as any effect we measured.** The README names
  this outcome as an acceptable, publishable one.

## 2. The plan and what actually happened

The plan (`WEEK3_PLAN_3DAYS.md`) was: run all four DAPT pipelines on Day 1, fine-tune each three times,
aggregate, and write the paper. Extra seeds and error analysis were explicitly out of scope.

| Step | Plan | What happened |
|---|---|---|
| Commit question | Run `git diff --stat freeze-ft-v1 HEAD`, accept the mixed-commit warning if no training file changed | Only `.gitignore` and `requirements.txt` differed, so the warning was accepted. Baselines are on `0ffc6e8`, DAPT runs on `c6173b6`. |
| DAPT, 4 models | roberta, t5, gpt2, t5small | Done. All passed Gate 2 (section 3). |
| Fine-tunes, 4 x 3 seeds | 12 DAPT runs | Done. 24 runs in `runs/`: 12 baselines and 12 DAPT, all on transformers 4.46.3 and an L4 (checked with a provenance count). |
| Aggregate | n = 3 table | Done (section 4). |
| Extra seeds | out of scope | **Added after seeing the n = 3 result.** Seeds 7 and 2024 for all four architectures, base and DAPT (16 runs, on `main` at `c6173b6`). |
| DAPT replicates | not planned | **Added.** Two more GPT-2 DAPT runs (seeds 1 and 2), each fine-tuned at 5 seeds (10 runs). |
| LR sensitivity | not planned | **Added.** Learning rate x0.5 for gpt2, t5 and roberta, base and DAPT, 5 seeds (30 runs). |
| Extra analysis | not planned | **Added.** `scripts/extras.py`: sentence-level bootstrap intervals, Holm correction, per-seed Deltas. |

The extra experiments were decided after seeing the first results, so they are post-hoc. The paper must
say so. Every run that was started is reported; none was dropped.

Incidents that cost time but did not change results:

- **`t5small` pack files vanished from Drive** twice after being written. They were re-packed, and the new
  `t5small_train.npy` was checked identical to `t5_train.npy` (`np.array_equal` returned `True`), which
  also confirms the corpus file had not changed. The cause was not established (the Drive account was
  near 90% full).
- **`runs_extra/` was not linked to Drive** at first, so early extra results lived only on the Colab
  disk. A runtime deletion lost 10 runs, which were re-run. The skip-if-exists loops make re-running safe.
  The link must be created in every session (section 6).
- **The laptop slept** during Part C, which disconnected Colab. Nothing was lost beyond the run in
  progress; the loops resumed from the missing runs.
- **Same-seed reruns reproduced exactly.** For GPT-2 base at lr x0.5, seeds 13, 42 and 1337 and DAPT seed
  13 gave identical test F1 across the interrupted attempt and the rerun. So fine-tuning is repeatable on
  this setup, at least for these runs.

## 3. DAPT results (Gate 2)

Held-out Yelp language-modelling loss, same blocks and same mask seed before and after. All reports show
`step` 6103 and `tokens` 49,995,776 (6,103 steps x 32 x 256).

| Checkpoint | loss before | loss after | drop |
|---|---|---|---|
| roberta | 1.897 | 1.487 | +0.410 |
| gpt2 (original) | 3.766 | 3.314 | +0.452 |
| gpt2 replicate d1 (seed 1) | 3.766 | 3.314 | +0.452 |
| gpt2 replicate d2 (seed 2) | 3.766 | 3.314 | +0.451 |
| t5 | 2.273 | 1.870 | +0.403 |
| t5small | 2.796 | 2.422 | +0.374 |

The GPT-2 replicates have the same `loss_before` as the original, as designed, and end within 0.0005 of
each other on language-modelling loss. That does not carry over to ABSA scores (section 5).

## 4. Fine-tuning results

### 4.1 With 3 seeds (the original plan, seeds 13, 42, 1337)

| Architecture | base F1 | DAPT F1 | Delta F1 | paired p |
|---|---|---|---|---|
| roberta | 84.61 +/- 0.72 | 84.18 +/- 0.46 | -0.43 +/- 1.04 | 0.549 |
| gpt2 | 81.54 +/- 0.09 | 82.37 +/- 0.31 | +0.83 +/- 0.39 | 0.067 |
| t5 | 83.57 +/- 0.77 | 83.97 +/- 0.47 | +0.39 +/- 0.36 | 0.200 |
| t5small | 82.19 +/- 0.52 | 82.28 +/- 0.68 | +0.09 +/- 0.23 | 0.562 |

Only one cross-architecture comparison was flagged (gpt2 vs t5small, p = 0.048).

### 4.2 With 5 seeds (13, 42, 1337, 7, 2024; post-hoc extension)

| Architecture | base F1 | DAPT F1 | Delta F1 | paired p |
|---|---|---|---|---|
| roberta | 84.59 +/- 0.56 | 84.03 +/- 0.53 | -0.56 +/- 0.96 | 0.257 |
| gpt2 | 81.66 +/- 0.37 | 82.51 +/- 0.32 | +0.84 +/- 0.54 | 0.025 |
| t5 | 83.96 +/- 0.76 | 83.79 +/- 0.41 | -0.17 +/- 0.82 | 0.664 |
| t5small | 82.35 +/- 0.43 | 82.66 +/- 0.72 | +0.31 +/- 0.35 | 0.123 |

Cross-architecture tests flagged roberta vs gpt2 (p = 0.021) and gpt2 vs t5 (p = 0.049) as different. With
6 uncorrected comparisons and one DAPT checkpoint per architecture, these are exploratory. T5-base's
n = 3 gain (+0.39) disappeared at n = 5, which shows how fragile n = 3 was; GPT-2's estimate stayed
(+0.83 to +0.84).

Test accuracy at n = 5 (for the paper's table): roberta 85.07 / 84.51, gpt2 82.20 / 83.05,
t5 84.40 / 84.24, t5small 82.77 / 83.08 (base / DAPT).

## 5. Robustness checks (`scripts/extras.py`)

Each line is the Delta of DAPT against the base runs at the same learning rate, paired by seed, with a
sentence-level bootstrap 95% interval (2,000 resamples of the 500 test sentences, pooling the five seeds).
The bootstrap covers test-set sampling noise. The t-test covers seed noise.

### 5.1 GPT-2 DAPT replicates (lr x1, five seeds each)

| DAPT checkpoint | Delta F1 | bootstrap 95% CI | paired p | per-seed (7, 13, 42, 1337, 2024) |
|---|---|---|---|---|
| original | +0.84 +/- 0.54 | [-0.09, +1.78] | 0.025 | +0.21, +1.28, +0.65, +0.56, +1.52 |
| replicate d1 | +0.07 +/- 0.65 | [-0.87, +0.97] | 0.817 | -0.46, +0.91, -0.34, -0.40, +0.65 |
| replicate d2 | +0.30 +/- 0.78 | [-0.59, +1.17] | 0.437 | -0.78, +1.28, +0.02, +0.77, +0.21 |

Mean of the three checkpoints is about +0.40 with a standard deviation of about 0.4. Before the runs we
proposed this rule: "supported" if both replicates are positive and at least +0.4, "not supported" if
either is at or below 0. Both replicates are positive but below +0.4, so by that rule the result is
inconclusive, and in practice it does not support the original size of the gain. The three checkpoints'
intervals overlap, so this is not a proof that DAPT does nothing for GPT-2; it shows that +0.84 is not a
stable property of GPT-2 DAPT.

### 5.2 Main Deltas with Holm correction (original DAPT checkpoint, lr x1)

| Architecture | Delta F1 | bootstrap 95% CI | raw p | Holm p |
|---|---|---|---|---|
| gpt2 | +0.84 | [-0.09, +1.78] | 0.025 | 0.100 |
| roberta | -0.56 | [-1.36, +0.17] | 0.257 | 0.515 |
| t5 | -0.17 | [-0.78, +0.42] | 0.664 | 0.664 |
| t5small | +0.31 | [-0.29, +0.98] | 0.123 | 0.368 |

For t5 and t5small the interval stays inside +/-1 F1 point, so a larger effect either way is ruled out
for the checkpoint we trained. For roberta the lower end reaches -1.36, so a small harm cannot be ruled
out. The bootstrap does not include DAPT run-to-run variation, and only GPT-2 has been replicated.

### 5.3 Learning rate halved (x0.5, five seeds, original DAPT checkpoint)

| Architecture | Delta F1 (x1) | Delta F1 (x0.5) | x0.5 bootstrap 95% CI | x0.5 paired p |
|---|---|---|---|---|
| gpt2 | +0.84 | **+1.28 +/- 0.70** | [+0.26, +2.28] | 0.015 |
| roberta | -0.56 | -0.05 +/- 0.74 | [-0.86, +0.80] | 0.882 |
| t5 | -0.17 | +0.12 +/- 0.32 | [-0.53, +0.73] | 0.443 |

- A lower rate lowers the absolute scores of both base and DAPT models, and lowers the base score more
  (GPT-2 base F1 81.66 to 80.76, DAPT 82.51 to 82.03; RoBERTa base 84.59 to 83.94, DAPT 84.03 to 83.89).
  One reading is that DAPT helps more when fine-tuning is under-trained. This is a hypothesis, and it
  holds only for GPT-2.
- The GPT-2 x0.5 result uses only the original checkpoint, the one that looks lucky at x1. It is also an
  exploratory test: Holm over the three x0.5 tests gives 0.044 for GPT-2, which would not survive a
  correction over all tests we ran. It must be reported as exploratory.

### 5.4 What the data supports

- Supported: for the recipe in `config.py`, none of the four architectures shows a DAPT gain that
  survives correction, and DAPT checkpoint variation for GPT-2 is about 0.4 F1.
- Supported: for T5-base and T5-small, an effect larger than about 1 F1 point is unlikely (for the
  checkpoint trained).
- Not supported: that topology modulates the DAPT benefit. The only cross-architecture differences rest
  on GPT-2's original checkpoint.
- Not supported: that GPT-2 benefits from DAPT (the replicates do not show it at the standard rate).
- Open: whether the GPT-2 gain at lr x0.5 holds on other DAPT checkpoints (section 7).

## 6. How the code changed from the previous plan

The plan said not to change `train_absa.py`, `absa_models.py`, `dapt.py` or `config.py` once real runs
start. The core runs (baselines, the four DAPT runs, the 24 main fine-tunes, and Part A) used the
original code. The later experiments needed three optional flags, added on the branch
`b/extra-robustness` (commit `01f54eb`). **With none of the new flags, behaviour is unchanged.** The only
visible differences are extra keys in the output JSON and in `dapt_report.json`, which `aggregate.py`
and the DAPT guard ignore.

| File | Change |
|---|---|
| `src/dapt.py` | `--seed N`. Uses a new data permutation (`perm_seed`), calls `torch.manual_seed(N)` after the "before" loss measurement (because `heldout_loss` reseeds CUDA to 0), writes to `checkpoints/<arch>_dapt_d<N>`, and records `"seed"` in the report. No seed keeps the original behaviour. |
| `src/train_absa.py` | `--dapt-seed N` loads `<arch>_dapt_d<N>`. `--lr-mult M` scales the configured learning rate. Runs using either flag write to `runs_extra/` with a distinct tag (`..._d<N>_s<seed>`, `..._lrx<M>`), so the main table never mixes with variants. The JSON gains `dapt_seed` and `lr`. |
| `scripts/extras.py` (new) | Reads `runs/` and `runs_extra/`. Per-cell Delta against the base at the same lr multiplier, paired by seed, with a sentence-level bootstrap interval, per-seed Deltas and Holm correction for the main Deltas. |
| `src/config.py`, `src/aggregate.py`, `src/absa_models.py`, `src/collators.py`, `src/build_corpus.py` | Unchanged. |

New unit of reproducibility: three commits are involved. `0ffc6e8` (the 12 original baselines),
`c6173b6` (12 DAPT fine-tunes and the 16 extra-seed runs, all on `main`) and `01f54eb` (DAPT replicates
and the lr x0.5 runs). The training code is the same for all three on default settings.

## 7. Open items and risks

1. **GPT-2 replicates at lr x0.5 (10 runs, about 1 hour 50 minutes).** This decides whether the x0.5 gain
   belongs to DAPT or to one checkpoint. Command per run:
   `python -u -m src.train_absa --arch gpt2 --init dapt --dapt-seed D --lr-mult 0.5 --seed S`, for D in
   1, 2 and S in 13, 42, 1337, 7, 2024. `extras.py` picks them up without changes.
2. **Only GPT-2 has been replicated.** RoBERTa, T5-base and T5-small each have one DAPT checkpoint, so their
   near-zero Deltas have the same blind spot.
3. **Post-hoc analysis.** The extra seeds and replicates were added after seeing the n = 3 result. The
   paper should present n = 3 as the planned analysis and n = 5 and the robustness checks as extensions.
4. **Multiple comparisons.** Only the four main Deltas have a Holm correction. The cross-architecture tests
   (6 comparisons) and the x0.5 tests are exploratory.
5. **Environment evidence.** Each run JSON records the `transformers` version (4.46.3, verified for the 24
   main runs), but not the numpy version. A traceback in one session showed Python 3.13, while the README
   asks for runtime 2025.07 (Python 3.11). The per-session Cell 2 printouts are the only evidence of the
   numpy version; keep them.
6. **`corpus_source` is never written by `dapt.py`,** so every DAPT run JSON says `"unknown"` for it.
7. **`runs_extra` must be linked to Drive in every new session** (`ln -s` to
   `dapt-absa-shared/runs_extra`), or results stay on the Colab disk and vanish on disconnect.
8. **In-code DAPT guard passes a `nan` loss.** The guard in `train_absa.py` only checks `loss_after >=
   loss_before`, which is false for `nan`. Gate 2 (the manual check) catches it. None of the reports
   contained `nan`.

## 8. What goes in the paper

- A main table at n = 5 (and n = 3 as the planned result), with Delta, standard deviation and paired
  p, labelled as post-hoc for the extra seeds.
- The three GPT-2 checkpoint Deltas and the DAPT-variation finding, as the main robustness result.
- The lr x0.5 table as exploratory, with the single-checkpoint caveat.
- Bootstrap intervals for the main Deltas, and the Holm-adjusted p-values.
- Limitations: one DAPT checkpoint per architecture except GPT-2, post-hoc extensions, multiple
  comparisons, test-set size (1,336 rows from 500 sentences), MLM and CLM supervision density, T5-base
  size (220M against about 125M), one alternative learning rate, and the environment note above.
- The conclusion, stated plainly: at this scale and recipe, DAPT on 50M Yelp restaurant tokens did not
  give a robust gain on MAMS-ATSA for any of the four models, and the variation between DAPT runs is
  as large as the effects measured.

## Appendix A. Glossary for the steps that were not in the plan

Read this if you were not in the room when we added the extra seeds, the DAPT replicates, the
learning-rate check and the extra analysis. Each entry says what the term means, how we used it, and how to
read it.

### A.1 Ideas about the experiment

**Post-hoc.** A decision made *after* seeing results. Our extra seeds, the GPT-2 DAPT replicates and the
learning-rate check were all decided after the planned 3-seed table showed GPT-2 at p = 0.067. Post-hoc
steps are legitimate, but they must be labelled as such. A result found by looking harder only where the
first result looked interesting is more likely to be luck, so the paper presents the 3-seed table as the
planned analysis and everything else as an extension.

**Exploratory vs confirmatory.** A *confirmatory* test is planned in advance to answer one question. An
*exploratory* test is a look around: interesting if it shows something, but not proof. Everything in this
document after the 3-seed table is exploratory, including the learning-rate result.

**Seed.** A number that fixes the random choices of a run, so the same seed gives the same result. We use
two kinds:
- A **fine-tune seed** (13, 42, 1337 in the plan, then 7 and 2024) fixes the order of the training rows and
  the random start of any new layer during fine-tuning.
- A **DAPT seed** (1 and 2, used only for the GPT-2 replicates) fixes the order of the Yelp blocks, which
  tokens get hidden, and the dropout during DAPT. Without a DAPT seed, the code runs the original DAPT.

**Checkpoint.** A saved copy of a model's weights. The folder `checkpoints/<arch>_dapt` holds the DAPT
checkpoint of one architecture. "The original checkpoint" is the one trained on Day 1. `gpt2_dapt_d1` and
`gpt2_dapt_d2` are two more GPT-2 DAPT checkpoints, trained with DAPT seeds 1 and 2.

**Replicate / replication.** Repeating a procedure to see whether the same result comes out again. A *DAPT
replicate* is a whole new DAPT run (a new checkpoint), not just a new fine-tune. It differs from adding
fine-tune seeds:

| | Repeats | Same DAPT checkpoint? | Tests |
|---|---|---|---|
| More fine-tune seeds | only the fine-tuning | yes | luck in fine-tuning |
| DAPT replicates | the DAPT run, then the fine-tuning | no | luck in the DAPT run |

We replicated only GPT-2. The result: the original checkpoint gave +0.84 F1 and the two replicates gave
+0.07 and +0.30, so the first gain did not come out again at its original size. A result that does not
repeat is said not to *replicate*.

**Delta (Delta_DAPT).** The effect of DAPT on one architecture: the test macro-F1 of the DAPT model minus
that of the base model, for the same fine-tune seed, in **points** (percentage points of F1). +0.84 means
DAPT scored 0.84 points higher. We report the mean over seeds and the spread.

**n = 3 vs n = 5.** The number of seeds averaged. n = 3 (seeds 13, 42, 1337) was the plan. n = 5 adds 7 and
2024. More seeds reduce the noise from fine-tuning, but they do not reduce test-set noise or DAPT-run noise.

**Regression to the mean.** When you pick out a result because it looked unusually good, later
measurements tend to come out closer to the truth. The first GPT-2 checkpoint looked good, and the
replicates moved back toward zero. This is the likely explanation for the lost gain, and it is also why
T5-base's n = 3 gain (+0.39) disappeared at n = 5.

**Learning rate (LR) and the LR multiplier.** The learning rate is how big a step the optimizer takes when
updating the weights. Each architecture has its own rate in `config.py` (RoBERTa 2e-5, GPT-2 5e-5, T5-base
1e-4, T5-small 3e-4). The **LR multiplier** (`--lr-mult 0.5`) scales that rate. We re-ran fine-tuning at half
the rate for both the base and the DAPT models. This is a **sensitivity check**: does the Delta still point
the same way if we change the recipe a little?

**Why a lower rate?** Two ideas, neither proven here:
- *Forgetting:* a large rate may overwrite some of what DAPT taught the model during fine-tuning. A smaller
  rate should keep more of it.
- *Under-training:* with a small rate and only 5 epochs the model may not finish learning (validation F1
  was still rising at epoch 5). DAPT gives a better starting point, so it may matter more there.

We saw lower scores for both models at the lower rate, with the base model dropping more. The Delta moved
up for all three architectures tested (gpt2, t5, roberta), but only GPT-2's interval excludes zero.

### A.2 Statistics, in plain language

**Standard deviation (the "±" values).** How spread out the results are across seeds. "+0.84 ± 0.54" means
the mean Delta was 0.84 and individual seeds typically differ from it by about 0.54. We use the *sample*
standard deviation (n − 1), which is the right one for few runs.

**p-value and the paired t-test.** The paired t-test asks: if DAPT truly had no effect, how likely would
differences this large be? The *p-value* is that probability. Small (below 0.05) means "unlikely to be
chance". It does **not** mean the effect is large or important, and a p-value above 0.05 does not prove there
is no effect. *Paired* means each DAPT run is compared with the base run of the **same seed**. With only 3
or 5 seeds the test has little power, so real small effects often show p above 0.05.

**Confidence interval (CI).** A range that likely contains the true effect. "[−0.09, +1.78]" means the data
are consistent with an effect anywhere from slightly negative to quite positive. If the interval includes
0, we cannot rule out "no effect". The narrower it is, the more certain the estimate.

**Bootstrap, and the sentence-level bootstrap.** A way to get a CI without formulas: repeat the evaluation
many times on resampled test data and see how the answer varies. Ours (`scripts/extras.py`):
1. Draw 500 test sentences at random *with replacement* (the test set has 500 sentences and 1,336
   sentence-aspect rows). Keep all the rows of each drawn sentence.
2. For each of the five seeds, recompute the macro-F1 of the base and DAPT predictions on those rows, take
   the Delta, and average the five Deltas.
3. Repeat 2,000 times. The 2.5th and 97.5th percentiles of those averages form the 95% interval.

We resample *sentences*, not rows, because rows of the same sentence are related (the same words, different
aspects), and resampling rows would make the interval too narrow. The bootstrap tells you how much the
answer changes if the test set were a different sample of sentences (**test-set noise**, about ±1 point on
accuracy for 1,336 rows). It does **not** include DAPT-run noise.

**Three kinds of noise, and which tool covers which**

| Source of noise | Example | What measures it |
|---|---|---|
| Fine-tuning luck | different seeds give different scores | the ± and the paired t-test |
| Test-set sampling | a different set of 500 sentences | the bootstrap interval |
| DAPT-run luck | a different DAPT run gives a different model | DAPT replicates (GPT-2 only) |

**Multiple comparisons and Holm correction.** If you run many tests, some will look significant by chance:
at the 5% level, about 1 in 20. We tested four architectures, so a lone p = 0.025 is less impressive. The
*Holm correction* adjusts for this. Sort the p-values from smallest; multiply the smallest by the number of
tests, the next by that number minus one, and so on (never letting an adjusted value fall below an earlier
one, and capping at 1). For the four main Deltas, raw p = 0.025, 0.123, 0.257, 0.664 become Holm p = 0.100,
0.368, 0.515, 0.664. After correction none is below 0.05. The cross-architecture tests and the x0.5 tests are
not corrected in the aggregate output, so treat them as exploratory.

**"Bounded" effects.** If a 95% interval lies inside, say, ±1 point, an effect larger than one point (in
either direction) is unlikely. This is how we state that T5-base and T5-small show no large DAPT effect
without claiming there is exactly none. It only applies to the checkpoint we trained.

### A.3 Code and files added for the extra steps

**Branch `b/extra-robustness` (commit `01f54eb`).** Adds three optional flags. With none of them, the code
behaves exactly as before.

| Flag | Script | Meaning |
|---|---|---|
| `--seed N` | `src/dapt.py` | Make a DAPT replicate: new data order, masks and dropout. Saved to `checkpoints/<arch>_dapt_d<N>`. |
| `--dapt-seed N` | `src/train_absa.py` | Fine-tune from that replicate instead of the original DAPT checkpoint. |
| `--lr-mult M` | `src/train_absa.py` | Multiply the configured fine-tuning learning rate by M. |

**`runs_extra/` and file names.** Runs that use a new flag are saved in `runs_extra/`, never in `runs/`, so
the main table is unaffected. The name tells you what a file is:

| File | Meaning |
|---|---|
| `runs/gpt2_dapt_s13.json` | Original DAPT checkpoint, fine-tune seed 13, normal learning rate. |
| `runs_extra/gpt2_dapt_d1_s13.json` | DAPT replicate 1, fine-tune seed 13. |
| `runs_extra/gpt2_base_s13_lrx0.5.json` | Base model, seed 13, learning rate x0.5. |
| `runs_extra/gpt2_dapt_s13_lrx0.5.json` | Original DAPT checkpoint, seed 13, learning rate x0.5. |

**Parts A, B and C** are the names used in `B_week3_extra.ipynb`: **A** = 2 extra fine-tune seeds (7 and
2024) for all four architectures, on `main`; **B** = the two GPT-2 DAPT replicates, each fine-tuned at 5
seeds; **C** = the learning-rate x0.5 runs for gpt2, t5 and roberta, 5 seeds each.

**Skip-if-exists loops.** Each run loop checks whether the result file already exists and skips it. This
makes a rerun safe after a disconnect: it only does the missing runs.

**`scripts/extras.py`.** Reads `runs/` and `runs_extra/` and prints one line per setting. Example, annotated:

```
gpt2  dapt=d1  lr x1  n=5  Delta F1 +0.07 +/- 0.65  boot95 [-0.87, +0.97]  paired p=0.817
      per-seed: -0.46, +0.91, -0.34, -0.40, +0.65   (seeds [7, 13, 42, 1337, 2024])
```

- `gpt2`: the architecture. `dapt=d1`: the DAPT checkpoint (`orig`, `d1` or `d2`). `lr x1`: the learning
  rate multiplier.
- `n=5`: five seeds were paired.
- `Delta F1 +0.07 +/- 0.65`: the mean Delta and its standard deviation over the five seeds.
- `boot95 [-0.87, +0.97]`: the sentence-level bootstrap 95% interval for the Delta.
- `paired p=0.817`: the paired t-test p-value across seeds.
- `per-seed`: the Delta for each seed, in the order shown. It lets you see whether the sign is consistent.

How to run it, on Colab with the branch checked out: `PYTHONPATH=src python scripts/extras.py`. It compares
each DAPT line with the base runs at the *same* learning-rate multiplier, so a x0.5 DAPT run is compared
with a x0.5 base run.
