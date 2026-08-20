"""Single source of truth. Nobody edits hyperparameters anywhere else.

The comparison in this paper is between DELTAS, not absolute scores. That only
works if the fine-tuning recipe is byte-for-byte identical between the `base`
and `dapt` runs of a given architecture. Everything that could differ lives here.
"""
from dataclasses import dataclass, field

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# ---------------------------------------------------------------- DAPT budget
DAPT_TOKEN_BUDGET = 50_000_000   # matched across all architectures (50_000_000)
DAPT_BLOCK_LEN    = 256          # reviews are short; 512 is wasted compute
DAPT_BATCH_TOKENS = 8192            # per optimizer step (bs * block_len), keep constant


@dataclass
class ArchConfig:
    name: str
    hf_id: str
    dapt_objective: str          # mlm | clm | span
    # Fine-tuning. lr differs by arch BY DESIGN -- see README "on identical recipes".
    lr: float
    epochs: int = 5
    train_bs: int = 32
    eval_bs: int = 64
    max_len: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    # DAPT
    dapt_lr: float = 5e-5
    dapt_bs: int = 32            # * DAPT_BLOCK_LEN = 8192 tokens/step


ARCHS = {
    "roberta": ArchConfig("roberta", "roberta-base", "mlm", lr=2e-5, epochs=5, train_bs=32),
    "gpt2": ArchConfig("gpt2", "gpt2", "clm", lr=5e-5, epochs=5, train_bs=32),
    "t5":      ArchConfig("t5",      "t5-base",       "span", lr=1e-4),
    # scale control -- guards against "your T5 result is just parameter count"
    "t5small": ArchConfig("t5small", "t5-small",      "span", lr=3e-4),
}

SEEDS = [13, 42, 1337]

PATHS = dict(
    mams="data/mams",                # MAMS-ATSA xml files
    corpus="data/yelp_clean.txt",    # one review per line, cleaned
    packed="data/packed",            # tokenized+packed blocks, per arch
    ckpt="checkpoints",              # DAPT checkpoints (mirror to Drive!)
    runs="runs",                     # per-run metrics json
)
