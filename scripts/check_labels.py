"""Print how many tokens each label becomes, using the exact tokenization in absa_models.py.

Unequal lengths bias the summed-logprob argmax toward shorter labels. Report in the paper; do not change.
Run: PYTHONPATH=src python scripts/check_labels.py
"""
from transformers import AutoTokenizer
from config import ARCHS, LABELS

for arch, fmt in [("gpt2", " {}"), ("t5", "{}"), ("t5small", "{}")]:
    tok = AutoTokenizer.from_pretrained(ARCHS[arch].hf_id)
    for lab in LABELS:
        ids = tok(fmt.format(lab))["input_ids"]   # T5 includes </s>, as the model does
        print(f"{arch:8s} {fmt.format(lab)!r:12s} n_tokens={len(ids)} pieces={tok.convert_ids_to_tokens(ids)}")
