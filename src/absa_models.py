"""ABSA task heads.

THE FAIRNESS RULE: all three models must be scored the same way -- pick the
argmax over exactly 3 label options. If RoBERTa gets a clean softmax and GPT-2 is
allowed to free-generate "I think it's kind of positive?", we are measuring
decoding strategy, not architecture, and the paper's central claim collapses.

So: T5 and GPT-2 never generate. We compute the log-likelihood of each of the
three label strings and take the argmax. Nobody can emit an invalid label; nobody
gets format-error penalties the others don't face.
"""
import torch
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForCausalLM, AutoModelForSeq2SeqLM)
from config import LABELS


# --------------------------------------------------------------------- RoBERTa
class RobertaABSA:
    kind = "enc"

    def __init__(self, path, dev, max_len=128):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=len(LABELS)).to(dev)
        self.dev, self.max_len = dev, max_len

    def encode(self, batch):
        # sentence-pair encoding: <s> sentence </s></s> aspect </s>
        enc = self.tok([b["sentence"] for b in batch], [b["aspect"] for b in batch],
                       truncation=True, max_length=self.max_len,
                       padding=True, return_tensors="pt").to(self.dev)
        enc["labels"] = torch.tensor([b["label"] for b in batch]).to(self.dev)
        return enc

    def loss(self, enc):
        return self.model(**enc).loss

    @torch.no_grad()
    def predict(self, enc):
        enc = {k: v for k, v in enc.items() if k != "labels"}
        return self.model(**enc).logits.argmax(-1)


# -------------------------------------------------------------------------- T5
class T5ABSA:
    kind = "encdec"
    TEMPLATE = "absa sentence: {s} aspect: {a}"

    def __init__(self, path, dev, max_len=128):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(dev)
        self.dev, self.max_len = dev, max_len
        self.label_ids = [self.tok(l, return_tensors="pt").input_ids[0] for l in LABELS]

    def encode(self, batch):
        src = [self.TEMPLATE.format(s=b["sentence"], a=b["aspect"]) for b in batch]
        enc = self.tok(src, truncation=True, max_length=self.max_len,
                       padding=True, return_tensors="pt").to(self.dev)
        tgt = self.tok([LABELS[b["label"]] for b in batch],
                       padding=True, return_tensors="pt").input_ids.to(self.dev)
        tgt[tgt == self.tok.pad_token_id] = -100
        enc["labels"] = tgt
        enc["_gold"] = torch.tensor([b["label"] for b in batch]).to(self.dev)
        return enc

    def loss(self, enc):
        return self.model(input_ids=enc["input_ids"],
                          attention_mask=enc["attention_mask"],
                          labels=enc["labels"]).loss

    @torch.no_grad()
    def predict(self, enc):
        """Score all 3 label sequences; argmax. No free generation."""
        B = enc["input_ids"].size(0)
        scores = torch.zeros(B, len(LABELS), device=self.dev)
        for li, lab in enumerate(self.label_ids):
            dec = lab.unsqueeze(0).repeat(B, 1).to(self.dev)
            out = self.model(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"], labels=dec)
            lp = F.log_softmax(out.logits, dim=-1)
            tokw = lp.gather(-1, dec.unsqueeze(-1)).squeeze(-1)   # B x L
            scores[:, li] = tokw.sum(-1)
        return scores.argmax(-1)


# ----------------------------------------------------------------------- GPT-2
class GPT2ABSA:
    kind = "dec"
    PROMPT = "Review: {s}\nAspect: {a}\nSentiment:"

    def __init__(self, path, dev, max_len=128):
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path).to(dev)
        self.model.config.pad_token_id = self.tok.pad_token_id
        self.dev, self.max_len = dev, max_len
        self.label_ids = [self.tok(" " + l)["input_ids"] for l in LABELS]

    def _build(self, batch, label_override=None):
        seqs, masks, labs = [], [], []
        for i, b in enumerate(batch):
            p = self.tok(self.PROMPT.format(s=b["sentence"], a=b["aspect"]),
                         truncation=True, max_length=self.max_len - 4)["input_ids"]
            li = b["label"] if label_override is None else label_override
            lab_ids = self.label_ids[li]
            ids = p + lab_ids
            # loss ONLY on the label tokens -- the prompt is not a prediction target
            tgt = [-100] * len(p) + lab_ids
            seqs.append(ids); labs.append(tgt)
        L = max(len(s) for s in seqs)
        ii = torch.full((len(seqs), L), self.tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(seqs), L), dtype=torch.long)
        lb = torch.full((len(seqs), L), -100, dtype=torch.long)
        for k, (s, t) in enumerate(zip(seqs, labs)):
            ii[k, :len(s)] = torch.tensor(s)
            am[k, :len(s)] = 1
            lb[k, :len(t)] = torch.tensor(t)
        return {"input_ids": ii.to(self.dev), "attention_mask": am.to(self.dev),
                "labels": lb.to(self.dev)}

    def encode(self, batch):
        enc = self._build(batch)
        enc["_batch"] = batch
        enc["_gold"] = torch.tensor([b["label"] for b in batch]).to(self.dev)
        return enc

    def loss(self, enc):
        return self.model(input_ids=enc["input_ids"],
                          attention_mask=enc["attention_mask"],
                          labels=enc["labels"]).loss

    @torch.no_grad()
    def predict(self, enc):
        """Log-prob of each candidate label continuation; argmax."""
        batch = enc["_batch"]
        scores = torch.zeros(len(batch), len(LABELS), device=self.dev)
        for li in range(len(LABELS)):
            e = self._build(batch, label_override=li)
            out = self.model(input_ids=e["input_ids"], attention_mask=e["attention_mask"])
            lp = F.log_softmax(out.logits[:, :-1], dim=-1)
            tgt = e["labels"][:, 1:]
            m = tgt != -100
            safe = tgt.clone(); safe[~m] = 0
            tokw = lp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
            scores[:, li] = (tokw * m).sum(-1)          # sum logprob of label tokens
        return scores.argmax(-1)


def build(arch, path, dev, max_len=128):
    return {"roberta": RobertaABSA, "gpt2": GPT2ABSA,
            "t5": T5ABSA, "t5small": T5ABSA}[arch](path, dev, max_len)
