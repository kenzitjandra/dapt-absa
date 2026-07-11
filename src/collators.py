"""Objective-specific collators.

The three objectives are NOT interchangeable and we should stop pretending they
are. MLM supervises ~15% of tokens per pass; CLM supervises ~100%. That asymmetry
is a genuine property of the architectures, not a bug we can normalise away. We
control the one knob we CAN control -- tokens *processed* -- and we say the rest
out loud in Limitations.
"""
import numpy as np
import torch


class MLMCollator:
    """RoBERTa: standard BERT-style 15% masking (80/10/10)."""
    def __init__(self, tok, p=0.15):
        self.tok, self.p = tok, p

    def __call__(self, blocks):
        x = torch.tensor(np.array(blocks), dtype=torch.long)
        labels = x.clone()
        special = torch.zeros_like(x, dtype=torch.bool)
        for sid in self.tok.all_special_ids:
            special |= (x == sid)

        prob = torch.full(x.shape, self.p)
        prob.masked_fill_(special, 0.0)
        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100

        r = torch.rand(x.shape)
        x[masked & (r < 0.8)] = self.tok.mask_token_id
        rand_tok = torch.randint(len(self.tok), x.shape, dtype=torch.long)
        swap = masked & (r >= 0.8) & (r < 0.9)
        x[swap] = rand_tok[swap]
        return {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": labels}


class CLMCollator:
    """GPT-2: next-token prediction. Labels == inputs; HF shifts internally."""
    def __call__(self, blocks):
        x = torch.tensor(np.array(blocks), dtype=torch.long)
        return {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x.clone()}


class SpanCorruptionCollator:
    """T5: replace contiguous spans with sentinels, reconstruct them in the decoder.

    Reimplemented from Raffel et al. because HF only ships this in the Flax
    examples. noise_density=0.15, mean_span_length=3 -- the paper's defaults.
    """
    def __init__(self, tok, noise_density=0.15, mean_span=3.0, max_target=128):
        self.tok = tok
        self.nd = noise_density
        self.mean_span = mean_span
        self.max_target = max_target
        self.sentinels = tok.convert_tokens_to_ids(
            [f"<extra_id_{i}>" for i in range(100)])
        self.pad = tok.pad_token_id
        self.eos = tok.eos_token_id

    def _noise_mask(self, n, rng):
        n_noise = max(1, int(round(n * self.nd)))
        n_spans = max(1, int(round(n_noise / self.mean_span)))
        n_noise = min(n_noise, n - 1)

        def segment(total, k):
            # randomly split `total` items into k positive-sized runs
            cuts = np.sort(rng.choice(np.arange(1, total), size=k - 1, replace=False)) \
                   if k > 1 else np.array([], dtype=int)
            return np.diff(np.concatenate([[0], cuts, [total]]))

        noise_runs = segment(n_noise, n_spans)
        keep_runs = segment(n - n_noise, n_spans)

        mask = np.zeros(n, dtype=bool)
        i = 0
        for k, m in zip(keep_runs, noise_runs):
            i += k
            mask[i:i + m] = True
            i += m
        return mask

    def _one(self, block, rng):
        n = len(block)
        mask = self._noise_mask(n, rng)
        inp, tgt, s = [], [], 0
        i = 0
        while i < n:
            if mask[i]:
                if s < len(self.sentinels):
                    inp.append(self.sentinels[s])
                    tgt.append(self.sentinels[s])
                    s += 1
                while i < n and mask[i]:
                    tgt.append(int(block[i]))
                    i += 1
            else:
                inp.append(int(block[i]))
                i += 1
        inp.append(self.eos)
        tgt.append(self.eos)
        return inp, tgt[:self.max_target]

    def __call__(self, blocks):
        rng = np.random.default_rng()
        pairs = [self._one(b, rng) for b in blocks]
        li = max(len(a) for a, _ in pairs)
        lt = max(len(b) for _, b in pairs)

        ii = torch.full((len(pairs), li), self.pad, dtype=torch.long)
        am = torch.zeros((len(pairs), li), dtype=torch.long)
        lb = torch.full((len(pairs), lt), -100, dtype=torch.long)
        for k, (a, b) in enumerate(pairs):
            ii[k, :len(a)] = torch.tensor(a)
            am[k, :len(a)] = 1
            lb[k, :len(b)] = torch.tensor(b)
        return {"input_ids": ii, "attention_mask": am, "labels": lb}


def get_collator(objective, tok):
    return {"mlm": lambda: MLMCollator(tok),
            "clm": lambda: CLMCollator(),
            "span": lambda: SpanCorruptionCollator(tok)}[objective]()
