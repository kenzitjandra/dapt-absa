"""MAMS-ATSA loader.

Grab the XML from https://github.com/siat-nlp/MAMS-for-ABSA
  data/MAMS-ATSA/raw/{train,val,test}.xml   ->  PATHS['mams']/

One XML <sentence> yields SEVERAL rows -- one per aspectTerm. That explosion is
the whole point of MAMS: the same sentence appears with different aspects and
DIFFERENT gold labels, so a model that ignores the aspect cannot score well.
"""
import os, xml.etree.ElementTree as ET
from config import LABEL2ID, PATHS


def load_split(split):
    path = os.path.join(PATHS["mams"], f"{split}.xml")
    rows = []
    for sent in ET.parse(path).getroot().findall("sentence"):
        text = sent.find("text").text.strip()
        terms = sent.find("aspectTerms")
        if terms is None:
            continue
        for t in terms.findall("aspectTerm"):
            pol = t.get("polarity")
            if pol not in LABEL2ID:      # skip 'conflict' if present
                continue
            rows.append({
                "sentence": text,
                "aspect":   t.get("term"),
                "label":    LABEL2ID[pol],
            })
    return rows


def sanity_report():
    """Run this ONCE and paste the numbers into the paper. If the val/test counts
    don't match the MAMS paper (~4297 val / 4297 test aspect terms), the parse is wrong."""
    from collections import Counter
    for split in ["train", "val", "test"]:
        rows = load_split(split)
        c = Counter(r["label"] for r in rows)
        n_sent = len({r["sentence"] for r in rows})
        print(f"{split:5s}  rows={len(rows):6d}  sentences={n_sent:5d}  "
              f"aspects/sent={len(rows)/n_sent:.2f}  dist={dict(c)}")


if __name__ == "__main__":
    sanity_report()
