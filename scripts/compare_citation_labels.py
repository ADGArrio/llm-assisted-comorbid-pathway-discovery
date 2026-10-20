"""Compare a second rater's hand labels with the citation-evidence classifier output.

Usage: python scripts/compare_citation_labels.py MODEL_JSON LABELS_TSV [--out report.json]

LABELS_TSV columns: pair, idx (1-based position in the classifier's `papers` list), label,
gene_focus, disease_focus, conf. Computes exact agreement and Cohen's kappa on the evidence label,
agreement on the two focus flags, and agreement on the two derived rules used in the thesis
(supports_pair with gene focus; is_gene_focused_mechanism). Lists every disagreement.
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify_citations import supports_pair, is_gene_focused_mechanism  # noqa: E402


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def load_labels(path):
    rows = [l.split("\t") for l in Path(path).read_text().splitlines()[1:] if l.strip()]
    return {int(r[1]): dict(evidence_type=r[2], gene_is_focus=r[3] == "1",
                            disease_is_focus=r[4] == "1", conf=r[5]) for r in rows}


def compare(model_json, labels_tsv):
    papers = json.loads(Path(model_json).read_text())["papers"]
    mine = load_labels(labels_tsv)
    assert len(papers) == len(mine), (len(papers), len(mine))
    ml, hl, rows = [], [], []
    for i, p in enumerate(papers, 1):
        h = mine[i]
        ml.append(p["evidence_type"]); hl.append(h["evidence_type"])
        rows.append((i, p, h))
    n = len(rows)
    out = {"n": n,
           "label_agreement": sum(a == b for a, b in zip(ml, hl)) / n,
           "label_kappa": kappa(ml, hl),
           "gene_focus_agreement": sum(p["gene_is_focus"] is h["gene_is_focus"] for _, p, h in rows) / n,
           "disease_focus_agreement": sum(p["disease_is_focus"] is h["disease_is_focus"] for _, p, h in rows) / n,
           "supports_pair_gene_focus_agreement": sum(
               supports_pair(p, True) == supports_pair(h, True) for _, p, h in rows) / n,
           "gene_focused_mechanism_agreement": sum(
               is_gene_focused_mechanism(p) == is_gene_focused_mechanism(h) for _, p, h in rows) / n,
           "model_gene_focused_mechanism": sum(is_gene_focused_mechanism(p) for _, p, h in rows),
           "mine_gene_focused_mechanism": sum(is_gene_focused_mechanism(h) for _, p, h in rows),
           "model_supports_pair_gene_focus": sum(supports_pair(p, True) for _, p, h in rows),
           "mine_supports_pair_gene_focus": sum(supports_pair(h, True) for _, p, h in rows),
           "confusion": {f"{a} -> {b}": c for (a, b), c in sorted(Counter(zip(hl, ml)).items()) if a != b},
           "disagreements": [dict(idx=i, gene=p["gene_symbol"], pmid=p["pmid"], title=p["title"][:90],
                                  mine=h["evidence_type"], model=p["evidence_type"], conf=h["conf"],
                                  mine_gene=h["gene_is_focus"], model_gene=p["gene_is_focus"])
                             for i, p, h in rows if p["evidence_type"] != h["evidence_type"]]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_json"); ap.add_argument("labels_tsv"); ap.add_argument("--out")
    a = ap.parse_args()
    r = compare(a.model_json, a.labels_tsv)
    print(json.dumps({k: v for k, v in r.items() if k != "disagreements"}, indent=1))
    for d in r["disagreements"]:
        print(d)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))
