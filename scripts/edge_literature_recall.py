"""Edge-literature-check recall and precision: compares the production edge-literature check against an independent, stricter check.

For every bridge edge already checked in a saved run (outputs/pipeline_b_<pair>.json): (1) read the pipeline's own verdict (`supported`);
(2) re-pull a larger pool of the same literature through PubMedClient.search_edge_citations; (3) scan every sentence for one naming both
genes and containing a word from a small hand-written interaction vocabulary; (4) call the edge gold-positive if any such sentence exists.
Recall = (gold-positive and supported) / gold-positive. Precision = (gold-positive and supported) / supported.

Must be run on your own machine (the E-utilities and BioC-PMC endpoints are blocked from the cloud sandbox and the device bridge).

Usage:
    python3 -m scripts.edge_literature_recall --pair t2dm_cad
    python3 -m scripts.edge_literature_recall --pair t2dm_ad
    python3 -m scripts.edge_literature_recall            # default: both pairs

Output: outputs/edge_literature_recall/{pair_id}.json (per-edge detail and the summary) plus a printed summary table.
"""
import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

from PipelineB.pubmed_client import PubMedClient

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
RESULT_DIR = OUTPUTS_DIR / "edge_literature_recall"

# The production check uses retmax=3; this script pulls a larger pool of the same literature to build an independent
# gold standard.
RECALL_POOL_SIZE = 15

# Hand-written interaction vocabulary: a sentence naming both genes and containing one of these word stems counts as
# describing an interaction. Not exhaustive.
INTERACTION_VERBS = [
    "activat", "inhibit", "phosphorylat", "dephosphorylat", "bind", "interact",
    "regulat", "promot", "suppress", "induc", "modulat", "increas", "decreas",
    "mediat", "block", "trigger", "upregulat", "downregulat", "stimulat",
    "cleav", "degrad", "signal", "target", "associat", "repress", "enhanc",
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# Input: a gene symbol. Output: a case-insensitive, word-boundary regex for it.
def _gene_pattern(gene: str) -> "re.Pattern":
    return re.compile(rf"\b{re.escape(gene)}\b", re.IGNORECASE)


# Input: one sentence and two gene symbols. Output: True if both symbols and at least one interaction word stem appear
# in the sentence.
def _sentence_has_interaction_language(sentence: str, gene_a: str, gene_b: str) -> bool:
    if not (_gene_pattern(gene_a).search(sentence) and _gene_pattern(gene_b).search(sentence)):
        return False
    lowered = sentence.lower()
    return any(verb in lowered for verb in INTERACTION_VERBS)


# Re-pull a larger literature pool with the production method. Input: a PubMed client, two gene symbols and the pool
# size. Output: every retrieved passage (full-text paragraph if a PMC copy exists, else the abstract).
def _passages_for_pair(client: PubMedClient, gene_a: str, gene_b: str, retmax: int) -> List[str]:
    edge_citations = client.search_edge_citations(gene_a, gene_b, retmax=retmax)
    return [c.matched_text for c in edge_citations if c.matched_text]


# Independent ground truth for one edge. Input: a PubMed client and two gene symbols. Output: the verdict plus the first
# supporting sentence found.
def gold_positive(client: PubMedClient, gene_a: str, gene_b: str) -> dict:
    passages = _passages_for_pair(client, gene_a, gene_b, RECALL_POOL_SIZE)
    for passage in passages:
        for sentence in _SENTENCE_SPLIT_RE.split(passage):
            if _sentence_has_interaction_language(sentence, gene_a, gene_b):
                return {
                    "gold_positive": True,
                    "evidence_sentence": sentence.strip(),
                    "passages_scanned": len(passages),
                }
    return {"gold_positive": False, "evidence_sentence": None, "passages_scanned": len(passages)}


# Read the edge_literature_checks already produced by a saved run. Input: pair id. Output: the checks; a missing file or
# key gives an empty list.
def load_pipeline_edge_checks(pair_id: str) -> List[dict]:
    path = OUTPUTS_DIR / f"pipeline_b_{pair_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run Pipeline B for {pair_id} first "
            f"(main.py's run_pipeline_b_and_save) before this script."
        )
    data = json.loads(path.read_text())
    checks = []
    for chain_result in data.get("stage3", []):
        for check in chain_result.get("edge_literature_checks", []):
            checks.append(
                {
                    "gene_a": check["gene_a"],
                    "gene_b": check["gene_b"],
                    "pipeline_supported": check["supported"],
                }
            )
    return checks


# Input: pair id and a PubMed client. Output: the per-edge detail and the recall/precision summary.
def run_for_pair(pair_id: str, client: PubMedClient) -> dict:
    checks = load_pipeline_edge_checks(pair_id)
    # Check each bridge edge once: the same edge can appear in several chains of a pair, and its verdict does not depend
    # on the chain.
    deduped = {}
    for c in checks:
        key = tuple(sorted((c["gene_a"], c["gene_b"])))
        deduped[key] = c
    results = []
    for (gene_a, gene_b), c in deduped.items():
        print(f"  re-pulling larger literature pool for {gene_a} <-> {gene_b} ...")
        gold = gold_positive(client, gene_a, gene_b)
        results.append(
            {"gene_a": gene_a, "gene_b": gene_b, "pipeline_supported": c["pipeline_supported"], **gold}
        )

    tp = sum(1 for r in results if r["pipeline_supported"] and r["gold_positive"])
    fn = sum(1 for r in results if not r["pipeline_supported"] and r["gold_positive"])
    fp = sum(1 for r in results if r["pipeline_supported"] and not r["gold_positive"])
    tn = sum(1 for r in results if not r["pipeline_supported"] and not r["gold_positive"])
    gold_positive_count = tp + fn
    pipeline_positive_count = tp + fp

    return {
        "pair_id": pair_id,
        "bridge_edges_checked": len(results),
        "gold_positive_count": gold_positive_count,
        "pipeline_positive_count": pipeline_positive_count,
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "recall": (tp / gold_positive_count) if gold_positive_count else None,
        "precision": (tp / pipeline_positive_count) if pipeline_positive_count else None,
        "edges": results,
    }


# Command-line entry point: run the recall check for one pair or both and save the results.
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", choices=["t2dm_ad", "t2dm_cad", "both"], default="both")
    args = parser.parse_args()
    pair_ids = ["t2dm_ad", "t2dm_cad"] if args.pair == "both" else [args.pair]

    client = PubMedClient()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for pair_id in pair_ids:
        print(f"\n=== {pair_id} ===")
        summary = run_for_pair(pair_id, client)
        out_path = RESULT_DIR / f"{pair_id}.json"
        out_path.write_text(json.dumps(summary, indent=2))
        recall_str = f"{summary['recall']:.2f}" if summary["recall"] is not None else "n/a (no gold-positive edges)"
        precision_str = (
            f"{summary['precision']:.2f}" if summary["precision"] is not None else "n/a (pipeline flagged nothing)"
        )
        print(f"  bridge edges checked: {summary['bridge_edges_checked']}")
        print(f"  gold-positive (independent, broader pool + verb check): {summary['gold_positive_count']}")
        print(f"  pipeline-positive (production retmax=3, no verb check): {summary['pipeline_positive_count']}")
        print(f"  recall:    {recall_str}")
        print(f"  precision: {precision_str}")
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
