"""
Edge-literature-check RECALL metric -- Task 3 from the user's 2026-10-04 request ("create a
python file that pulls the same literature the edge verification pipeline pulls for our diseases
and compare the words that are picked up by the pipeline vs the actual words in the literature
where the interactions are referenced... a gene/interaction recall fraction"). See
claude/possible_metrics.md for how this fits alongside the pipeline's other metrics.

Why this is a separate script and a separate metric, not just re-reading the pipeline's own
`supported` flag: the production edge-literature check
(PipelineB/agent3_confirmation.py's `_edge_literature_checks`, backed by
PipelineB/pubmed_client.py's `search_edge_citations`/`_check_edge_support`) is deliberately
frugal -- retmax=3 papers per edge, and "supported" means only "both gene symbols appear together
in the same retrieved paragraph/abstract," with no check that the surrounding text actually
DESCRIBES an interaction between them. pubmed_client.py's own EdgeCitation docstring says this
plainly: "co-mention in the same paper isn't automatically evidence of the specific claimed
edge." That's a real, already-acknowledged limitation -- this script measures it directly with an
independent check, instead of just asserting it in prose.

What this does, concretely, for every real bridge edge already checked in a real saved pipeline
run (outputs/pipeline_b_t2dm_ad.json, outputs/pipeline_b_t2dm_cad.json -- both already exist from
real live runs):
  1. Reads the pipeline's own real verdict for that edge (`supported: true/false`) straight out
     of the saved JSON -- the actual EdgeLiteratureCheck a real run already produced, not
     recomputed here.
  2. Independently re-pulls a LARGER pool of the SAME literature, via the exact same method and
     the exact same NCBI endpoints (PubMedClient.search_edge_citations) -- just RECALL_POOL_SIZE
     papers instead of the production check's 3 -- for the same two gene symbols.
  3. Scans every sentence of every retrieved passage/abstract in that larger pool for a sentence
     that names both gene symbols AND contains at least one word from a small, hand-written
     interaction vocabulary (INTERACTION_VERBS below) -- a simple, inspectable rule, written
     independently of anything inside the pipeline itself, not an LLM call.
  4. Calls an edge "gold-positive" if step 3 found at least one such sentence anywhere in the
     larger pool -- this script's own independent ground truth.
  5. RECALL = (edges gold-positive AND pipeline-supported) / (edges gold-positive) -- of the
     edges where a broader, verb-aware search finds real interaction-describing language, what
     fraction did the production (narrower, verb-blind) check actually catch.
     PRECISION = (edges gold-positive AND pipeline-supported) / (edges pipeline-supported) -- the
     mirror question: of the edges the pipeline flagged as supported, how many hold up against
     the stricter, independent check (does mere co-mention over-count real support).

MUST be run on the user's own machine, not through the Claude device bridge or cloud sandbox --
same constraint as scripts/compare_llm_models.py: both the E-utilities and BioC-PMC endpoints are
blocked from that network.

Usage:
    python3 -m scripts.edge_literature_recall --pair t2dm_cad
    python3 -m scripts.edge_literature_recall --pair t2dm_ad
    python3 -m scripts.edge_literature_recall            # default: both pairs

Output: outputs/edge_literature_recall/{pair_id}.json (per-edge detail + the recall/precision
summary) plus a printed summary table.
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

# The production edge-literature check uses retmax=3 (PipelineB/pubmed_client.py's
# search_edge_citations default). This script deliberately pulls a bigger pool of the same
# literature to build an independent gold standard -- a real edge can have supporting language in
# its 5th or 10th co-mentioning paper even when its top-3 don't mention an interaction explicitly.
RECALL_POOL_SIZE = 15

# Hand-written, deliberately simple interaction vocabulary -- a sentence naming both genes AND
# containing one of these word-stems is this script's own, independent definition of "the text is
# actually describing an interaction," not just mentioning both gene names. Not exhaustive by
# design; chosen to cover OmniPath's own interaction_type vocabulary (activates/inhibits/unknown)
# plus the common verbs real papers use for the same biological relationships.
INTERACTION_VERBS = [
    "activat", "inhibit", "phosphorylat", "dephosphorylat", "bind", "interact",
    "regulat", "promot", "suppress", "induc", "modulat", "increas", "decreas",
    "mediat", "block", "trigger", "upregulat", "downregulat", "stimulat",
    "cleav", "degrad", "signal", "target", "associat", "repress", "enhanc",
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _gene_pattern(gene: str) -> "re.Pattern":
    # Same word-boundary approach as pubmed_client.py's own _gene_pattern -- avoids a bare
    # substring match wrongly hitting an unrelated longer token.
    return re.compile(rf"\b{re.escape(gene)}\b", re.IGNORECASE)


def _sentence_has_interaction_language(sentence: str, gene_a: str, gene_b: str) -> bool:
    # Input: one sentence, the two gene symbols. Output: whether this sentence is this script's
    # own definition of "describes an interaction" -- both gene symbols present AND at least one
    # interaction-vocabulary word-stem present, all within the same sentence.
    if not (_gene_pattern(gene_a).search(sentence) and _gene_pattern(gene_b).search(sentence)):
        return False
    lowered = sentence.lower()
    return any(verb in lowered for verb in INTERACTION_VERBS)


def _passages_for_pair(client: PubMedClient, gene_a: str, gene_b: str, retmax: int) -> List[str]:
    # Re-pulls a larger pool via the exact same method/endpoints the production check uses, and
    # returns every real passage of text actually retrieved (full-text paragraph when a PMC
    # open-access version exists, else the abstract) -- the raw text this script's own
    # independent sentence scan runs over.
    edge_citations = client.search_edge_citations(gene_a, gene_b, retmax=retmax)
    return [c.matched_text for c in edge_citations if c.matched_text]


def gold_positive(client: PubMedClient, gene_a: str, gene_b: str) -> dict:
    # Independent ground truth for one edge: pulls a larger literature pool and scans every
    # sentence of every retrieved passage for real interaction-describing language naming both
    # genes. Returns the verdict plus the first supporting sentence found, kept for inspection --
    # a recall number alone isn't very checkable; the actual sentence is.
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


def load_pipeline_edge_checks(pair_id: str) -> List[dict]:
    # Reads the real, already-run edge_literature_checks straight out of a saved pipeline_b
    # output -- the pipeline's own real verdict for each bridge edge from an actual live run, not
    # re-derived here. Missing file / missing key both degrade to "nothing to check" rather than
    # crashing, since an older saved run predating this feature simply has no such checks.
    path = OUTPUTS_DIR / f"pipeline_b_{pair_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run Pipeline B for {pair_id} first "
            f"(main.py's run_pipeline_b_and_save) before this script."
        )
    data = json.loads(path.read_text())
    checks = []
    for chain_result in data.get("agent3", []):
        for check in chain_result.get("edge_literature_checks", []):
            checks.append(
                {
                    "gene_a": check["gene_a"],
                    "gene_b": check["gene_b"],
                    "pipeline_supported": check["supported"],
                }
            )
    return checks


def run_for_pair(pair_id: str, client: PubMedClient) -> dict:
    checks = load_pipeline_edge_checks(pair_id)
    # De-duplicate -- the same bridge edge can appear in more than one candidate chain for the
    # same disease pair; the pipeline's own verdict for a given gene pair doesn't depend on which
    # chain it showed up in, so only check it once.
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
