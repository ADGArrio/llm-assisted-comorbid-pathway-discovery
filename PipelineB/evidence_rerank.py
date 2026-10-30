"""Evidence floor: Stage 3's tissue-expression result decides which of Stage 2's chains survive, by a rule written in code.

A chain FAILS the floor if (a) any gene on it was checked in at least one tissue and is expressed in none of the tissues
checked for it, or (b) fewer than `min_fraction` (default 0.5) of its tissue checks with data came back expressed.
Chains that pass keep their debate order; failing chains move below them, also in debate order. No chain is dropped.
If every chain fails, `abstains` is True and no chain is recommended.

    python -m PipelineB.evidence_rerank --result outputs/pipeline_b_t2dm_cad_nometab.json
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_MIN_FRACTION = 0.5


# Count tissue checks per gene. Input: one chain's tissue_checks (dicts). Output: {gene: {"checked": checks with data,
# "expressed": checks expressed}}.
def gene_tissue_summary(tissue_checks: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for t in tissue_checks:
        g = out.setdefault(t["gene_symbol"], {"checked": 0, "expressed": 0})
        if t.get("expressed") is not None:
            g["checked"] += 1
            g["expressed"] += int(bool(t["expressed"]))
    return out


# Apply the evidence floor to one chain. Input: its Stage 3 result (dict) and the minimum expressed fraction. Output: a
# dict with whether it passes and the reasons it does not.
def assess_chain(stage3_chain: dict, min_fraction: float = DEFAULT_MIN_FRACTION) -> dict:
    genes = gene_tissue_summary(stage3_chain.get("tissue_checks", []))
    absent = sorted(g for g, s in genes.items() if s["checked"] > 0 and s["expressed"] == 0)
    unchecked = sorted(g for g, s in genes.items() if s["checked"] == 0)
    checked = sum(s["checked"] for s in genes.values())
    expressed = sum(s["expressed"] for s in genes.values())
    fraction: Optional[float] = expressed / checked if checked else None
    reasons = []
    if absent:
        reasons.append(f"expressed in none of the checked tissues: {', '.join(absent)}")
    if fraction is not None and fraction < min_fraction:
        reasons.append(f"only {expressed}/{checked} tissue checks expressed (floor {min_fraction:.0%})")
    return {
        "passes": not reasons,
        "assessable": checked > 0,  # no tissue data at all: not penalised, but flagged
        "tissue_expressed": expressed,
        "tissue_checked": checked,
        "tissue_fraction": fraction,
        "absent_genes": absent,
        "unchecked_genes": unchecked,
        "reasons": reasons,
    }


# Reorder chains by the evidence floor. Input: Stage 2's chains in debate order, Stage 3's per-chain results in the same
# order, and the minimum fraction. Output: the new order, per-chain assessments, and whether the top chain changed or
# the step abstains.
def rerank(chains: List[dict], stage3: List[dict], min_fraction: float = DEFAULT_MIN_FRACTION) -> dict:
    if len(chains) != len(stage3):
        raise ValueError(f"{len(chains)} chains but {len(stage3)} Stage 3 results")
    assessments = [assess_chain(a, min_fraction) for a in stage3]
    order = [i for i, a in enumerate(assessments) if a["passes"]] + [i for i, a in enumerate(assessments) if not a["passes"]]
    n_pass = sum(1 for a in assessments if a["passes"])
    rows = []
    for new_rank, i in enumerate(order):
        rows.append({
            "chain_index": i,
            "original_rank": i + 1,
            "new_rank": new_rank + 1,
            "elo_rating": chains[i].get("elo_rating"),
            **assessments[i],
        })
    return {
        "rule": {"min_fraction": min_fraction, "absent_gene_fails": True},
        "n_chains": len(chains),
        "n_pass": n_pass,
        "new_order": order,
        "top_changed": bool(order) and order[0] != 0,
        "top_original_passes": bool(assessments) and assessments[0]["passes"],
        "abstains": bool(chains) and n_pass == 0,
        "chains": rows,
    }


# Input: a chain dict. Output: its genes joined by ">" (the label used in result tables).
def chain_label(chain: dict) -> str:
    edges = chain["edges"]
    return ">".join([edges[0]["source_gene"]] + [e["target_gene"] for e in edges])


# Apply the floor to a saved result file. Input: a saved pipeline_b_<pair>.json as a dict and the minimum fraction.
# Output: the rerank dict, or None if it has no chains.
def rerank_saved(result: dict, min_fraction: float = DEFAULT_MIN_FRACTION) -> Optional[dict]:
    a2 = result.get("stage2")
    a3 = result.get("stage3") or []
    if not a2 or not a2.get("chains") or not a3:
        return None
    out = rerank(a2["chains"], a3, min_fraction)
    for row in out["chains"]:
        row["label"] = chain_label(a2["chains"][row["chain_index"]])
    return out


# Render a rerank result as plain text. Input: the rerank dict. Output: the text.
def _format(out: dict) -> str:
    lines = [f"Evidence floor: {out['n_pass']}/{out['n_chains']} chains pass"
             + ("  -> NO chain recommended" if out["abstains"] else "")]
    for r in out["chains"]:
        mark = "PASS" if r["passes"] else "FAIL"
        frac = "n/a" if r["tissue_fraction"] is None else f"{r['tissue_expressed']}/{r['tissue_checked']}"
        lines.append(f"  new #{r['new_rank']} (debate #{r['original_rank']}, Elo {r['elo_rating']}) {mark} {r.get('label', '')}  tissue {frac}")
        for why in r["reasons"]:
            lines.append(f"      - {why}")
    return "\n".join(lines)


# Command-line entry point: read a saved result, print the evidence-floor outcome and save it as JSON.
def _cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--result", required=True, help="a saved outputs/pipeline_b_<pair>*.json")
    ap.add_argument("--min-fraction", type=float, default=DEFAULT_MIN_FRACTION)
    ap.add_argument("--out", default=None, help="write the rerank JSON here (default: outputs/evidence_rerank/<file name>)")
    args = ap.parse_args()
    result = json.loads(Path(args.result).read_text())
    out = rerank_saved(result, args.min_fraction)
    if out is None:
        print("no chains with Stage 3 results in that file")
        return
    print(_format(out))
    dest = Path(args.out) if args.out else Path("outputs/evidence_rerank") / Path(args.result).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved {dest}")


if __name__ == "__main__":
    _cli()
