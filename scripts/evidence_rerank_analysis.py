"""Quantify the evidence-aware re-rank (PipelineB/evidence_rerank.py) on every saved run.

    python -m scripts.evidence_rerank_analysis            # saved runs only
    python -m scripts.evidence_rerank_analysis --null-live  # also query GTEx for the random baseline genes (~3 min per pair)

Questions answered, each with counts a reader can check:
 1. How often does the debate's top chain fail the floor, and how often would the step abstain?
 2. Does the floor make the recommended chain more stable from run to run?
 3. Chains that pass vs fail, on evidence the rule did NOT use (bridge-gene GWAS hits, edge literature).
 4. How often does the same floor reject random bridge-gene sets (live GTEx for the baseline genes)?
Every run file is read once; byte-identical copies (archives of an unchanged file) are counted once.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PipelineB.evidence_rerank import DEFAULT_MIN_FRACTION, assess_chain, chain_label, gene_tissue_summary, rerank_saved

OUT = Path("outputs")
DEST = OUT / "evidence_rerank"


def load_runs():
    # Input: none. Output: one dict per distinct saved run that has chains and Stage 3 results.
    files = sorted(OUT.glob("pipeline_b_t2dm_*.json")) + sorted(OUT.glob("archive_*/pipeline_b_t2dm_*.json"))
    seen, runs = set(), []
    for f in files:
        raw = f.read_bytes()
        h = hashlib.md5(raw).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        d = json.loads(raw)
        a2 = d.get("stage2") or {}
        if not a2.get("chains") or not d.get("stage3"):
            continue
        bfs = a2.get("bfs_stats") or {}
        runs.append({
            "file": str(f), "pair": "t2dm_ad" if "t2dm_ad" in f.name else "t2dm_cad", "data": d,
            "hardened": bool(bfs.get("judge_control_pass")),
            "metabolites": bool(bfs.get("metabolites_included")),
            "rerank": rerank_saved(d),
        })
    return runs


def stability(runs):
    # Input: runs of one pair. Output: distinct top chains and the modal share before and after the floor.
    before = Counter(chain_label(r["data"]["stage2"]["chains"][0]) for r in runs)
    after = Counter()
    for r in runs:
        rr = r["rerank"]
        after["NO CHAIN (abstains)" if rr["abstains"] else rr["chains"][0]["label"]] += 1
    n = len(runs)
    def stats(c):
        return {"runs": n, "distinct": len(c), "modal_share": round(max(c.values()) / n, 3) if n else None, "counts": dict(c.most_common())}
    return {"before": stats(before), "after": stats(after)}


def held_out(runs):
    # Input: all runs. Output: per unique chain, whether it passes and what held-out evidence it has.
    unique = {}
    for r in runs:
        d = r["data"]
        for i, row in enumerate(r["rerank"]["chains"]):
            c = d["stage2"]["chains"][row["chain_index"]]
            a3 = d["stage3"][row["chain_index"]]
            key = (r["pair"], row["label"])
            if key in unique:
                continue
            bg = [b for b in a3.get("bridge_gwas_checks", []) if b["confirmed"] or not b.get("possibly_incomplete")]
            el = a3.get("edge_literature_checks", [])
            unique[key] = {
                "pair": r["pair"], "label": row["label"], "passes": row["passes"],
                "bridge_checks": len(bg), "bridge_hits": sum(1 for b in bg if b["confirmed"]),
                "edges_checked": sum(1 for e in el if e.get("searched")), "edges_supported": sum(1 for e in el if e.get("supported")),
            }
    groups = {}
    for name, pred in (("pass", True), ("fail", False)):
        rows = [u for u in unique.values() if u["passes"] is pred]
        bc, bh = sum(u["bridge_checks"] for u in rows), sum(u["bridge_hits"] for u in rows)
        ec, es = sum(u["edges_checked"] for u in rows), sum(u["edges_supported"] for u in rows)
        groups[name] = {"unique_chains": len(rows), "bridge_gene_x_disease_checks": bc, "bridge_hits": bh,
                        "bridge_hit_rate": round(bh / bc, 3) if bc else None,
                        "bridge_edges_searched": ec, "bridge_edges_supported": es,
                        "edge_support_rate": round(es / ec, 3) if ec else None}
    return {"unique_chains": list(unique.values()), "groups": groups}


def middle_genes(label):
    parts = label.split(">")
    return parts[1:-1]


def null_floor(runs, pair, n_sim=20000):
    # Input: runs, pair. Output: live GTEx for the pair's random baseline genes, then how often random bridge-gene sets
    # of each observed size contain a gene expressed in none of the bridge tissues, against the pipeline's own chains.
    from PipelineB.tissue_client import GTExClient, check_tissue_expression
    final = [r for r in runs if r["pair"] == pair and r["file"].endswith(f"{pair}_nometab.json")]
    if not final:
        return None
    nb = final[0]["data"]["stage3"][0]["null_baseline"]
    cache = DEST / f"null_gtex_{pair}.json"
    if cache.exists():
        per_gene = json.loads(cache.read_text())
    else:
        client, per_gene = GTExClient(), {}
        for g in nb["genes"]:
            flags = [check_tissue_expression(g, t, client).expressed for t in nb["tissues"]]
            per_gene[g] = flags
        cache.write_text(json.dumps(per_gene))
    absent = [g for g, f in per_gene.items() if any(x is not None for x in f) and not any(x for x in f if x is not None)]
    usable = [g for g, f in per_gene.items() if any(x is not None for x in f)]
    # observed: pipeline chains' middle (bridge) genes, same clause
    obs = []
    for r in runs:
        if r["pair"] != pair:
            continue
        for row in r["rerank"]["chains"]:
            a3 = r["data"]["stage3"][row["chain_index"]]
            s = gene_tissue_summary(a3["tissue_checks"])
            mids = [m for m in middle_genes(row["label"]) if m in s and s[m]["checked"]]
            obs.append((row["label"], len(mids), any(s[m]["expressed"] == 0 for m in mids)))
    uniq = {l: (k, f) for l, k, f in obs}
    sizes = Counter(k for k, _ in uniq.values())
    rng = random.Random(20261025)
    sim = {}
    for k in sorted(sizes):
        hits = sum(any(g in absent for g in rng.sample(usable, k)) for _ in range(n_sim))
        sim[k] = round(hits / n_sim, 3)
    return {"baseline_genes_with_data": len(usable), "baseline_genes_absent_in_all_bridge_tissues": len(absent),
            "absent_genes": absent, "random_set_failure_rate_by_size": sim,
            "pipeline_unique_chains": len(uniq), "pipeline_chains_with_absent_bridge_gene": sum(1 for _, f in uniq.values() if f),
            "pipeline_chain_bridge_sizes": dict(sizes)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--null-live", action="store_true")
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    report = {"rule_min_fraction": DEFAULT_MIN_FRACTION, "n_runs": len(runs), "pairs": {}}
    print(f"{len(runs)} distinct saved runs with chains and Stage 3 results\n")
    for pair in ("t2dm_ad", "t2dm_cad"):
        rs = [r for r in runs if r["pair"] == pair]
        sub = {"runs": len(rs), "hardened_runs": sum(r["hardened"] for r in rs)}
        sub["top_chain_fails_floor"] = sum(1 for r in rs if not r["rerank"]["top_original_passes"])
        sub["top_changed"] = sum(1 for r in rs if r["rerank"]["top_changed"])
        sub["abstains"] = sum(1 for r in rs if r["rerank"]["abstains"])
        sub["chains_total"] = sum(r["rerank"]["n_chains"] for r in rs)
        sub["chains_passing"] = sum(r["rerank"]["n_pass"] for r in rs)
        sub["stability"] = stability(rs)
        hs = [r for r in rs if r["hardened"]]
        sub["stability_hardened_only"] = stability(hs) if hs else None
        if args.null_live:
            sub["null_floor"] = null_floor(runs, pair)
        report["pairs"][pair] = sub
        print(f"== {pair}: {sub['runs']} runs ({sub['hardened_runs']} hardened)")
        print(f"   debate top chain fails the floor in {sub['top_chain_fails_floor']}/{sub['runs']} runs; recommended chain changes in "
              f"{sub['top_changed']}; step abstains in {sub['abstains']}")
        print(f"   chains passing: {sub['chains_passing']}/{sub['chains_total']}")
        for k in ("before", "after"):
            st = sub["stability"][k]
            print(f"   top chain {k}: {st['distinct']} distinct over {st['runs']} runs, modal share {st['modal_share']}  {st['counts']}")
        if args.null_live and sub.get("null_floor"):
            print(f"   null: {sub['null_floor']}")
    report["held_out"] = held_out(runs)
    print("\nHeld-out evidence by floor outcome (unique chains, pooled over pairs):")
    for k, g in report["held_out"]["groups"].items():
        print(f"   {k}: {g}")
    (DEST / "analysis.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {DEST / 'analysis.json'}")


if __name__ == "__main__":
    main()
