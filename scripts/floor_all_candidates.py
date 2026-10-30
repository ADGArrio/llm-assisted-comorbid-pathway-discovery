"""
Evidence floor applied to EVERY pre-trim candidate path, not only the ten the trim keeps.

Why: the trim to ten is evidence-blind (longest first, fully-signed first, at most 3 per start gene), the judge
then endorses a few, and the evidence floor (PipelineB.evidence_rerank) only runs on what survives. That leaves
open which step lets an unsupported chain through. This script asks it directly: of ALL paths the BFS finds
within max_hops, what share pass the floor, and do the ten the trim keeps pass at a lower or higher rate than the
rest?

No LLM. It reuses, unchanged:
  * Stage 2's BFS (`_bfs_shortest_paths`) and trim (`_trim_candidates`) on the saved Stage 1 gene lists,
    over a local copy of the OmniPath network (same set-up as scripts/max_hops_analysis.py);
  * Stage 3's tissue lookups (`GTExClient`, `check_tissue_expression`, `DISEASE_RELEVANT_TISSUES`), with the same
    role rule as `_confirm_chain` (start gene: disease-1 tissues; end gene: disease-2 tissues; middle genes: both);
  * the floor itself (`assess_chain`, min_fraction 0.5). Nothing about the rule is changed here.

It is a post hoc diagnostic reported for both pairs whatever it shows. The floor was written after the CAD result
was known, so this is not an out-of-sample test of the rule.

Run:  python -m scripts.floor_all_candidates            # both pairs
      python -m scripts.floor_all_candidates --enumerate-only  # BFS + list the GTEx lookups only, no network to GTEx
Writes outputs/evidence_floor_all_candidates/<pair>.json, summary.json, and a gene-tissue cache
(gtex_cache.json) so an interrupted run resumes where it stopped.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from common.config import DISEASE_RELEVANT_TISSUES, get_pair  # noqa: E402
from PipelineB import stage2_judging as a2  # noqa: E402
from PipelineB.evidence_rerank import assess_chain  # noqa: E402
from PipelineB.tissue_client import EXPRESSION_THRESHOLD, GTExClient  # noqa: E402
from scripts.max_hops_analysis import InMemoryOmniPath, load_full_network, run_bfs  # noqa: E402

OUT = os.path.join(HERE, "outputs", "evidence_floor_all_candidates")
CACHE = os.path.join(OUT, "gtex_cache.json")
PAIRS = ["t2dm_ad", "t2dm_cad"]
MAX_HOPS = a2.DEFAULT_MAX_HOPS
START = time.time()


def path_label(path):
    return ">".join([path[0].source_id] + [e.target_id for e in path])


def path_roles(path):
    start, end = path[0].source_id, path[-1].target_id
    middle = []
    for e in path[:-1]:
        g = e.target_id
        if g not in middle and g not in (start, end):
            middle.append(g)
    return start, end, middle


def needed_lookups(path, d1, d2):
    # Same rule as stage3_confirmation._confirm_chain: (gene, tissue) pairs this path needs.
    start, end, middle = path_roles(path)
    t1, t2 = DISEASE_RELEVANT_TISSUES[d1], DISEASE_RELEVANT_TISSUES[d2]
    out = [(start, t) for t in t1] + [(end, t) for t in t2]
    for g in middle:
        out += [(g, t) for t in t1 + t2]
    return out


def load_cache():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    return {}


def save_cache(cache):
    os.makedirs(OUT, exist_ok=True)
    tmp = CACHE + ".tmp"
    json.dump(cache, open(tmp, "w"))
    os.replace(tmp, CACHE)


def fetch_all(pairs_needed, cache, budget=None):
    client = GTExClient()
    todo = [k for k in sorted(pairs_needed) if k not in cache]
    print(f"GTEx lookups needed: {len(pairs_needed)}; cached: {len(pairs_needed) - len(todo)}; to fetch: {len(todo)}", flush=True)
    t0 = time.time()
    for i, k in enumerate(todo, 1):
        gene, tissue = k.split("|")
        cache[k] = client.get_median_expression(gene, tissue)  # None = no record / unresolved / rate-limited
        if budget and time.time() - START > budget:
            save_cache(cache)
            print(f'budget reached; {len(todo) - i} lookups left; run again to resume', flush=True)
            sys.exit(3)
        if i % 10 == 0:
            save_cache(cache)
            rate = (time.time() - t0) / i
            print(f"  {i}/{len(todo)} ({rate:.2f}s each, ~{rate * (len(todo) - i) / 60:.1f} min left)", flush=True)
    save_cache(cache)


def tissue_checks_for(path, d1, d2, cache):
    rows = []
    for gene, tissue in needed_lookups(path, d1, d2):
        v = cache.get(f"{gene}|{tissue}")
        rows.append({"gene_symbol": gene, "tissue": tissue, "value": v,
                     "expressed": None if v is None else bool(v >= EXPRESSION_THRESHOLD)})
    return rows


def enumerate_paths(pair_id, interactions):
    pair = get_pair(pair_id)
    saved = json.load(open(os.path.join(HERE, "outputs", f"pipeline_b_{pair_id}.json")))["stage1"]
    starts = [t["gene_symbol"] for t in sorted(saved["disease1_targets"], key=lambda t: -t["disease_assoc_score"])]
    goals = {t["gene_symbol"] for t in saved["disease2_targets"]}
    client = InMemoryOmniPath(interactions)
    paths, stats = run_bfs(client, starts, goals, MAX_HOPS)
    kept = a2._trim_candidates(paths, 10, a2.DEFAULT_MAX_PER_START_GENE)
    kept_ids = {id(p) for p in kept}
    return pair, paths, [id(p) in kept_ids for p in paths], kept


def analyse(pair, paths, is_kept, cache):
    d1, d2 = pair["disease1"], pair["disease2"]
    rows = []
    for p, kept in zip(paths, is_kept):
        checks = tissue_checks_for(p, d1, d2, cache)
        a = assess_chain({"tissue_checks": checks})
        rows.append({
            "label": path_label(p), "hops": len(p), "kept_by_trim": kept,
            "fully_signed": not any(e.interaction_type == "unknown" for e in p),
            "start_gene": p[0].source_id,
            "passes": a["passes"], "assessable": a["assessable"],
            "tissue_expressed": a["tissue_expressed"], "tissue_checked": a["tissue_checked"],
            "absent_genes": a["absent_genes"], "unchecked_genes": a["unchecked_genes"], "reasons": a["reasons"],
        })
    return rows


def summarise(rows):
    def block(rs):
        n = len(rs)
        return {"n": n, "pass": sum(r["passes"] for r in rs), "fail": sum(not r["passes"] for r in rs),
                "pass_rate": round(sum(r["passes"] for r in rs) / n, 3) if n else None,
                "fail_absent_gene": sum(bool(r["absent_genes"]) for r in rs),
                "unassessable": sum(not r["assessable"] for r in rs)}
    by_hops = {h: block([r for r in rows if r["hops"] == h]) for h in sorted({r["hops"] for r in rows})}
    return {"all": block(rows), "kept_by_trim": block([r for r in rows if r["kept_by_trim"]]),
            "not_kept": block([r for r in rows if not r["kept_by_trim"]]), "by_hops": by_hops,
            "fully_signed": block([r for r in rows if r["fully_signed"]]),
            "has_unknown_edge": block([r for r in rows if not r["fully_signed"]])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None, help="stop fetching after this many seconds (re-run to resume)")
    ap.add_argument("--enumerate-only", action="store_true", help="only enumerate paths and lookups; no GTEx calls")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("loading OmniPath ...", flush=True)
    interactions = load_full_network()
    per_pair, need = {}, set()
    for pid in PAIRS:
        pair, paths, is_kept, kept = enumerate_paths(pid, interactions)
        per_pair[pid] = (pair, paths, is_kept)
        for p in paths:
            for g, t in needed_lookups(p, pair["disease1"], pair["disease2"]):
                need.add(f"{g}|{t}")
        print(f"{pid}: {len(paths)} paths within {MAX_HOPS} hops, {len(kept)} kept by trim; "
              f"hops {dict(sorted(Counter(len(p) for p in paths).items()))}", flush=True)
    print(f"unique (gene, tissue) lookups: {len(need)}; genes: {len({k.split('|')[0] for k in need})}", flush=True)
    if args.enumerate_only:
        return
    cache = load_cache()
    fetch_all(need, cache, args.budget)
    # one retry pass for lookups that came back empty (no record is real; a dropped request is not)
    empty = [k for k in need if cache.get(k) is None]
    print(f"empty after first pass: {len(empty)}; retrying once", flush=True)
    client = GTExClient()
    for k in empty:
        gene, tissue = k.split("|")
        cache[k] = client.get_median_expression(gene, tissue)
    save_cache(cache)
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "max_hops": MAX_HOPS,
               "expression_threshold": EXPRESSION_THRESHOLD, "min_fraction": 0.5,
               "note": "post hoc diagnostic; the rule was written after the CAD result was known"}
    for pid, (pair, paths, is_kept) in per_pair.items():
        rows = analyse(pair, paths, is_kept, cache)
        s = summarise(rows)
        json.dump({"pair": pid, "summary": s, "paths": rows}, open(os.path.join(OUT, f"{pid}.json"), "w"), indent=1)
        summary[pid] = s
        print(f"\n{pid}: pass {s['all']['pass']}/{s['all']['n']}; kept-by-trim {s['kept_by_trim']['pass']}/{s['kept_by_trim']['n']}; "
              f"not kept {s['not_kept']['pass']}/{s['not_kept']['n']}", flush=True)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
