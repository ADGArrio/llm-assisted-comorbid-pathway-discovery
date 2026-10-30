"""
Why max_hops = 5?  Measures what each extra hop buys and costs, against a degree-matched random baseline.
(Added 2026-10-15.  The previous justification -- "5 sits inside the 4-6 hop small-world diameter" in
the Stage 2 BFS and prompt recalibration note -- was an argument, not a measurement.)

It runs Stage 2's OWN breadth-first search (`PipelineB.stage2_judging._bfs_shortest_paths`, no LLM) on the
saved Stage 1 gene lists, to depth `--max-depth` (default 10), over a local copy of the OmniPath network
(one download, ~13 s) so the search -- and a random baseline -- can be repeated quickly and identically.
Because the search is layered, "max_hops = h" is exactly "the paths found at depth <= h".

Per hop count it reports
  * yield:   new candidate paths found at exactly that depth (and cumulative; goal genes reached);
  * signal:  the same count for DEGREE-MATCHED random gene sets (each observed start gene is replaced by a
             random gene of similar out-degree, each goal gene by one of similar in-degree) -- so the ratio
             observed / random asks whether the two diseases' genes are connected more closely than
             equally-connected random genes are, at that depth, rather than just through hubs;
  * quality: share of new paths in which every edge has a curated direction (no "unknown" edge);
  * cost:    genes queried in that BFS layer (the pipeline caps a layer at 3000), edges returned;
  * what the final candidates look like: the pipeline keeps the `max_paths` (10) LONGEST paths after the
             search, so a larger max_hops directly makes the candidates longer and less often fully signed.

Run:  python -m scripts.max_hops_analysis            # both pairs, 100 random replicates, writes figures
      python -m scripts.max_hops_analysis --replicates 20 --no-figures
Writes outputs/max_hops_analysis/<pair>.json, outputs/max_hops_analysis/summary.json and
docs/figures/fig_maxhops_*.png (copied to ../Thesis/figures/).

Scope notes: protein-only (no MetalinksDB metabolites) and the core OmniPath dataset, as the table in
the Stage 2 code walkthrough; today's OmniPath release (recorded in the output).
"""
import argparse
import bisect
import json
import math
import os
import random
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MAX_DEPTH = 10
DEFAULT_REPLICATES = 100
DEFAULT_PAIRS = ["t2dm_ad", "t2dm_cad"]
FRONTIER_CAP = 3000  # mirrors PipelineB.stage2_judging._MAX_FRONTIER_SIZE (re-read at run time, see run_pair)


class InMemoryOmniPath:
    """Serves get_outgoing_interactions from a list already in memory -- same interface as OmniPathClient,
    so Stage 2's real BFS can run on it unchanged, thousands of times, without HTTP."""

    def __init__(self, interactions, datasets="omnipath"):
        self.datasets = datasets
        self._by_source = defaultdict(list)
        for i in interactions:
            self._by_source[i.source_gene].append(i)

    def get_outgoing_interactions(self, source_genes):
        out = []
        for g in sorted(set(source_genes)):
            out.extend(self._by_source.get(g, []))
        return out


def load_full_network(datasets: str = "omnipath"):
    # One request for the whole directed network, parsed with the pipeline's own parser.
    import requests
    from PipelineB.omnipath_client import DEFAULT_BASE_URL, OmniPathClient

    resp = requests.get(DEFAULT_BASE_URL, params={"genesymbols": "1", "datasets": datasets}, timeout=180)
    resp.raise_for_status()
    return OmniPathClient()._parse_response(resp.text)


def run_bfs(client, start_genes: Sequence[str], goal_genes: set, max_depth: int):
    # Stage 2's real search, trim disabled, returning (paths, stats).
    from PipelineB import stage2_judging as a2

    stats: dict = {}
    paths = a2._bfs_shortest_paths(client, None, list(start_genes), set(goal_genes), max_depth, 10 ** 9, stats=stats)
    return paths, stats


def _has_unknown(path) -> bool:
    return any(e.interaction_type == "unknown" for e in path)


def per_depth_table(paths, stats: dict, max_depth: int, max_paths: int = 10) -> dict:
    # Input: all BFS paths (untrimmed), the BFS stats, depth. Output: lists indexed by hop count 1..max_depth.
    from PipelineB import stage2_judging as a2

    by_len = defaultdict(list)
    for p in paths:
        by_len[len(p)].append(p)
    layers = {l["depth"]: l for l in stats.get("layers", [])}
    rows = []
    cumulative = 0
    for h in range(1, max_depth + 1):
        new = by_len.get(h, [])
        cumulative += len(new)
        signed = sum(1 for p in new if not _has_unknown(p))
        # What the pipeline would hand to the LLM if max_hops were h: the shipped trim rule (Stage 2's own
        # _trim_candidates: longest first, fully-signed first among equal lengths, at most DEFAULT_MAX_PER_START_GENE
        # per start gene) applied to the paths of length <= h. `legacy` = the pre-2026-10-18 rule (longest first only),
        # kept for comparison of how many distinct start genes survive.
        available = [p for p in paths if len(p) <= h]
        kept = a2._trim_candidates(available, max_paths, a2.DEFAULT_MAX_PER_START_GENE)
        legacy_sorted = sorted(available, key=lambda p: -len(p))[:max_paths]  # the exact old rule, for the comparison
        layer = layers.get(h, {})
        rows.append({
            "hops": h,
            "new_paths": len(new),
            "cumulative_paths": cumulative,
            "fully_signed_new_paths": signed,
            "fraction_fully_signed": round(signed / len(new), 3) if new else None,
            "genes_queried": layer.get("frontier_size", 0),
            "layer_truncated": bool(layer.get("truncated", False)),
            "edges_returned": layer.get("edges_returned", 0),
            "kept_n": len(kept),
            "kept_mean_hops": round(statistics.mean(len(p) for p in kept), 2) if kept else None,
            "kept_fraction_with_unknown_edge": round(sum(_has_unknown(p) for p in kept) / len(kept), 3) if kept else None,
            "kept_distinct_start_genes": len({p[0].source_id for p in kept}),
            "legacy_kept_distinct_start_genes": len({p[0].source_id for p in legacy_sorted}),
            "legacy_kept_fraction_with_unknown_edge": round(sum(_has_unknown(p) for p in legacy_sorted) / len(legacy_sorted), 3) if legacy_sorted else None,
        })
    return {"rows": rows, "n_goal_genes": stats.get("n_goal_genes"), "n_start_genes": stats.get("n_start_genes"),
            "total_paths": len(paths)}


def _bin(degree: int) -> int:
    return int(math.log2(degree + 1))  # 0, 1, 2-3, 4-7, ... (log2 buckets)


def degree_pools(interactions):
    # Returns (out_pools, in_pools, out_deg, in_deg): gene lists per log2 degree bucket, over every gene in the network.
    out_deg, in_deg = Counter(), Counter()
    genes = set()
    for i in interactions:
        out_deg[i.source_gene] += 1
        in_deg[i.target_gene] += 1
        genes.add(i.source_gene)
        genes.add(i.target_gene)
    out_pools, in_pools = defaultdict(list), defaultdict(list)
    for g in sorted(genes):
        out_pools[_bin(out_deg[g])].append(g)
        in_pools[_bin(in_deg[g])].append(g)
    return out_pools, in_pools, out_deg, in_deg


def sample_matched(observed: Sequence[str], pools: Dict[int, List[str]], degree: Counter, rng: random.Random) -> List[str]:
    # For each observed gene pick a random network gene from the same degree bucket (distinct picks where the
    # bucket allows; a gene absent from the network counts as degree 0).
    chosen: List[str] = []
    taken: set = set()
    for g in observed:
        pool = pools.get(_bin(degree.get(g, 0)), [])
        candidates = [x for x in pool if x not in taken] or pool
        pick = rng.choice(candidates)
        taken.add(pick)
        chosen.append(pick)
    return chosen


def null_new_paths(interactions, start_genes, goal_genes, replicates: int, max_depth: int, seed: int = 0) -> dict:
    # Runs the real BFS on `replicates` degree-matched random start/goal sets. Returns, per hop count, the mean and the
    # 2.5/97.5 percentiles of the number of NEW paths found at that depth.
    out_pools, in_pools, out_deg, in_deg = degree_pools(interactions)
    client = InMemoryOmniPath(interactions)
    rng = random.Random(seed)
    counts = {h: [] for h in range(1, max_depth + 1)}
    for _ in range(replicates):
        starts = sample_matched(start_genes, out_pools, out_deg, rng)
        goals = set(sample_matched(sorted(goal_genes), in_pools, in_deg, rng))
        paths, _ = run_bfs(client, starts, goals, max_depth)
        c = Counter(len(p) for p in paths)
        for h in counts:
            counts[h].append(c.get(h, 0))
    summary = {}
    for h, vals in counts.items():
        vals_sorted = sorted(vals)
        lo = vals_sorted[max(0, int(0.025 * len(vals_sorted)))]
        hi = vals_sorted[min(len(vals_sorted) - 1, int(math.ceil(0.975 * len(vals_sorted))) - 1)]
        summary[h] = {"mean": round(statistics.mean(vals), 2), "p2_5": lo, "p97_5": hi, "n": len(vals)}
    return summary


def empirical_p(observed: int, null_values: Sequence[int]) -> float:
    # One-sided: fraction of null replicates with at least as many new paths as observed (with +1 smoothing).
    ordered = sorted(null_values)
    at_least = len(ordered) - bisect.bisect_left(ordered, observed)
    return round((at_least + 1) / (len(ordered) + 1), 3)


def run_pair(pair_id: str, interactions, replicates: int, max_depth: int, seed: int = 0) -> dict:
    from PipelineB import stage2_judging as a2

    saved = json.load(open(os.path.join(HERE, "outputs", f"pipeline_b_{pair_id}.json")))["stage1"]
    starts = [t["gene_symbol"] for t in sorted(saved["disease1_targets"], key=lambda t: -t["disease_assoc_score"])]
    goals = {t["gene_symbol"] for t in saved["disease2_targets"]}
    client = InMemoryOmniPath(interactions)
    paths, stats = run_bfs(client, starts, goals, max_depth)
    table = per_depth_table(paths, stats, max_depth)
    null = null_new_paths(interactions, starts, goals, replicates, max_depth, seed) if replicates else {}
    for row in table["rows"]:
        n = null.get(row["hops"])
        if n:
            row["null_mean_new_paths"], row["null_p2_5"], row["null_p97_5"] = n["mean"], n["p2_5"], n["p97_5"]
            row["enrichment_over_null"] = round(row["new_paths"] / n["mean"], 2) if n["mean"] > 0 else None
    return {
        "pair": pair_id, "max_depth": max_depth, "replicates": replicates, "seed": seed,
        "frontier_cap": a2._MAX_FRONTIER_SIZE, "default_max_hops": a2.DEFAULT_MAX_HOPS,
        "omnipath_edges_in_network": len(interactions),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **table,
    }


# --------------------------------------------------------------------------- figures
def make_figures(results: Dict[str, dict], out_dirs: Sequence[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, GREY, INK, MUTED, AXIS, GRID = "#2a78d6", "#eb6834", "#a9a79f", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.8, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    colors = {"t2dm_ad": BLUE, "t2dm_cad": ORANGE}
    labels = {"t2dm_ad": "T2DM → AD", "t2dm_cad": "T2DM → CAD"}
    pairs = [p for p in DEFAULT_PAIRS if p in results]
    depth = max(max(r["hops"] for r in results[p]["rows"] if r["new_paths"] or r["genes_queried"]) for p in pairs)
    depth = min(max(depth, 6), results[pairs[0]]["max_depth"])
    hops = list(range(1, depth + 1))
    default_h = results[pairs[0]]["default_max_hops"]
    width = 0.27

    def style(ax, ylabel, title):
        ax.set_xticks(hops)
        ax.set_xlabel("Hop count (path length)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, loc="left", pad=18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    def mark_default(ax):
        ax.axvline(default_h + 0.5, color=INK, linestyle=":", linewidth=1)
        # Label sits just above the axes (blended transform) so it can never collide with a bar.
        ax.text(default_h + 0.5, 1.01, f"shipped max_hops = {default_h}", fontsize=7.5, va="bottom", ha="center", color=MUTED,
                transform=ax.get_xaxis_transform())

    def row(p, h):
        return next(r for r in results[p]["rows"] if r["hops"] == h)

    # ---- Figure 1: yield and signal
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3))
    ax = axes[0]
    for k, p in enumerate(pairs):
        ax.bar([h + (k - 1) * width for h in hops], [row(p, h)["new_paths"] for h in hops], width=width, color=colors[p], label=labels[p])
    null_means = [row(pairs[0], h).get("null_mean_new_paths") for h in hops]
    if all(v is not None for v in null_means):
        lo = [row(pairs[0], h)["null_mean_new_paths"] - row(pairs[0], h)["null_p2_5"] for h in hops]
        hi = [row(pairs[0], h)["null_p97_5"] - row(pairs[0], h)["null_mean_new_paths"] for h in hops]
        ax.bar([h + width for h in hops], null_means, width=width, color=GREY, label="Degree-matched random genes (mean, 95% range)",
               yerr=[lo, hi], error_kw={"ecolor": MUTED, "elinewidth": 0.9, "capsize": 2})
    style(ax, "New candidate paths found at this depth", "A. Yield per hop")
    mark_default(ax)
    ax2 = axes[1]
    for k, p in enumerate(pairs):
        # Ratios are only meaningful where the baseline finds a few paths; beyond that both sides are ~0.
        ok = [h for h in hops if (row(p, h).get("null_mean_new_paths") or 0) >= 3]
        ax2.bar([h + (k - 0.5) * width * 1.4 for h in ok], [row(p, h)["enrichment_over_null"] or 0 for h in ok], width=width * 1.4, color=colors[p], label=labels[p])
    ax2.axhline(1, color=INK, linestyle="--", linewidth=1)
    ax2.text(hops[-1] + 0.4, 1.0, "chance", fontsize=8, va="bottom", ha="right", color=MUTED)
    ax2.text(default_h + 0.65, 0.06, "no ratio shown where the random\nbaseline averages < 3 paths", fontsize=7.5, va="bottom", ha="left", color=MUTED)
    style(ax2, "Observed ÷ random baseline", "B. Enrichment over degree-matched random genes")
    mark_default(ax2)
    handles, lab = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.legend(handles, lab, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_maxhops_yield.png", out_dirs)

    # ---- Figure 2: quality and cost
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3))
    ax = axes[0]
    for k, p in enumerate(pairs):
        vals = [row(p, h)["fraction_fully_signed"] for h in hops]
        ax.bar([h + (k - 0.5) * width * 1.4 for h in hops], [(v or 0) for v in vals], width=width * 1.4, color=colors[p], label=labels[p])
    ax.set_ylim(0, 1.0)
    for k, p in enumerate(pairs):  # a zero bar is ambiguous (none signed vs no paths): label the former as k/n
        for h in hops:
            r_ = row(p, h)
            if r_["new_paths"] and not r_["fully_signed_new_paths"]:
                ax.text(h + (k - 0.5) * width * 1.4, 0.01, f"0/{r_['new_paths']}", fontsize=7, ha="center", va="bottom", color=MUTED)
    style(ax, "Share of new paths with every edge signed", "A. Direction known for every hop")
    ax.yaxis.set_major_formatter(__import__("matplotlib.ticker", fromlist=["PercentFormatter"]).PercentFormatter(xmax=1.0))
    mark_default(ax)
    ax2 = axes[1]
    for k, p in enumerate(pairs):
        ax2.bar([h + (k - 0.5) * width * 1.4 for h in hops], [row(p, h)["genes_queried"] for h in hops], width=width * 1.4, color=colors[p], label=labels[p])
    ax2.axhline(results[pairs[0]]["frontier_cap"], color=INK, linestyle="--", linewidth=1)
    ax2.set_ylim(0, results[pairs[0]]["frontier_cap"] * 1.12)
    ax2.text(hops[-1] + 0.4, results[pairs[0]]["frontier_cap"] * 1.015, "frontier cap (3000)", fontsize=8, va="bottom", ha="right", color=MUTED)
    style(ax2, "Genes queried in this BFS layer", "B. Search cost per layer")
    mark_default(ax2)
    handles, lab = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.legend(handles, lab, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_maxhops_quality_cost.png", out_dirs)

    # ---- Figure 3: what the 10 candidates handed to the LLM look like, by hop budget
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3))
    ax = axes[0]
    for k, p in enumerate(pairs):
        ax.bar([h + (k - 0.5) * width * 1.4 for h in hops], [row(p, h)["kept_mean_hops"] or 0 for h in hops], width=width * 1.4, color=colors[p], label=labels[p])
    style(ax, "Mean length of the 10 kept candidates (hops)", "A. Candidate length grows with the budget")
    ax.set_xlabel("max_hops budget")
    mark_default(ax)
    ax2 = axes[1]
    for k, p in enumerate(pairs):
        ax2.bar([h + (k - 0.5) * width * 1.4 for h in hops], [row(p, h)["kept_fraction_with_unknown_edge"] or 0 for h in hops], width=width * 1.4, color=colors[p], label=labels[p])
    ax2.set_ylim(0, 1.0)
    for k, p in enumerate(pairs):  # a zero bar is invisible: say "0%" so it is not read as missing data
        for h in hops:
            if row(p, h)["kept_fraction_with_unknown_edge"] == 0:
                ax2.text(h + (k - 0.5) * width * 1.4, 0.01, "0%", fontsize=7, ha="center", va="bottom", color=MUTED)
    style(ax2, "Kept candidates with ≥ 1 unsigned edge", "B. Unsigned edges among the kept candidates")
    ax2.set_xlabel("max_hops budget")
    ax2.yaxis.set_major_formatter(__import__("matplotlib.ticker", fromlist=["PercentFormatter"]).PercentFormatter(xmax=1.0))
    mark_default(ax2)
    handles, lab = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.legend(handles, lab, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_maxhops_candidates.png", out_dirs)


def save(fig, name: str, out_dirs: Sequence[str]):
    import matplotlib.pyplot as plt
    first = None
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        if first is None:
            fig.savefig(path, dpi=200, bbox_inches="tight")
            first = path
        else:
            shutil.copy(first, path)
    plt.close(fig)
    print("wrote", name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    ap.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "outputs", "max_hops_analysis"))
    ap.add_argument("--figure-dirs", nargs="+", default=[os.path.join(HERE, "docs", "figures"), os.path.join(HERE, "..", "Thesis", "figures")])
    args = ap.parse_args()

    interactions = load_full_network()
    print(f"OmniPath core network: {len(interactions)} directed edges")
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for pair in args.pairs:
        results[pair] = run_pair(pair, interactions, args.replicates, args.max_depth, args.seed)
        with open(os.path.join(args.out_dir, f"{pair}.json"), "w") as f:
            json.dump(results[pair], f, indent=2)
        print(f"\n{pair}: total paths {results[pair]['total_paths']}")
        print("hops  new  cum  signed%  genes_queried  null_mean[2.5-97.5]  enrich  kept_mean_hops  kept_unk%")
        for r in results[pair]["rows"]:
            sp = "-" if r["fraction_fully_signed"] is None else f"{r['fraction_fully_signed']*100:.0f}"
            nm = f"{r.get('null_mean_new_paths', '-')}[{r.get('null_p2_5', '-')}-{r.get('null_p97_5', '-')}]"
            print(f"{r['hops']:>4} {r['new_paths']:>4} {r['cumulative_paths']:>4} {sp:>7} {r['genes_queried']:>14}  {nm:>19} {str(r.get('enrichment_over_null', '-')):>7} {str(r['kept_mean_hops']):>14} {str(r['kept_fraction_with_unknown_edge']):>10}")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    if not args.no_figures:
        make_figures(results, args.figure_dirs)


if __name__ == "__main__":
    main()
