"""Edge-provenance audit for Stage 2: evidence that the LLM only ranks chains the code already found and cannot add an edge.

Shown: every edge in every saved chain exists in the source database (OmniPath, or MetalinksDB for metabolite hops) with the same source,
target and direction label; chains are rebuilt from the code-found path, never from model text. Not shown: that the model reasons only from
the supplied edges, since a plausibility judgement draws on training knowledge. Measured instead: how often a rationale names a gene that
is neither in the chain nor in the citation titles the model was shown (exact gene symbols OmniPath knows; aliases are not counted).

Run (needs network for OmniPath; the BFS universe step takes about 20 s per pair):
    python -m scripts.verify_edge_provenance --pair t2dm_ad
    python -m scripts.verify_edge_provenance --pair t2dm_cad --omnipath-datasets extended
Writes outputs/provenance_audit/<pair>.json and prints a summary.
"""
import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Disease abbreviations that are also valid gene symbols (CAD = carbamoyl-phosphate synthetase); in a rationale they
# almost always mean the disease, so they are not counted as gene mentions.
DISEASE_ABBREVIATIONS = {"CAD", "AD", "T2DM", "T2D"}

_GENE_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,}(?:_[A-Z0-9]+)*(?![a-z0-9])")


# Input: a saved edge dict. Output: its (source, target) pair.
def edge_key(edge: dict) -> Tuple[str, str]:
    return (edge["source_gene"], edge["target_gene"])


# Input: one saved chain edge and {(source, target): interaction_type} from the live source. Output: {"status": "exact"
# | "type_mismatch" | "not_found", "saved_type", "live_type"}.
def check_edge(edge: dict, live_index: Dict[Tuple[str, str], str]) -> dict:
    live_type = live_index.get(edge_key(edge))
    if live_type is None:
        status = "not_found"
    elif live_type == edge["interaction_type"]:
        status = "exact"
    else:
        status = "type_mismatch"
    return {"source_gene": edge["source_gene"], "target_gene": edge["target_gene"],
            "saved_type": edge["interaction_type"], "live_type": live_type, "status": status}


# Input: free text and the set of known gene symbols. Output: the symbols named in the text; only exact symbols in the
# universe count, so acronyms (DNA, ATP, AD) and aliases are ignored.
def extract_gene_mentions(text: Optional[str], universe: Set[str]) -> Set[str]:
    if not text:
        return set()
    return {tok for tok in _GENE_TOKEN.findall(text) if tok in universe and tok not in DISEASE_ABBREVIATIONS}


# Input: a chain dict. Output: the set of its gene symbols.
def chain_genes(chain: dict) -> Set[str]:
    return ({e["source_gene"] for e in chain["edges"] if e.get("source_kind", "gene") == "gene"}
            | {e["target_gene"] for e in chain["edges"] if e.get("target_kind", "gene") == "gene"})


# Input: a chain's genes, the citations by gene and the gene universe. Output: genes named in the titles shown to the
# model for this chain (up to 2 per path gene), so mentioning them is not outside knowledge.
def genes_in_cited_titles(chain_gene_set: Iterable[str], citations: Dict[str, List[dict]], universe: Set[str]) -> Set[str]:
    shown: Set[str] = set()
    for gene in chain_genes_sorted(chain_gene_set):
        for cite in (citations.get(gene) or [])[:2]:
            shown |= extract_gene_mentions(cite.get("title", ""), universe)
    return shown


# Input: gene symbols. Output: them sorted.
def chain_genes_sorted(genes: Iterable[str]) -> List[str]:
    return sorted(genes)


# Input: a rationale, the allowed genes and the gene universe. Output: the genes it mentions and those not allowed (off
# path).
def audit_rationale(text: Optional[str], allowed: Set[str], universe: Set[str]) -> dict:
    mentioned = extract_gene_mentions(text, universe)
    return {"genes_mentioned": sorted(mentioned), "off_path_genes": sorted(mentioned - allowed)}


# Input: the saved stage1 and stage2 dicts, the live edge index and the gene-symbol universe. Output: per-edge checks,
# the per-rationale gene-mention audit and a summary.
def audit_result(stage1: dict, stage2: dict, live_index: Dict[Tuple[str, str], str], universe: Set[str]) -> dict:
    citations = stage1.get("citations", {})
    chains_out = []
    n_edges = n_exact = n_mismatch = n_missing = n_metabolite_unchecked = 0
    n_rationales = n_with_off_path = 0
    for ci, chain in enumerate(stage2.get("chains", [])):
        checks = []
        for e in chain["edges"]:
            if e.get("source_kind", "gene") != "gene" or e.get("target_kind", "gene") != "gene":
                n_metabolite_unchecked += 1
                checks.append({"source_gene": e["source_gene"], "target_gene": e["target_gene"],
                               "saved_type": e["interaction_type"], "live_type": None, "status": "metabolite_not_checked"})
                continue
            c = check_edge(e, live_index)
            checks.append(c)
            n_edges += 1
            n_exact += c["status"] == "exact"
            n_mismatch += c["status"] == "type_mismatch"
            n_missing += c["status"] == "not_found"
        allowed = chain_genes(chain) | genes_in_cited_titles(chain_genes(chain), citations, universe)
        rationale = audit_rationale(chain.get("rationale"), allowed, universe)
        n_rationales += 1
        n_with_off_path += bool(rationale["off_path_genes"])
        chains_out.append({"chain_index": ci, "path": " -> ".join([chain["edges"][0]["source_gene"]] + [e["target_gene"] for e in chain["edges"]]),
                           "edge_checks": checks, "rationale_audit": rationale})
    debate_out = []
    for d in stage2.get("debate_log", []):
        winner, loser = stage2["chains"][d["winner_index"]], stage2["chains"][d["loser_index"]]
        allowed = chain_genes(winner) | chain_genes(loser)
        for chain in (winner, loser):
            allowed |= genes_in_cited_titles(chain_genes(chain), citations, universe)
        a = audit_rationale(d.get("rationale"), allowed, universe)
        debate_out.append(a)
        n_rationales += 1
        n_with_off_path += bool(a["off_path_genes"])
    return {
        "chains": chains_out,
        "debates": debate_out,
        "summary": {
            "n_gene_edges_checked": n_edges, "n_exact": n_exact, "n_type_mismatch": n_mismatch, "n_not_found": n_missing,
            "n_metabolite_edges_not_checked": n_metabolite_unchecked,
            "all_edges_grounded": n_edges > 0 and n_exact == n_edges,
            "n_rationales_audited": n_rationales, "n_rationales_naming_off_path_genes": n_with_off_path,
        },
    }


# Input: saved stage1 and stage2 and the OmniPath dataset choice. Output: ({(source, target): type} for the chain edges'
# source genes, the set of every gene symbol the BFS from Stage 1's lists touches). The universe is built by re-running
# Stage 2's own BFS (no LLM), so it is exactly the graph the pipeline searched.
def build_live_index_and_universe(stage1: dict, stage2: dict, datasets: str) -> Tuple[Dict[Tuple[str, str], str], Set[str]]:
    from PipelineB import stage2_judging as a2
    from PipelineB.omnipath_client import EXTENDED_DATASETS, OmniPathClient

    client = OmniPathClient(datasets=EXTENDED_DATASETS if datasets == "extended" else "omnipath")
    universe: Set[str] = set()
    index: Dict[Tuple[str, str], str] = {}
    chain_sources = {e["source_gene"] for c in stage2.get("chains", []) for e in c["edges"] if e.get("source_kind", "gene") == "gene"}

    # OmniPath client that records every gene symbol it returns into the universe.
    class Spy(OmniPathClient):
        # Input: source genes. Output: the interactions, after adding both ends of each to the universe.
        def get_outgoing_interactions(self, genes):
            out = super().get_outgoing_interactions(genes)
            for i in out:
                universe.add(i.source_gene)
                universe.add(i.target_gene)
            return out

    spy = Spy(datasets=client.datasets)
    d1 = [t["gene_symbol"] for t in stage1["disease1_targets"]]
    d2 = {t["gene_symbol"] for t in stage1["disease2_targets"]}
    universe |= set(d1) | d2
    a2._bfs_shortest_paths(spy, None, d1, d2, max_hops=a2.DEFAULT_MAX_HOPS, max_paths=10 ** 6)
    for i in client.get_outgoing_interactions(sorted(chain_sources)):
        index[(i.source_gene, i.target_gene)] = i.interaction_type
    return index, universe


# Command-line entry point: audit one pair and save the report.
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pair", required=True)
    ap.add_argument("--result", help="override the input json path")
    ap.add_argument("--omnipath-datasets", choices=["core", "extended"], default="core")
    ap.add_argument("--out-dir", default=os.path.join("outputs", "provenance_audit"))
    args = ap.parse_args()

    path = args.result or os.path.join("outputs", f"pipeline_b_{args.pair}.json")
    with open(path) as f:
        saved = json.load(f)
    index, universe = build_live_index_and_universe(saved["stage1"], saved["stage2"], args.omnipath_datasets)
    audit = audit_result(saved["stage1"], saved["stage2"], index, universe)
    audit["source"], audit["omnipath_datasets"] = path, args.omnipath_datasets
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.pair}.json")
    with open(out, "w") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit["summary"], indent=2))
    for c in audit["chains"]:
        bad = [x for x in c["edge_checks"] if x["status"] not in ("exact", "metabolite_not_checked")]
        print(f"chain {c['chain_index']}: {c['path']} -> edges not exact: {bad or 'none'}; off-path genes in rationale: {c['rationale_audit']['off_path_genes'] or 'none'}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
