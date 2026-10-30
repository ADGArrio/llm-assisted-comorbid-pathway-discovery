"""Stage 3 Confirmation: independent checks on every Stage 2 chain, reported per check rather than as one pass/fail verdict.
  - Tissue expression (GTEx) for every gene on the chain, against the relevant tissues of both diseases.
  - GWAS Catalog hits for the two endpoint genes (and for each bridge gene, against both diseases).
  - Edge literature (PubMed and PMC full text) for bridge edges only, i.e. edges whose two ends are both middle genes, so
    1- and 2-hop chains have none.
Metabolite middle hops are not checked and are listed in chain_metabolites. A random-gene null baseline gives the hit rates to
compare the chains' bridge genes against. A missing GWAS hit is not evidence against a chain. Output is one
Stage3ConfirmationResult per chain.
"""
import argparse
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from common.config import DISEASE_RELEVANT_TISSUES, DISEASE_PAIRS, get_pair
from PipelineB.stage1_discovery import Stage1DiscoveryResult, run_stage1_discovery
from PipelineB.stage2_judging import Stage2JudgingResult, ChainCandidate, run_stage2_judging
from PipelineB.crossdb_client import GwasAssociation, GWASCatalogClient
from PipelineB.metalinks_client import MetalinksClient
from PipelineB.pubmed_client import EdgeCitation, PubMedClient
from PipelineB.tissue_client import GTExClient, TissueExpressionResult, check_tissue_expression


# GWAS Catalog result for one gene and one disease.
@dataclass
class EndpointCrossDbCheck:
    gene_symbol: str
    disease_name: str
    gwas_associations: List[GwasAssociation]
    # True if the GWAS client gave up on part of this check after exhausting rate-limit retries; then confirmed=False
    # means not fully checked, not checked and negative. Always False for the stub.
    possibly_incomplete: bool = False

    # False for a chain endpoint: Stage 1 chose it by an Open Targets score that already includes genetic evidence, so a
    # hit is consistent with the selection evidence, not independent confirmation. True for a bridge gene.
    independent_of_selection: bool = True

    # Output: whether any GWAS association was found.
    @property
    def confirmed(self) -> bool:
        return bool(self.gwas_associations)

    # Output: the check as a plain dict.
    def to_dict(self) -> dict:
        return {
            "gene_symbol": self.gene_symbol,
            "disease_name": self.disease_name,
            "gwas_associations": [a.to_dict() for a in self.gwas_associations],
            "confirmed": self.confirmed,
            "possibly_incomplete": self.possibly_incomplete,
            "independent_of_selection": self.independent_of_selection,
        }


# Literature-support check for one bridge edge.
@dataclass
class EdgeLiteratureCheck:

    gene_a: str
    gene_b: str
    searched: bool  # Always True today (every bridge edge is searched); kept as a field so a skipped edge can be recorded.
    citations: List[EdgeCitation]

    # Output: True only if at least one citation names both genes together in a checked passage.
    @property
    def supported(self) -> bool:
        return any(c.both_genes_confirmed for c in self.citations)

    # Output: whether at least one citation reached real full text rather than only an abstract.
    @property
    def any_full_text_checked(self) -> bool:
        return any(c.full_text_checked for c in self.citations)

    # Output: the check as a plain dict.
    def to_dict(self) -> dict:
        return {
            "gene_a": self.gene_a,
            "gene_b": self.gene_b,
            "searched": self.searched,
            "citations": [c.to_dict() for c in self.citations],
            "supported": self.supported,
            "any_full_text_checked": self.any_full_text_checked,
        }


# All Stage 3 checks for one chain.
@dataclass
class Stage3ConfirmationResult:
    tissue_checks: List[TissueExpressionResult]
    endpoint_checks: List[EndpointCrossDbCheck]  # exactly 2: [disease1-endpoint, disease2-endpoint]
    chain_metabolites: List[str]  # metabolite-kind middle hops in this chain; never tissue/cross-db checked
    edge_literature_checks: List[EdgeLiteratureCheck] = field(default_factory=list)  # bridge edges only
    bridge_gwas_checks: List[EndpointCrossDbCheck] = field(default_factory=list)  # one per bridge gene x disease (independent of selection)
    null_baseline: Optional[dict] = None  # random-gene baseline + chain comparison; same dict on every chain of a run

    # Output: how many tissue checks found the gene expressed.
    @property
    def tissue_expressed_count(self) -> int:
        return sum(1 for t in self.tissue_checks if t.expressed is True)

    # Output: how many tissue checks returned data.
    @property
    def tissue_checked_count(self) -> int:
        return sum(1 for t in self.tissue_checks if t.expressed is not None)

    # Output: the result as a JSON-serializable dict.
    def to_dict(self) -> dict:
        return {
            "tissue_checks": [t.to_dict() for t in self.tissue_checks],
            "endpoint_checks": [e.to_dict() for e in self.endpoint_checks],
            "tissue_expressed_count": self.tissue_expressed_count,
            "tissue_checked_count": self.tissue_checked_count,
            "tissue_total_count": len(self.tissue_checks),
            "chain_metabolites": self.chain_metabolites,
            "edge_literature_checks": [c.to_dict() for c in self.edge_literature_checks],
            "bridge_gwas_checks": [c.to_dict() for c in self.bridge_gwas_checks],
            "null_baseline": self.null_baseline,
        }


# Split a chain's nodes by role. Input: a chain. Output: {"disease1_endpoint", "disease2_endpoint", "middle" (genes),
# "middle_metabolites"}; endpoints are always genes, and a single-edge chain has no middle nodes.
def _chain_gene_roles(chain: ChainCandidate) -> Dict[str, List[str]]:
    edges = chain.edges
    disease1_endpoint = edges[0].source_gene
    disease2_endpoint = edges[-1].target_gene

    nodes_in_order: List[Tuple[str, str]] = [(edges[0].source_kind, edges[0].source_gene)]
    for edge in edges:
        nodes_in_order.append((edge.target_kind, edge.target_gene))

    seen_genes = set()
    seen_metabolites = set()
    middle_genes: List[str] = []
    middle_metabolites: List[str] = []
    for kind, node_id in nodes_in_order[1:-1]:
        if kind == "metabolite":
            if node_id not in seen_metabolites:
                seen_metabolites.add(node_id)
                middle_metabolites.append(node_id)
        elif node_id not in seen_genes and node_id not in (disease1_endpoint, disease2_endpoint):
            seen_genes.add(node_id)
            middle_genes.append(node_id)

    return {
        "disease1_endpoint": disease1_endpoint,
        "disease2_endpoint": disease2_endpoint,
        "middle": middle_genes,
        "middle_metabolites": middle_metabolites,
    }


# Run the literature check on bridge edges. Input: a chain, its roles and a PubMed client. Output: one
# EdgeLiteratureCheck per edge whose two ends are both gene-kind middle genes.
def _edge_literature_checks(chain: ChainCandidate, roles: Dict[str, List[str]], pubmed_client) -> List[EdgeLiteratureCheck]:
    bridge_genes = set(roles["middle"])
    checks: List[EdgeLiteratureCheck] = []
    for edge in chain.edges:
        if edge.source_kind != "gene" or edge.target_kind != "gene":
            continue  # a metabolite-involved edge has no gene-pair literature search to run
        if edge.source_gene in bridge_genes and edge.target_gene in bridge_genes:
            citations = pubmed_client.search_edge_citations(edge.source_gene, edge.target_gene)
            checks.append(
                EdgeLiteratureCheck(gene_a=edge.source_gene, gene_b=edge.target_gene, searched=True, citations=citations)
            )
    return checks


# Input: a gene, a disease, a GWAS client and whether the gene was independent of Stage 1's selection. Output: its
# EndpointCrossDbCheck (also used for bridge genes).
def _gwas_check(gene: str, disease: str, gwas_client, independent_of_selection: bool) -> EndpointCrossDbCheck:
    associations = gwas_client.get_gene_disease_associations(gene, disease)
    # getattr with a default so a client without this attribute does not fail the check.
    return EndpointCrossDbCheck(
        gene_symbol=gene,
        disease_name=disease,
        gwas_associations=associations,
        possibly_incomplete=getattr(gwas_client, "last_check_rate_limited", False),
        independent_of_selection=independent_of_selection,
    )


# Run all Stage 3 checks on one chain. Input: the chain, both disease names and the GTEx, GWAS and PubMed clients.
# Output: its Stage3ConfirmationResult.
def _confirm_chain(
    chain: ChainCandidate,
    disease1: str,
    disease2: str,
    gtex_client,
    gwas_client,
    pubmed_client,
) -> Stage3ConfirmationResult:
    roles = _chain_gene_roles(chain)
    tissue_checks: List[TissueExpressionResult] = []

    for tissue in DISEASE_RELEVANT_TISSUES.get(disease1, []):
        tissue_checks.append(check_tissue_expression(roles["disease1_endpoint"], tissue, gtex_client))
    for tissue in DISEASE_RELEVANT_TISSUES.get(disease2, []):
        tissue_checks.append(check_tissue_expression(roles["disease2_endpoint"], tissue, gtex_client))
    middle_tissues = DISEASE_RELEVANT_TISSUES.get(disease1, []) + DISEASE_RELEVANT_TISSUES.get(disease2, [])
    for gene in roles["middle"]:
        for tissue in middle_tissues:
            tissue_checks.append(check_tissue_expression(gene, tissue, gtex_client))

    endpoint_checks: List[EndpointCrossDbCheck] = []
    for gene, disease in ((roles["disease1_endpoint"], disease1), (roles["disease2_endpoint"], disease2)):
        endpoint_checks.append(_gwas_check(gene, disease, gwas_client, independent_of_selection=False))

    # Bridge genes (never ranked by Open Targets) are checked against both diseases.
    bridge_gwas_checks: List[EndpointCrossDbCheck] = []
    for gene in roles["middle"]:
        for disease in (disease1, disease2):
            bridge_gwas_checks.append(_gwas_check(gene, disease, gwas_client, independent_of_selection=True))

    edge_literature_checks = _edge_literature_checks(chain, roles, pubmed_client)

    return Stage3ConfirmationResult(
        tissue_checks=tissue_checks,
        endpoint_checks=endpoint_checks,
        chain_metabolites=roles["middle_metabolites"],
        edge_literature_checks=edge_literature_checks,
        bridge_gwas_checks=bridge_gwas_checks,
    )


DEFAULT_NULL_SIZE = 30
NULL_SEED = 20261018


# Input: hits and total. Output: hits / total, or None if the total is zero.
def _rate(hits: int, n: int) -> Optional[float]:
    return (hits / n) if n else None


# Measure how often random genes pass the same checks. Input: the pool of possible bridge genes, genes to exclude, both
# disease names, the GTEx and GWAS clients, sample size and seed. Output: the sampled genes and the hit rates for tissue
# expression and for GWAS hits per disease; checks that gave up after rate limiting are counted separately, not as
# negatives.
def compute_null_baseline(
    pool: List[str],
    exclude: List[str],
    disease1: str,
    disease2: str,
    gtex_client,
    gwas_client,
    size: int = DEFAULT_NULL_SIZE,
    seed: int = NULL_SEED,
) -> dict:
    candidates = sorted(set(pool) - set(exclude))
    sample = random.Random(seed).sample(candidates, min(size, len(candidates)))
    tissues = list(dict.fromkeys(DISEASE_RELEVANT_TISSUES.get(disease1, []) + DISEASE_RELEVANT_TISSUES.get(disease2, [])))

    tissue_with_data = tissue_expressed = 0
    gwas = {disease1: [0, 0], disease2: [0, 0]}  # disease -> [checked, hits]
    incomplete = 0
    for gene in sample:
        for tissue in tissues:
            check = check_tissue_expression(gene, tissue, gtex_client)
            if check.expressed is not None:
                tissue_with_data += 1
                tissue_expressed += int(check.expressed)
        for disease in (disease1, disease2):
            check = _gwas_check(gene, disease, gwas_client, independent_of_selection=True)
            if check.possibly_incomplete and not check.confirmed:
                incomplete += 1
                continue
            gwas[disease][0] += 1
            gwas[disease][1] += int(check.confirmed)
    return {
        "seed": seed,
        "pool_size": len(candidates),
        "n_genes": len(sample),
        "genes": sample,
        "tissues": tissues,
        "tissue_checks_with_data": tissue_with_data,
        "tissue_expressed": tissue_expressed,
        "tissue_expressed_rate": _rate(tissue_expressed, tissue_with_data),
        "gwas": {
            disease: {"genes_checked": n, "genes_with_hit": hits, "hit_rate": _rate(hits, n)}
            for disease, (n, hits) in gwas.items()
        },
        "gwas_checks_inconclusive": incomplete,
    }


# Input: all chains' results and both disease names. Output: the same rates for the chains' distinct bridge genes (a
# gene shared by several chains counts once), to read against the baseline.
def _chain_bridge_summary(results: List[Stage3ConfirmationResult], disease1: str, disease2: str) -> dict:
    bridge_genes = sorted({c.gene_symbol for r in results for c in r.bridge_gwas_checks})
    tissue_by_gene_tissue: Dict[Tuple[str, str], Optional[bool]] = {}
    for r in results:
        for t in r.tissue_checks:
            if t.gene_symbol in bridge_genes:
                tissue_by_gene_tissue[(t.gene_symbol, t.tissue)] = t.expressed
    with_data = [v for v in tissue_by_gene_tissue.values() if v is not None]
    gwas: Dict[str, Dict[str, bool]] = {disease1: {}, disease2: {}}
    incomplete = 0
    for r in results:
        for c in r.bridge_gwas_checks:
            if c.possibly_incomplete and not c.confirmed:
                incomplete += 1
                continue
            gwas[c.disease_name][c.gene_symbol] = c.confirmed
    return {
        "bridge_genes": bridge_genes,
        "tissue_checks_with_data": len(with_data),
        "tissue_expressed": sum(with_data),
        "tissue_expressed_rate": _rate(sum(with_data), len(with_data)),
        "gwas": {
            disease: {"genes_checked": len(found), "genes_with_hit": sum(found.values()),
                      "hit_rate": _rate(sum(found.values()), len(found))}
            for disease, found in gwas.items()
        },
        "gwas_checks_inconclusive": incomplete,
    }


# Run Stage 3. Input: the Stage 1 and Stage 2 results, optional clients (defaults to the real ones; tests pass stubs)
# and the null-baseline size (0 skips it). Output: one Stage3ConfirmationResult per chain in Stage 2's order, empty if
# Stage 2 found no chain.
def run_stage3_confirmation(
    stage1_result: Stage1DiscoveryResult,
    stage2_result: Stage2JudgingResult,
    gtex_client: Optional[object] = None,
    gwas_client: Optional[object] = None,
    pubmed_client: Optional[object] = None,
    null_baseline_size: int = DEFAULT_NULL_SIZE,
) -> List[Stage3ConfirmationResult]:
    if not stage2_result.chains:
        return []

    gtex_client = gtex_client or GTExClient()
    gwas_client = gwas_client or GWASCatalogClient()
    pubmed_client = pubmed_client or PubMedClient()

    results = [
        _confirm_chain(chain, stage1_result.disease1, stage1_result.disease2, gtex_client, gwas_client, pubmed_client)
        for chain in stage2_result.chains
    ]

    pool = (getattr(stage2_result, "bfs_stats", None) or {}).get("candidate_bridge_genes") or []
    if null_baseline_size and pool:
        disease1, disease2 = stage1_result.disease1, stage1_result.disease2
        chain_genes = sorted({t.gene_symbol for r in results for t in r.tissue_checks})
        baseline = compute_null_baseline(pool, chain_genes, disease1, disease2, gtex_client, gwas_client, size=null_baseline_size)
        baseline["chains"] = _chain_bridge_summary(results, disease1, disease2)
        for r in results:
            r.null_baseline = baseline
    return results


# Command-line entry point: run Stages 1-3 for a disease pair and print the Stage 3 results as JSON.
def _cli():
    parser = argparse.ArgumentParser(description="Run Stages 1-3 for one disease pair.")
    parser.add_argument("--pair", choices=[p["id"] for p in DISEASE_PAIRS], help="Disease pair id from common/config.py")
    parser.add_argument("--disease1", help="Override: disease 1 name")
    parser.add_argument("--disease2", help="Override: disease 2 name")
    parser.add_argument("--min-score", type=float, default=0.1, help="Minimum Open Targets association score, passed to Stage 1")
    parser.add_argument("--max-hops", type=int, default=None, help="Max PPI/metabolite hops, passed to Stage 2 (defaults to stage2_judging's own default)")
    parser.add_argument(
        "--with-metabolites", action="store_true", help="Opt in to MetalinksDB gene<->metabolite hops in Stage 2 (default: protein-only)"
    )
    parser.add_argument("--no-metabolites", action="store_true", help="Deprecated no-op: protein-only is now the default")
    parser.add_argument(
        "--no-edge-literature",
        action="store_true",
        help="Skip Stage 3's edge-level literature check on bridge edges (saves PubMed/PMC queries; see module docstring)",
    )
    parser.add_argument("--null-size", type=int, default=DEFAULT_NULL_SIZE, help="Random genes in the baseline (0 = skip it)")
    args = parser.parse_args()

    if args.pair:
        pair = get_pair(args.pair)
        disease1, disease2 = pair["disease1"], pair["disease2"]
    elif args.disease1 and args.disease2:
        disease1, disease2 = args.disease1, args.disease2
    else:
        parser.error("Pass --pair, or both --disease1 and --disease2.")

    print(f"Running Stage 1 (Discovery) for {disease1} / {disease2}...", flush=True)
    stage1_result = run_stage1_discovery(disease1, disease2, min_score=args.min_score)

    print("Running Stage 2 (LLM-based Judging)...", flush=True)
    stage2_kwargs = {"max_hops": args.max_hops} if args.max_hops else {}
    metalinks_client = MetalinksClient() if args.with_metabolites else None
    stage2_result = run_stage2_judging(stage1_result, metalinks_client=metalinks_client, **stage2_kwargs)

    print("Running Stage 3 (Confirmation)...", flush=True)
    if args.no_edge_literature:
        pubmed_client = _NoOpPubMedClient()
    else:
        pubmed_client = PubMedClient()
    stage3_results = run_stage3_confirmation(
        stage1_result, stage2_result, pubmed_client=pubmed_client, null_baseline_size=args.null_size
    )

    print(
        json.dumps(
            {
                "disease1": disease1,
                "disease2": disease2,
                "chains": [
                    {**chain.to_dict(), "confirmation": confirmation.to_dict()}
                    for chain, confirmation in zip(stage2_result.chains, stage3_results)
                ],
            },
            indent=2,
        )
    )


# PubMed stand-in that returns no citations, used by --no-edge-literature.
class _NoOpPubMedClient:

    # Input: two gene symbols and a result limit. Output: an empty list.
    def search_edge_citations(self, gene_a: str, gene_b: str, retmax: int = 3):
        return []


if __name__ == "__main__":
    _cli()
