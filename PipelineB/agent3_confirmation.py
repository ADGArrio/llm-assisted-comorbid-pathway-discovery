"""
Agent 3 -- Independent Confirmation.

Service: none directly -- orchestrates tissue_client.py (GTEx),
crossdb_client.py (GWAS Catalog), and pubmed_client.py (PubMed, for the
new edge-level literature check below), sources Agent 1 never looked at
during discovery (PubMed aside -- Agent 1 already uses it gene-level;
see pubmed_client.py's module docstring).
Data: per chain, runs (1) a tissue-expression check, (2) a cross-database
gene-disease association check, and (3, new) an edge-level literature
check on bridge edges; see those modules for each service's own data
shape. Output is a list of Agent3ConfirmationResult, one per chain.
Used by: the top-level pipeline, after Agent 2.

Simplified 2026-09-14 to two sources total -- see
claude/agent3_simplification.md. Dropped the Human Protein Atlas
fallback (GTEx alone covers every tissue this project's three diseases
need) and dropped DisGeNET (GWAS Catalog is real numerical study-level
evidence -- p-values, study accessions -- while DisGeNET is itself a
literature-aggregation source, and Agent 1's PubMed citations already
cover literature evidence). Both removed clients are still in git
history if a real run later justifies adding either back.

Which nodes get which check: a chain's first gene (disease1-associated
per Agent 1) and last gene (disease2-associated) get both tissue and
cross-db checks. Every gene in between is a BFS bridge Agent 1 never
claimed was disease-associated, so it only gets tissue-checked, against
the union of both diseases' relevant tissues. Since Agent 2 can route a
chain through a metabolite (metalinks_client.py), a middle hop may be
metabolite-kind rather than gene-kind -- GTEx/GWAS Catalog are both
gene-centric with no meaningful check for a metabolite name, so
metabolite middle hops are excluded from tissue/cross-db checks
entirely and instead just listed in `chain_metabolites` for visibility.
Chain endpoints are always genes by construction (see agent2_chain.py's
module docstring), so endpoint handling is unaffected by this.

Findings are reported per-check, not collapsed into one pass/fail
verdict -- no GWAS Catalog hit is real negative evidence, not proof a
chain is wrong (a gene can matter mechanistically without registering
in a genetics-only database).

Edge-level literature validation (added 2026-09-27, per committee
feedback -- see claude/committee_feedback_round2_coscientist.md section
3a): the checks above validate individual GENES (is this gene expressed
where it needs to be, is this gene independently disease-associated).
None of them validate a specific claimed EDGE -- e.g. whether there is
any real literature support for "PTPN1 inhibits CAPN1" specifically, as
opposed to each gene separately being a reasonable biological player.
`_edge_literature_checks` below adds exactly that, scoped deliberately
to "bridge edges" only -- an edge whose BOTH ends are bridge/middle
genes (`roles["middle"]`), not the chain's overall disease-anchored
endpoints. This scoping was an explicit user decision (2026-09-27): a
chain's two endpoint genes already have independent GWAS Catalog
evidence, so bridge-to-bridge interactions are the part of a chain with
the thinnest independent support today, and that's where a literature
check adds the most value per PubMed/PMC query spent. A real, stated
consequence: a 1-hop chain (a single edge, both ends are endpoints) or
a 2-hop chain (one bridge gene, but every edge still touches an
endpoint) has NO edge with both ends being bridge genes, so it gets
zero literature checks -- an expected scoping outcome, not a bug, same
spirit as this module's other documented scoping choices above. A
citation existing is not automatically support -- see pubmed_client.py's
EdgeCitation docstring; `EdgeLiteratureCheck.supported` only counts a
real co-mentioning passage (full text or abstract), never a bare
title-only match.

Independence of the evidence (revised 2026-10-18). Agent 1 selects each disease's genes by Open Targets' OVERALL
association score, and that score includes a `genetic_association` datatype built partly from GWAS data (checked
live: 140 of the top 200 T2DM genes and 152 of the top 200 CAD genes have genetic_association >= 0.3). A GWAS
Catalog hit on a chain ENDPOINT therefore largely repeats the evidence that selected the endpoint, and is reported
as "consistent with the selection evidence" (`independent_of_selection: false`), not as independent confirmation.
Two additions keep a real, non-circular verification:
  1. BRIDGE genes get the GWAS check too (`bridge_gwas_checks`, both diseases). Bridge genes were found by the
     OmniPath search and never ranked by Open Targets, so a GWAS hit on a bridge gene is evidence the selection did
     not use.
  2. A random-gene baseline (`null_baseline`). Tissue-expression and GWAS hit rates are computed for a seeded random
     sample of genes drawn from the pool of genes the BFS could have used as bridges (Agent 2's
     `bfs_stats["candidate_bridge_genes"]`) and compared with the chains' own bridge genes, so "16/18 expressed" or
     "1 of 3 bridge genes has a GWAS hit" is read against what a comparable random bridge gene would score. The pool
     shares the chains' topology bias (well-connected genes), which a uniformly random gene would not.
"""
import argparse
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from common.config import DISEASE_RELEVANT_TISSUES, DISEASE_PAIRS, get_pair
from PipelineB.agent1_discovery import Agent1DiscoveryResult, run_agent1_discovery
from PipelineB.agent2_chain import Agent2ChainResult, ChainCandidate, run_agent2_chain
from PipelineB.crossdb_client import GwasAssociation, GWASCatalogClient
from PipelineB.metalinks_client import MetalinksClient
from PipelineB.pubmed_client import EdgeCitation, PubMedClient
from PipelineB.tissue_client import GTExClient, TissueExpressionResult, check_tissue_expression


@dataclass
class EndpointCrossDbCheck:
    gene_symbol: str
    disease_name: str
    gwas_associations: List[GwasAssociation]
    # True if crossdb_client.py's GWASCatalogClient gave up on some part of this specific check
    # (the search itself, or a per-SNP follow-up) after exhausting rate-limit retries -- see
    # crossdb_client.py's bug-fix #3 note. When True, `confirmed=False` here means "not fully
    # checked this run", not "checked and found no evidence" -- a real T2DM->AD run hit this
    # for IRS1, a gene with strong, well-replicated GWAS evidence for T2DM, which is how this
    # distinction was discovered to matter. Always False for StubGWASCatalogClient (see there).
    possibly_incomplete: bool = False

    # False for a chain endpoint: Agent 1 picked it by an Open Targets score that already includes GWAS-derived
    # genetic evidence, so a hit is "consistent with the selection evidence", not independent confirmation.
    # True for a bridge gene (never ranked by Open Targets).
    independent_of_selection: bool = True

    @property
    def confirmed(self) -> bool:
        return bool(self.gwas_associations)

    def to_dict(self) -> dict:
        return {
            "gene_symbol": self.gene_symbol,
            "disease_name": self.disease_name,
            "gwas_associations": [a.to_dict() for a in self.gwas_associations],
            "confirmed": self.confirmed,
            "possibly_incomplete": self.possibly_incomplete,
            "independent_of_selection": self.independent_of_selection,
        }


@dataclass
class EdgeLiteratureCheck:
    """One bridge edge's literature-support check -- see module docstring's "Edge-level
    literature validation" note. Only ever built for an edge whose both ends are bridge/middle
    genes; a chain with no such edge (1- or 2-hop chains) simply has an empty
    `edge_literature_checks` list on its Agent3ConfirmationResult."""

    gene_a: str
    gene_b: str
    searched: bool  # always True today (every bridge edge is searched) -- kept as a field, not a
    # constant, so a future reason to skip a specific edge (e.g. a transient PubMed failure) has
    # somewhere honest to record "we meant to check this and couldn't" separately from "there was
    # nothing here to check" (the 1-/2-hop chain case, which never creates this object at all).
    citations: List[EdgeCitation]

    @property
    def supported(self) -> bool:
        # True only if at least one citation actually confirms both genes named together in a
        # checked passage -- a citation merely matching the original gene_a AND gene_b PubMed
        # search is not enough on its own (see pubmed_client.py's EdgeCitation docstring).
        return any(c.both_genes_confirmed for c in self.citations)

    @property
    def any_full_text_checked(self) -> bool:
        # Whether at least one citation's check reached real full text rather than only an
        # abstract -- useful context for how thorough a `supported=False` result actually was.
        return any(c.full_text_checked for c in self.citations)

    def to_dict(self) -> dict:
        return {
            "gene_a": self.gene_a,
            "gene_b": self.gene_b,
            "searched": self.searched,
            "citations": [c.to_dict() for c in self.citations],
            "supported": self.supported,
            "any_full_text_checked": self.any_full_text_checked,
        }


@dataclass
class Agent3ConfirmationResult:
    tissue_checks: List[TissueExpressionResult]
    endpoint_checks: List[EndpointCrossDbCheck]  # exactly 2: [disease1-endpoint, disease2-endpoint]
    chain_metabolites: List[str]  # metabolite-kind middle hops in this chain; never tissue/cross-db checked
    edge_literature_checks: List[EdgeLiteratureCheck] = field(default_factory=list)  # bridge edges only; see module docstring
    bridge_gwas_checks: List[EndpointCrossDbCheck] = field(default_factory=list)  # one per bridge gene x disease (independent of selection)
    null_baseline: Optional[dict] = None  # random-gene baseline + chain comparison; same dict on every chain of a run

    @property
    def tissue_expressed_count(self) -> int:
        return sum(1 for t in self.tissue_checks if t.expressed is True)

    @property
    def tissue_checked_count(self) -> int:
        return sum(1 for t in self.tissue_checks if t.expressed is not None)

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


def _chain_gene_roles(chain: ChainCandidate) -> Dict[str, List[str]]:
    # Input: a chain. Output: {"disease1_endpoint": gene, "disease2_endpoint": gene,
    # "middle": [gene,...], "middle_metabolites": [name,...]}.
    # Chain endpoints are always gene-kind by construction (BFS starts and stops at genes --
    # see agent2_chain.py's module docstring); only middle hops can be metabolite-kind, which
    # are split out separately since gene-centric checks don't apply to them. A single-edge
    # chain has no middle nodes at all -- its one gene pair are both endpoints.
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


def _edge_literature_checks(chain: ChainCandidate, roles: Dict[str, List[str]], pubmed_client) -> List[EdgeLiteratureCheck]:
    # Input: a chain, its already-computed roles (see _chain_gene_roles), a PubMed client.
    # Output: one EdgeLiteratureCheck per bridge edge -- an edge whose both ends are gene-kind
    # AND both are bridge/middle genes (neither is the chain's disease1 or disease2 endpoint).
    # See module docstring's "Edge-level literature validation" note for the scoping rationale
    # and its real consequence for short chains (zero checks is expected, not a bug).
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


def _gwas_check(gene: str, disease: str, gwas_client, independent_of_selection: bool) -> EndpointCrossDbCheck:
    # Input: a gene, a disease, a GWAS client. Output: its EndpointCrossDbCheck (also used for bridge genes).
    associations = gwas_client.get_gene_disease_associations(gene, disease)
    # getattr with a default: only GWASCatalogClient/StubGWASCatalogClient are known to define this (see
    # crossdb_client.py's bug-fix #3 note); don't hard-fail against some other future client that doesn't.
    return EndpointCrossDbCheck(
        gene_symbol=gene,
        disease_name=disease,
        gwas_associations=associations,
        possibly_incomplete=getattr(gwas_client, "last_check_rate_limited", False),
        independent_of_selection=independent_of_selection,
    )


def _confirm_chain(
    chain: ChainCandidate,
    disease1: str,
    disease2: str,
    gtex_client,
    gwas_client,
    pubmed_client,
) -> Agent3ConfirmationResult:
    # Input: one chain + both disease names + clients. Output: its Agent3ConfirmationResult
    # (tissue checks for every gene, cross-db checks for the two endpoint genes only, edge
    # literature checks for bridge edges only; metabolite middle hops are skipped for all three
    # and listed in chain_metabolites instead).
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

    # Bridge genes (never ranked by Open Targets) are checked against BOTH diseases.
    bridge_gwas_checks: List[EndpointCrossDbCheck] = []
    for gene in roles["middle"]:
        for disease in (disease1, disease2):
            bridge_gwas_checks.append(_gwas_check(gene, disease, gwas_client, independent_of_selection=True))

    edge_literature_checks = _edge_literature_checks(chain, roles, pubmed_client)

    return Agent3ConfirmationResult(
        tissue_checks=tissue_checks,
        endpoint_checks=endpoint_checks,
        chain_metabolites=roles["middle_metabolites"],
        edge_literature_checks=edge_literature_checks,
        bridge_gwas_checks=bridge_gwas_checks,
    )


DEFAULT_NULL_SIZE = 30  # ~6.5 s per live GWAS check x 2 diseases -> roughly 6-7 minutes per pair
NULL_SEED = 20261018


def _rate(hits: int, n: int) -> Optional[float]:
    return (hits / n) if n else None


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
    # Input: the pool of genes the BFS could have used as bridges, genes to exclude (the chains' own genes), both
    # disease names, clients, sample size and RNG seed. Output: a dict with the sampled genes and the hit rates a
    # random bridge-like gene gets: tissue expression across the same tissues bridge genes are checked in (both
    # diseases' tissues), and GWAS Catalog hits for each disease. Checks that gave up after rate limiting are left out
    # of the denominators (inconclusive, not negative) and counted separately.
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


def _chain_bridge_summary(results: List[Agent3ConfirmationResult], disease1: str, disease2: str) -> dict:
    # Input: all chains' results. Output: the same rates for the chains' own DISTINCT bridge genes, so they can be read
    # side by side with the baseline (a gene shared by several chains counts once).
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


def run_agent3_confirmation(
    agent1_result: Agent1DiscoveryResult,
    agent2_result: Agent2ChainResult,
    gtex_client: Optional[object] = None,
    gwas_client: Optional[object] = None,
    pubmed_client: Optional[object] = None,
    null_baseline_size: int = DEFAULT_NULL_SIZE,
) -> List[Agent3ConfirmationResult]:
    # Input: Agent 1 + Agent 2 results, optional clients (all default to real clients;
    # pass Stub* clients in tests), and null_baseline_size (random genes for the baseline; 0 = skip it).
    # Output: one Agent3ConfirmationResult per chain, same order. [] if Agent 2 found no chain.
    if not agent2_result.chains:
        return []

    gtex_client = gtex_client or GTExClient()
    gwas_client = gwas_client or GWASCatalogClient()
    pubmed_client = pubmed_client or PubMedClient()

    results = [
        _confirm_chain(chain, agent1_result.disease1, agent1_result.disease2, gtex_client, gwas_client, pubmed_client)
        for chain in agent2_result.chains
    ]

    pool = (getattr(agent2_result, "bfs_stats", None) or {}).get("candidate_bridge_genes") or []
    if null_baseline_size and pool:
        disease1, disease2 = agent1_result.disease1, agent1_result.disease2
        chain_genes = sorted({t.gene_symbol for r in results for t in r.tissue_checks})
        baseline = compute_null_baseline(pool, chain_genes, disease1, disease2, gtex_client, gwas_client, size=null_baseline_size)
        baseline["chains"] = _chain_bridge_summary(results, disease1, disease2)
        for r in results:
            r.null_baseline = baseline
    return results


def _cli():
    parser = argparse.ArgumentParser(description="Run Agents 1-3 for one disease pair.")
    parser.add_argument("--pair", choices=[p["id"] for p in DISEASE_PAIRS], help="Disease pair id from common/config.py")
    parser.add_argument("--disease1", help="Override: disease 1 name")
    parser.add_argument("--disease2", help="Override: disease 2 name")
    parser.add_argument("--min-score", type=float, default=0.1, help="Minimum Open Targets association score, passed to Agent 1")
    parser.add_argument("--max-hops", type=int, default=None, help="Max PPI/metabolite hops, passed to Agent 2 (defaults to agent2_chain's own default)")
    parser.add_argument(
        "--with-metabolites", action="store_true", help="Opt in to MetalinksDB gene<->metabolite hops in Agent 2 (default: protein-only)"
    )
    parser.add_argument("--no-metabolites", action="store_true", help="Deprecated no-op: protein-only is now the default")
    parser.add_argument(
        "--no-edge-literature",
        action="store_true",
        help="Skip Agent 3's edge-level literature check on bridge edges (saves PubMed/PMC queries; see module docstring)",
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

    print(f"Running Agent 1 (Discovery) for {disease1} / {disease2}...", flush=True)
    agent1_result = run_agent1_discovery(disease1, disease2, min_score=args.min_score)

    print("Running Agent 2 (Chain Construction)...", flush=True)
    agent2_kwargs = {"max_hops": args.max_hops} if args.max_hops else {}
    metalinks_client = MetalinksClient() if args.with_metabolites else None
    agent2_result = run_agent2_chain(agent1_result, metalinks_client=metalinks_client, **agent2_kwargs)

    print("Running Agent 3 (Independent Confirmation)...", flush=True)
    if args.no_edge_literature:
        pubmed_client = _NoOpPubMedClient()
    else:
        pubmed_client = PubMedClient()
    agent3_results = run_agent3_confirmation(
        agent1_result, agent2_result, pubmed_client=pubmed_client, null_baseline_size=args.null_size
    )

    print(
        json.dumps(
            {
                "disease1": disease1,
                "disease2": disease2,
                "chains": [
                    {**chain.to_dict(), "confirmation": confirmation.to_dict()}
                    for chain, confirmation in zip(agent2_result.chains, agent3_results)
                ],
            },
            indent=2,
        )
    )


class _NoOpPubMedClient:
    """Used only by `--no-edge-literature` above -- returns no citations for any edge, without
    making a real PubMed request, so the literature-check step is a true no-op rather than a
    slower search that's discarded."""

    def search_edge_citations(self, gene_a: str, gene_b: str, retmax: int = 3):
        return []


if __name__ == "__main__":
    _cli()
