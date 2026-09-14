"""
Agent 3 -- Independent Confirmation.

Service: none directly -- orchestrates tissue_client.py (GTEx) and
crossdb_client.py (GWAS Catalog), both evidence sources Agent 1 never
looked at during discovery.
Data: per chain, runs (1) a tissue-expression check and (2) a
cross-database gene-disease association check; see those modules for
each service's own data shape. Output is a list of Agent3ConfirmationResult,
one per chain.
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
"""
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from common.config import DISEASE_RELEVANT_TISSUES, DISEASE_PAIRS, get_pair
from PipelineB.agent1_discovery import Agent1DiscoveryResult, run_agent1_discovery
from PipelineB.agent2_chain import Agent2ChainResult, ChainCandidate, run_agent2_chain
from PipelineB.crossdb_client import GwasAssociation, GWASCatalogClient
from PipelineB.metalinks_client import MetalinksClient
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
        }


@dataclass
class Agent3ConfirmationResult:
    tissue_checks: List[TissueExpressionResult]
    endpoint_checks: List[EndpointCrossDbCheck]  # exactly 2: [disease1-endpoint, disease2-endpoint]
    chain_metabolites: List[str]  # metabolite-kind middle hops in this chain; never tissue/cross-db checked

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


def _confirm_chain(
    chain: ChainCandidate,
    disease1: str,
    disease2: str,
    gtex_client,
    gwas_client,
) -> Agent3ConfirmationResult:
    # Input: one chain + both disease names + clients. Output: its Agent3ConfirmationResult
    # (tissue checks for every gene, cross-db checks for the two endpoint genes only;
    # metabolite middle hops are skipped for both and listed in chain_metabolites instead).
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
        gwas_associations = gwas_client.get_gene_disease_associations(gene, disease)
        # getattr with a default: only GWASCatalogClient/StubGWASCatalogClient are known to
        # define this (see crossdb_client.py's bug-fix #3 note); don't hard-fail against some
        # other future client that doesn't.
        possibly_incomplete = getattr(gwas_client, "last_check_rate_limited", False)
        endpoint_checks.append(
            EndpointCrossDbCheck(
                gene_symbol=gene,
                disease_name=disease,
                gwas_associations=gwas_associations,
                possibly_incomplete=possibly_incomplete,
            )
        )

    return Agent3ConfirmationResult(
        tissue_checks=tissue_checks, endpoint_checks=endpoint_checks, chain_metabolites=roles["middle_metabolites"]
    )


def run_agent3_confirmation(
    agent1_result: Agent1DiscoveryResult,
    agent2_result: Agent2ChainResult,
    gtex_client: Optional[object] = None,
    gwas_client: Optional[object] = None,
) -> List[Agent3ConfirmationResult]:
    # Input: Agent 1 + Agent 2 results, optional clients (both default to real clients;
    # pass Stub* clients in tests). Output: one Agent3ConfirmationResult per chain, same
    # order. [] if Agent 2 found no chain.
    if not agent2_result.chains:
        return []

    gtex_client = gtex_client or GTExClient()
    gwas_client = gwas_client or GWASCatalogClient()

    return [
        _confirm_chain(chain, agent1_result.disease1, agent1_result.disease2, gtex_client, gwas_client)
        for chain in agent2_result.chains
    ]


def _cli():
    parser = argparse.ArgumentParser(description="Run Agents 1-3 for one disease pair.")
    parser.add_argument("--pair", choices=[p["id"] for p in DISEASE_PAIRS], help="Disease pair id from common/config.py")
    parser.add_argument("--disease1", help="Override: disease 1 name")
    parser.add_argument("--disease2", help="Override: disease 2 name")
    parser.add_argument("--min-score", type=float, default=0.1, help="Minimum Open Targets association score, passed to Agent 1")
    parser.add_argument("--max-hops", type=int, default=None, help="Max PPI/metabolite hops, passed to Agent 2 (defaults to agent2_chain's own default)")
    parser.add_argument(
        "--no-metabolites", action="store_true", help="Protein-only search -- skip MetalinksDB gene<->metabolite hops in Agent 2"
    )
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
    metalinks_client = None if args.no_metabolites else MetalinksClient()
    agent2_result = run_agent2_chain(agent1_result, metalinks_client=metalinks_client, **agent2_kwargs)

    print("Running Agent 3 (Independent Confirmation)...", flush=True)
    agent3_results = run_agent3_confirmation(agent1_result, agent2_result)

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


if __name__ == "__main__":
    _cli()
