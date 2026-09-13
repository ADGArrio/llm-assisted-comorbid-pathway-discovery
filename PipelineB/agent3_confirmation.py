"""
Agent 3 -- Independent Confirmation.

Service: none directly -- orchestrates tissue_client.py (GTEx/HPA) and
crossdb_client.py (GWAS Catalog/DisGeNET), both evidence sources Agent 1
never looked at during discovery.
Data: per chain, runs (1) tissue-expression checks and (2) cross-database
gene-disease association checks; see those modules for each service's
data shape. Output is a list of Agent3ConfirmationResult, one per chain.
Used by: the top-level pipeline, after Agent 2.

Which genes get which check: a chain's first gene (disease1-associated
per Agent 1) and last gene (disease2-associated) get both tissue and
cross-db checks. Every gene in between is a BFS bridge Agent 1 never
claimed was disease-associated, so it only gets tissue-checked, against
the union of both diseases' relevant tissues.

Findings are reported per-check, not collapsed into one pass/fail
verdict -- non-expression or no cross-db hit is real negative evidence,
not proof a chain is wrong (a gene can matter mechanistically without
registering in a genetics-only database).
"""
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from common.config import DISEASE_RELEVANT_TISSUES, DISEASE_PAIRS, get_pair
from PipelineB.agent1_discovery import Agent1DiscoveryResult, run_agent1_discovery
from PipelineB.agent2_chain import Agent2ChainResult, ChainCandidate, run_agent2_chain
from PipelineB.crossdb_client import (
    DisgenetAssociation,
    DisGeNETClient,
    GwasAssociation,
    GWASCatalogClient,
)
from PipelineB.tissue_client import GTExClient, HPAClient, TissueExpressionResult, check_tissue_expression


@dataclass
class EndpointCrossDbCheck:
    gene_symbol: str
    disease_name: str
    gwas_associations: List[GwasAssociation]
    disgenet_association: Optional[DisgenetAssociation]

    @property
    def confirmed(self) -> bool:
        # OR, not AND: GWAS Catalog and DisGeNET have different coverage, so requiring
        # both would understate confirmation for genes well-covered by only one.
        return bool(self.gwas_associations) or self.disgenet_association is not None

    def to_dict(self) -> dict:
        return {
            "gene_symbol": self.gene_symbol,
            "disease_name": self.disease_name,
            "gwas_associations": [a.to_dict() for a in self.gwas_associations],
            "disgenet_association": self.disgenet_association.to_dict() if self.disgenet_association else None,
            "confirmed": self.confirmed,
        }


@dataclass
class Agent3ConfirmationResult:
    tissue_checks: List[TissueExpressionResult]
    endpoint_checks: List[EndpointCrossDbCheck]  # exactly 2: [disease1-endpoint, disease2-endpoint]

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
        }


def _ensembl_id_lookup(agent1_result: Agent1DiscoveryResult) -> Dict[str, str]:
    # Input: Agent 1's result. Output: gene_symbol -> ensembl_id, from both diseases' target lists.
    lookup: Dict[str, str] = {}
    for target in agent1_result.disease1_targets + agent1_result.disease2_targets:
        lookup[target.gene_symbol] = target.ensembl_id
    return lookup


def _chain_gene_roles(chain: ChainCandidate) -> Dict[str, List[str]]:
    # Input: a chain. Output: {"disease1_endpoint": gene, "disease2_endpoint": gene, "middle": [genes]}.
    # A single-edge chain has no middle genes -- its one gene pair are both endpoints.
    edges = chain.edges
    disease1_endpoint = edges[0].source_gene
    disease2_endpoint = edges[-1].target_gene

    all_genes_in_order: List[str] = [edges[0].source_gene]
    for edge in edges:
        all_genes_in_order.append(edge.target_gene)

    middle = [g for g in all_genes_in_order[1:-1]]
    seen = set()
    deduped_middle = []
    for g in middle:
        if g not in seen and g not in (disease1_endpoint, disease2_endpoint):
            seen.add(g)
            deduped_middle.append(g)

    return {"disease1_endpoint": disease1_endpoint, "disease2_endpoint": disease2_endpoint, "middle": deduped_middle}


def _confirm_chain(
    chain: ChainCandidate,
    disease1: str,
    disease2: str,
    ensembl_lookup: Dict[str, str],
    gtex_client,
    hpa_client,
    gwas_client,
    disgenet_client,
) -> Agent3ConfirmationResult:
    # Input: one chain + both disease names + clients. Output: its Agent3ConfirmationResult
    # (tissue checks for every gene, cross-db checks for the two endpoint genes only).
    roles = _chain_gene_roles(chain)
    tissue_checks: List[TissueExpressionResult] = []

    for tissue in DISEASE_RELEVANT_TISSUES.get(disease1, []):
        tissue_checks.append(
            check_tissue_expression(
                roles["disease1_endpoint"], tissue, gtex_client, hpa_client, ensembl_lookup.get(roles["disease1_endpoint"])
            )
        )
    for tissue in DISEASE_RELEVANT_TISSUES.get(disease2, []):
        tissue_checks.append(
            check_tissue_expression(
                roles["disease2_endpoint"], tissue, gtex_client, hpa_client, ensembl_lookup.get(roles["disease2_endpoint"])
            )
        )
    middle_tissues = DISEASE_RELEVANT_TISSUES.get(disease1, []) + DISEASE_RELEVANT_TISSUES.get(disease2, [])
    for gene in roles["middle"]:
        for tissue in middle_tissues:
            tissue_checks.append(check_tissue_expression(gene, tissue, gtex_client, hpa_client, ensembl_lookup.get(gene)))

    endpoint_checks: List[EndpointCrossDbCheck] = []
    for gene, disease in ((roles["disease1_endpoint"], disease1), (roles["disease2_endpoint"], disease2)):
        gwas_associations = gwas_client.get_gene_disease_associations(gene, disease)
        disgenet_association = disgenet_client.get_gene_disease_association(gene, disease) if disgenet_client else None
        endpoint_checks.append(
            EndpointCrossDbCheck(
                gene_symbol=gene,
                disease_name=disease,
                gwas_associations=gwas_associations,
                disgenet_association=disgenet_association,
            )
        )

    return Agent3ConfirmationResult(tissue_checks=tissue_checks, endpoint_checks=endpoint_checks)


def run_agent3_confirmation(
    agent1_result: Agent1DiscoveryResult,
    agent2_result: Agent2ChainResult,
    gtex_client: Optional[object] = None,
    hpa_client: Optional[object] = None,
    gwas_client: Optional[object] = None,
    disgenet_client: Optional[object] = None,
) -> List[Agent3ConfirmationResult]:
    # Input: Agent 1 + Agent 2 results, optional clients (gtex/hpa/gwas default to real
    # clients; pass None for disgenet_client to skip it entirely; pass Stub* in tests).
    # Output: one Agent3ConfirmationResult per chain, same order. [] if Agent 2 found no chain.
    if not agent2_result.chains:
        return []

    gtex_client = gtex_client or GTExClient()
    hpa_client = hpa_client or HPAClient()
    gwas_client = gwas_client or GWASCatalogClient()
    ensembl_lookup = _ensembl_id_lookup(agent1_result)

    return [
        _confirm_chain(
            chain, agent1_result.disease1, agent1_result.disease2, ensembl_lookup, gtex_client, hpa_client, gwas_client, disgenet_client
        )
        for chain in agent2_result.chains
    ]


def _cli():
    parser = argparse.ArgumentParser(description="Run Agents 1-3 for one disease pair.")
    parser.add_argument("--pair", choices=[p["id"] for p in DISEASE_PAIRS], help="Disease pair id from common/config.py")
    parser.add_argument("--disease1", help="Override: disease 1 name")
    parser.add_argument("--disease2", help="Override: disease 2 name")
    parser.add_argument("--min-score", type=float, default=0.1, help="Minimum Open Targets association score, passed to Agent 1")
    parser.add_argument("--max-hops", type=int, default=None, help="Max PPI hops, passed to Agent 2 (defaults to agent2_chain's own default)")
    parser.add_argument("--use-disgenet", action="store_true", help="Also query DisGeNET (requires DISGENET_API_KEY in .env)")
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
    agent2_result = run_agent2_chain(agent1_result, **agent2_kwargs)

    print("Running Agent 3 (Independent Confirmation)...", flush=True)
    disgenet_client = DisGeNETClient() if args.use_disgenet else None
    agent3_results = run_agent3_confirmation(agent1_result, agent2_result, disgenet_client=disgenet_client)

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
