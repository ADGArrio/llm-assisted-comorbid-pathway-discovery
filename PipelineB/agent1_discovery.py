"""
Agent 1 -- Discovery.

Service: none directly -- orchestrates OpenTargetsClient and PubMedClient
via LangGraph. discover_disease1/discover_disease2 (Open Targets lookups)
fan out from START and run concurrently; cite_genes (PubMed lookups) joins
after both finish, since it needs their gene lists.
Data: see opentargets_client.py / pubmed_client.py for each service's
own data shape. This module's output is Agent1DiscoveryResult: two gene
lists (one per disease) plus a gene->citations dict.
Used by: PipelineB.agent2_chain (candidate gene pools + citations).

Usage:
    python -m PipelineB.agent1_discovery --pair t2dm_ad
"""
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from common.config import DISEASE_PAIRS, get_pair 
from PipelineB.opentargets_client import GeneAssociation, OpenTargetsClient
from PipelineB.pubmed_client import Citation, PubMedClient

DEFAULT_MIN_SCORE = 0.1
DEFAULT_MAX_GENES_TO_CITE = 10
DEFAULT_CITATIONS_PER_GENE = 2


class Agent1State(TypedDict, total=False):
    disease1: str
    disease2: str
    disease1_efo_id: str
    disease2_efo_id: str
    disease1_targets: List[GeneAssociation]
    disease2_targets: List[GeneAssociation]
    citations: Dict[str, List[Citation]]


@dataclass
class Agent1DiscoveryResult:
    disease1: str
    disease2: str
    disease1_efo_id: str
    disease2_efo_id: str
    disease1_targets: List[GeneAssociation]
    disease2_targets: List[GeneAssociation]
    citations: Dict[str, List[Citation]]

    def to_dict(self) -> dict:
        return {
            "disease1": self.disease1,
            "disease2": self.disease2,
            "disease1_efo_id": self.disease1_efo_id,
            "disease2_efo_id": self.disease2_efo_id,
            "disease1_targets": [t.to_dict() for t in self.disease1_targets],
            "disease2_targets": [t.to_dict() for t in self.disease2_targets],
            "citations": {gene: [c.to_dict() for c in cs] for gene, cs in self.citations.items()},
        }


def _build_graph(
    ot_client,
    pubmed_client,
    min_score: float,
    max_genes_to_cite: int,
    citations_per_gene: int,
):
    # Input: clients + tuning params. Output: a compiled LangGraph graph (see module docstring for shape).
    def discover_disease1(state: Agent1State) -> dict:
        efo_id = ot_client.search_disease(state["disease1"])
        targets = ot_client.get_associated_targets(efo_id, min_score=min_score)
        return {"disease1_efo_id": efo_id, "disease1_targets": targets}

    def discover_disease2(state: Agent1State) -> dict:
        efo_id = ot_client.search_disease(state["disease2"])
        targets = ot_client.get_associated_targets(efo_id, min_score=min_score)
        return {"disease2_efo_id": efo_id, "disease2_targets": targets}

    def cite_genes(state: Agent1State) -> dict:
        citations: Dict[str, List[Citation]] = {}

        def top_genes(targets: List[GeneAssociation]) -> List[GeneAssociation]:
            return sorted(targets, key=lambda t: t.disease_assoc_score, reverse=True)[:max_genes_to_cite]

        for target in top_genes(state["disease1_targets"]):
            citations[target.gene_symbol] = pubmed_client.search_citations(
                target.gene_symbol, state["disease1"], retmax=citations_per_gene
            )
        for target in top_genes(state["disease2_targets"]):
            if target.gene_symbol in citations:
                continue  # already looked up via disease1's list
            citations[target.gene_symbol] = pubmed_client.search_citations(
                target.gene_symbol, state["disease2"], retmax=citations_per_gene
            )
        return {"citations": citations}

    graph = StateGraph(Agent1State)
    graph.add_node("discover_disease1", discover_disease1)
    graph.add_node("discover_disease2", discover_disease2)
    graph.add_node("cite_genes", cite_genes)

    graph.add_edge(START, "discover_disease1")
    graph.add_edge(START, "discover_disease2")
    graph.add_edge("discover_disease1", "cite_genes")
    graph.add_edge("discover_disease2", "cite_genes")
    graph.add_edge("cite_genes", END)

    return graph.compile()


def run_agent1_discovery(
    disease1: str,
    disease2: str,
    ot_client: Optional[object] = None,
    pubmed_client: Optional[object] = None,
    min_score: float = DEFAULT_MIN_SCORE,
    max_genes_to_cite: int = DEFAULT_MAX_GENES_TO_CITE,
    citations_per_gene: int = DEFAULT_CITATIONS_PER_GENE,
) -> Agent1DiscoveryResult:
    # Input: two disease names, optional clients (default to real OpenTargetsClient/PubMedClient;
    # pass Stub* clients in tests), tuning params. Output: Agent1DiscoveryResult.
    ot_client = ot_client or OpenTargetsClient()
    pubmed_client = pubmed_client or PubMedClient()

    compiled = _build_graph(ot_client, pubmed_client, min_score, max_genes_to_cite, citations_per_gene)
    final_state = compiled.invoke({"disease1": disease1, "disease2": disease2})

    return Agent1DiscoveryResult(
        disease1=disease1,
        disease2=disease2,
        disease1_efo_id=final_state["disease1_efo_id"],
        disease2_efo_id=final_state["disease2_efo_id"],
        disease1_targets=final_state["disease1_targets"],
        disease2_targets=final_state["disease2_targets"],
        citations=final_state["citations"],
    )


def _cli():
    parser = argparse.ArgumentParser(description="Run Agent 1 (Discovery) for one disease pair.")
    parser.add_argument("--pair", choices=[p["id"] for p in DISEASE_PAIRS], help="Disease pair id from common/config.py")
    parser.add_argument("--disease1", help="Override: disease 1 name")
    parser.add_argument("--disease2", help="Override: disease 2 name")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="Minimum Open Targets association score")
    args = parser.parse_args()

    if args.pair:
        pair = get_pair(args.pair)
        disease1, disease2 = pair["disease1"], pair["disease2"]
    elif args.disease1 and args.disease2:
        disease1, disease2 = args.disease1, args.disease2
    else:
        parser.error("Pass --pair, or both --disease1 and --disease2.")

    result = run_agent1_discovery(disease1, disease2, min_score=args.min_score)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()
