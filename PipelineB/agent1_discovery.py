"""
Agent 1 -- Discovery.

Service: none directly -- orchestrates OpenTargetsClient and PubMedClient
via LangGraph. discover_disease1/discover_disease2 (Open Targets lookups)
fan out from START and run concurrently; cite_genes (PubMed lookups) joins
after both finish, since it needs their gene lists.
Data: see opentargets_client.py / pubmed_client.py for each service's
own data shape. This module's output is Agent1DiscoveryResult: two gene
lists (one per disease), a gene->citations dict (every Citation tagged with
the gene/disease search that found it), and citation_searches (one record per
PubMed search actually run, including searches that returned nothing).
A gene ranked in BOTH diseases' top-N is searched once per disease.
Used by: PipelineB.agent2_chain (candidate gene pools + citations).

Usage:
    python -m PipelineB.agent1_discovery --pair t2dm_ad
"""
import argparse
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from common.config import DISEASE_PAIRS, get_pair 
from PipelineB.opentargets_client import GeneAssociation, OpenTargetsClient
from PipelineB.pubmed_client import Citation, PubMedClient, citation_query_term

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
    citation_searches: List[dict]


@dataclass
class Agent1DiscoveryResult:
    disease1: str
    disease2: str
    disease1_efo_id: str
    disease2_efo_id: str
    disease1_targets: List[GeneAssociation]
    disease2_targets: List[GeneAssociation]
    citations: Dict[str, List[Citation]]
    # One record per PubMed search run: which gene, which disease, the exact query, how many
    # papers came back and how many matched in total. Includes zero-hit searches, which leave
    # no Citation object behind. Defaults to [] so results built by hand or loaded from a run
    # saved before 2026-10-15 still work.
    citation_searches: List[dict] = field(default_factory=list)
    # Which data releases produced this result (Open Targets release, PubMed sort order, retrieval time), added
    # 2026-10-18 so a saved run can be tied to the database versions that answered it. {} on older saved runs.
    data_versions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "disease1": self.disease1,
            "disease2": self.disease2,
            "disease1_efo_id": self.disease1_efo_id,
            "disease2_efo_id": self.disease2_efo_id,
            "disease1_targets": [t.to_dict() for t in self.disease1_targets],
            "disease2_targets": [t.to_dict() for t in self.disease2_targets],
            "citations": {gene: [c.to_dict() for c in cs] for gene, cs in self.citations.items()},
            "citation_searches": self.citation_searches,
            "data_versions": self.data_versions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Agent1DiscoveryResult":
        # Inverse of to_dict(): rebuild a result from a saved pipeline_b_<pair>.json's "agent1" block, so Agents
        # 2-4 can be re-run on exactly the genes and citations of an earlier Agent 1 run (added 2026-10-18).
        fields = set(Citation.__dataclass_fields__)
        return cls(
            disease1=d["disease1"],
            disease2=d["disease2"],
            disease1_efo_id=d["disease1_efo_id"],
            disease2_efo_id=d["disease2_efo_id"],
            disease1_targets=[GeneAssociation(**t) for t in d["disease1_targets"]],
            disease2_targets=[GeneAssociation(**t) for t in d["disease2_targets"]],
            citations={g: [Citation(**{k: v for k, v in c.items() if k in fields}) for c in cs]
                       for g, cs in d["citations"].items()},
            citation_searches=d.get("citation_searches", []),
            data_versions=d.get("data_versions", {}),
        )


def _build_graph(
    ot_client,
    pubmed_client,
    min_score: float,
    max_genes_to_cite: int,
    citations_per_gene: int,
):
    # Input: clients + tuning params. Output: a compiled LangGraph graph (see module docstring for shape).
    def discover_disease1(state: Agent1State) -> dict:
        efo_id = ot_client.search_disease(state["disease1"]) # returns Experimental Factor Ontology (EFO) ID
        # Open Targets’ search actually returns up to 5 hits, each with an ID, a name, and a relevance score. 
        # The function throws away everything except the first hit’s ID, on the assumption that the top hit is the right disease. 
        # It does not check the name. It raises an error if no disease is found.
        targets = ot_client.get_associated_targets(efo_id, min_score=min_score) # returns list of GeneAssociation objects
        # returns a list of GeneAssociation objects, each containing the gene symbol, Ensembl ID, uniprot ID, and disease association score (range: 0.0 - 1.0, where 1.0 is the highest association).
        return {"disease1_efo_id": efo_id, "disease1_targets": targets} # Agent2 matches proteins by uniprot id.

    def discover_disease2(state: Agent1State) -> dict:
        efo_id = ot_client.search_disease(state["disease2"])
        targets = ot_client.get_associated_targets(efo_id, min_score=min_score)
        return {"disease2_efo_id": efo_id, "disease2_targets": targets}

    def cite_genes(state: Agent1State) -> dict: # Score Used: disease association score
        citations: Dict[str, List[Citation]] = {}
        searches: List[dict] = []
        # Select the top genes based on their disease association score.

        def top_genes(targets: List[GeneAssociation]) -> List[GeneAssociation]:
            return sorted(targets, key=lambda t: t.disease_assoc_score, reverse=True)[:max_genes_to_cite]

        # Every gene in a disease's top-N is searched against THAT disease. A gene in both diseases'
        # top-N is therefore searched twice (once per disease) and ends up with citations from both
        # searches under its one key, each Citation tagged with the disease that search used.
        # (Before 2026-10-15 such a gene was skipped on the disease-2 side, so its disease-2
        # literature was never looked up, and disease 2 lost a slot instead of backfilling.)
        for disease_key, efo_key, targets_key in (
            ("disease1", "disease1_efo_id", "disease1_targets"),
            ("disease2", "disease2_efo_id", "disease2_targets"),
        ):
            disease_name = state[disease_key]
            for rank, target in enumerate(top_genes(state[targets_key]), start=1):
                found = pubmed_client.search_citations(target.gene_symbol, disease_name, retmax=citations_per_gene)
                citations.setdefault(target.gene_symbol, []).extend(found)
                searches.append(
                    {
                        "gene_symbol": target.gene_symbol,
                        "disease_name": disease_name,
                        "disease_efo_id": state[efo_key],
                        "gene_rank_in_disease": rank,  # 1 = this disease's highest-scoring gene
                        "disease_assoc_score": target.disease_assoc_score,
                        "query_term": citation_query_term(target.gene_symbol, disease_name),
                        "retmax": citations_per_gene,
                        "n_returned": len(found),
                        "total_hits": found[0].search_total_hits if found else 0,  # 0 = a real zero-hit search
                    }
                )
        return {"citations": citations, "citation_searches": searches}

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
        citation_searches=final_state.get("citation_searches", []),
        data_versions=_data_versions(ot_client, pubmed_client),
    )


def _data_versions(ot_client, pubmed_client) -> dict:
    # Output: {"retrieved_at_utc", "open_targets": {...}, "pubmed_sort"}. Looking up the version must never
    # break a run (it is metadata), so a failure is recorded as an error string instead of raised.
    versions: dict = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    getter = getattr(ot_client, "get_data_version", None)
    if getter is not None:
        try:
            versions["open_targets"] = getter()
        except Exception as error:  # noqa: BLE001 -- metadata only
            versions["open_targets"] = {"error": f"{type(error).__name__}: {error}"}
    versions["pubmed_sort"] = getattr(pubmed_client, "sort", None) or "default (PubMed recency order)"
    return versions


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
