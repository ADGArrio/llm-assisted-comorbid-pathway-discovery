"""Stage 1 Discovery: finds each disease's top Open Targets genes and their PubMed citations, orchestrated with LangGraph.
The two Open Targets lookups run concurrently, then cite_genes searches PubMed for each disease's top genes (a gene ranked
in both diseases is searched once per disease). Output is a Stage1DiscoveryResult: two gene lists, a gene -> citations
dict and one record per PubMed search run.

Usage:
    python -m PipelineB.stage1_discovery --pair t2dm_ad
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


# LangGraph state passed between the Stage 1 nodes.
class Stage1State(TypedDict, total=False):
    disease1: str
    disease2: str
    disease1_efo_id: str
    disease2_efo_id: str
    disease1_targets: List[GeneAssociation]
    disease2_targets: List[GeneAssociation]
    citations: Dict[str, List[Citation]]
    citation_searches: List[dict]


# Stage 1 output: both diseases' EFO ids and gene lists, citations by gene, the PubMed search log and data versions.
@dataclass
class Stage1DiscoveryResult:
    disease1: str
    disease2: str
    disease1_efo_id: str
    disease2_efo_id: str
    disease1_targets: List[GeneAssociation]
    disease2_targets: List[GeneAssociation]
    citations: Dict[str, List[Citation]]
    # One record per PubMed search run (gene, disease, exact query, papers returned and total matches), including
    # zero-hit searches. Defaults to [] for hand-built or older saved results.
    citation_searches: List[dict] = field(default_factory=list)
    # Data releases that produced this result (Open Targets release, PubMed sort order, retrieval time); {} on older
    # saved runs.
    data_versions: dict = field(default_factory=dict)

    # Output: the result as a JSON-serializable dict.
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

    # Rebuild a result from a saved pipeline_b_<pair>.json "stage1" block, so Stages 2-4 can re-run on an earlier Stage
    # 1. Input: that dict. Output: a Stage1DiscoveryResult.
    @classmethod
    def from_dict(cls, d: dict) -> "Stage1DiscoveryResult":
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


# Build the Stage 1 LangGraph. Input: the clients and tuning parameters. Output: a compiled graph.
def _build_graph(
    ot_client,
    pubmed_client,
    min_score: float,
    max_genes_to_cite: int,
    citations_per_gene: int,
):
    # Graph node: resolve disease 1 to an EFO id and fetch its associated genes. Input: graph state. Output: state
    # update.
    def discover_disease1(state: Stage1State) -> dict:
        efo_id = ot_client.search_disease(state["disease1"])  # Experimental Factor Ontology (EFO) id
        # Takes the id of the top-ranked Open Targets hit without checking its name; raises if there is no hit.
        targets = ot_client.get_associated_targets(efo_id, min_score=min_score)
        # Each GeneAssociation holds the gene symbol, Ensembl id, UniProt id and a disease association score (0.0-1.0).
        return {"disease1_efo_id": efo_id, "disease1_targets": targets} # Stage 2 matches proteins by uniprot id.

    # Graph node: the same lookup for disease 2. Input: graph state. Output: state update.
    def discover_disease2(state: Stage1State) -> dict:
        efo_id = ot_client.search_disease(state["disease2"])
        targets = ot_client.get_associated_targets(efo_id, min_score=min_score)
        return {"disease2_efo_id": efo_id, "disease2_targets": targets}

    # Graph node: take each disease's top genes by association score and search PubMed for citations. Input: graph
    # state. Output: the citations by gene and the search log.
    def cite_genes(state: Stage1State) -> dict:
        citations: Dict[str, List[Citation]] = {}
        searches: List[dict] = []
        # Select the top genes by disease association score.

        # Input: a disease's gene list. Output: the highest-scoring genes, up to the citation limit.
        def top_genes(targets: List[GeneAssociation]) -> List[GeneAssociation]:
            return sorted(targets, key=lambda t: t.disease_assoc_score, reverse=True)[:max_genes_to_cite]

        # Every gene in a disease's top-N is searched against that disease, so a gene in both lists is searched twice
        # and carries citations from both searches under one key.
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

    graph = StateGraph(Stage1State)
    graph.add_node("discover_disease1", discover_disease1)
    graph.add_node("discover_disease2", discover_disease2)
    graph.add_node("cite_genes", cite_genes)

    graph.add_edge(START, "discover_disease1")
    graph.add_edge(START, "discover_disease2")
    graph.add_edge("discover_disease1", "cite_genes")
    graph.add_edge("discover_disease2", "cite_genes")
    graph.add_edge("cite_genes", END)

    return graph.compile()


# Run Stage 1. Input: two disease names, optional clients (defaults to the real ones; tests pass stubs) and tuning
# parameters. Output: a Stage1DiscoveryResult.
def run_stage1_discovery(
    disease1: str,
    disease2: str,
    ot_client: Optional[object] = None,
    pubmed_client: Optional[object] = None,
    min_score: float = DEFAULT_MIN_SCORE,
    max_genes_to_cite: int = DEFAULT_MAX_GENES_TO_CITE,
    citations_per_gene: int = DEFAULT_CITATIONS_PER_GENE,
) -> Stage1DiscoveryResult:
    ot_client = ot_client or OpenTargetsClient()
    pubmed_client = pubmed_client or PubMedClient()

    compiled = _build_graph(ot_client, pubmed_client, min_score, max_genes_to_cite, citations_per_gene)
    final_state = compiled.invoke({"disease1": disease1, "disease2": disease2})

    return Stage1DiscoveryResult(
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


# Record which data releases answered. Output: retrieval time, Open Targets version (or the error string if the lookup
# failed) and PubMed sort order.
def _data_versions(ot_client, pubmed_client) -> dict:
    versions: dict = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    getter = getattr(ot_client, "get_data_version", None)
    if getter is not None:
        try:
            versions["open_targets"] = getter()
        except Exception as error:  # noqa: BLE001 -- metadata only
            versions["open_targets"] = {"error": f"{type(error).__name__}: {error}"}
    versions["pubmed_sort"] = getattr(pubmed_client, "sort", None) or "default (PubMed recency order)"
    return versions


# Command-line entry point: run the stage for a disease pair chosen by id or by name and print the result as JSON.
def _cli():
    parser = argparse.ArgumentParser(description="Run Stage 1 (Discovery) for one disease pair.")
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

    result = run_stage1_discovery(disease1, disease2, min_score=args.min_score)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()
