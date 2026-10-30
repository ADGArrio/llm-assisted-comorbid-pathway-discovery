"""Tests for the MetalinksDB client's stub (PipelineB/metalinks_client.py)."""

import json

from PipelineB.metalinks_client import MetaboliteEdge, StubMetalinksClient


# Build a MetaboliteEdge.
def _edge(source_kind, source_id, target_kind, target_id, interaction_type="activates"):
    return MetaboliteEdge(
        source_kind=source_kind,
        source_id=source_id,
        target_kind=target_kind,
        target_id=target_id,
        interaction_type=interaction_type,
        hmdb_id="HMDB0000122",
        uniprot_id="P12345",
        edge_id=f"{source_id}->{target_id}",
    )


# Get gene to metabolite filters by gene and is case insensitive.
def test_get_gene_to_metabolite_filters_by_gene_and_is_case_insensitive():
    edges = [
        _edge("gene", "INSR", "metabolite", "Glucose", "produces"),
        _edge("gene", "APP", "metabolite", "Cholesterol", "degrades"),
    ]
    client = StubMetalinksClient(gene_to_metabolite=edges)

    result = client.get_gene_to_metabolite(["insr"])

    assert len(result) == 1
    assert result[0].target_id == "Glucose"
    assert result[0].interaction_type == "produces"
    assert client.calls == [("gene_to_metabolite", ["INSR"])]


# Get metabolite to gene filters by metabolite and is case insensitive.
def test_get_metabolite_to_gene_filters_by_metabolite_and_is_case_insensitive():
    edges = [
        _edge("metabolite", "Cholesterol", "gene", "LDLR", "activates"),
        _edge("metabolite", "Glucose", "gene", "IL5RA", "unknown"),
    ]
    client = StubMetalinksClient(metabolite_to_gene=edges)

    result = client.get_metabolite_to_gene(["cholesterol"])

    assert len(result) == 1
    assert result[0].target_id == "LDLR"
    assert result[0].interaction_type == "activates"
    assert client.calls == [("metabolite_to_gene", ["cholesterol"])]


# Empty input returns empty.
def test_empty_input_returns_empty():
    client = StubMetalinksClient()
    assert client.get_gene_to_metabolite(["INSR"]) == []
    assert client.get_metabolite_to_gene(["Glucose"]) == []


# To dict is JSON serializable.
def test_to_dict_is_json_serializable():
    edge = _edge("gene", "INSR", "metabolite", "Glucose", "produces")
    json.dumps(edge.to_dict())
