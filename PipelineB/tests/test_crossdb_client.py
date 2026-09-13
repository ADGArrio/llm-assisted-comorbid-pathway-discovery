import os

import pytest

from PipelineB.crossdb_client import (
    DisgenetAssociation,
    DisGeNETClient,
    GwasAssociation,
    StubDisGeNETClient,
    StubGWASCatalogClient,
    _trait_matches_disease,
)

DISEASE1 = "Type 2 Diabetes Mellitus"
DISEASE2 = "Alzheimer's Disease"


def test_disgenet_client_requires_an_api_key():
    os.environ.pop("DISGENET_API_KEY", None)
    with pytest.raises(RuntimeError, match="API key"):
        DisGeNETClient(api_key=None)


def test_disgenet_client_accepts_an_explicit_api_key():
    client = DisGeNETClient(api_key="fake-key-for-construction-only")
    assert client.api_key == "fake-key-for-construction-only"


def test_stub_gwas_client_returns_configured_associations():
    assoc = GwasAssociation("TCF7L2", "type 2 diabetes mellitus", "EFO_0001360", 1e-20, "GCST0001")
    client = StubGWASCatalogClient({("TCF7L2", DISEASE1): [assoc]})

    result = client.get_gene_disease_associations("TCF7L2", DISEASE1)

    assert result == [assoc]
    assert client.calls == [("TCF7L2", DISEASE1)]


def test_stub_gwas_client_returns_empty_list_for_unconfigured_gene():
    client = StubGWASCatalogClient({})

    assert client.get_gene_disease_associations("UNKNOWNGENE", DISEASE1) == []


def test_stub_disgenet_client_returns_configured_association_or_none():
    assoc = DisgenetAssociation("APP", DISEASE2, 0.8, 120)
    client = StubDisGeNETClient({("APP", DISEASE2): assoc, ("TCF7L2", DISEASE2): None})

    assert client.get_gene_disease_association("APP", DISEASE2) == assoc
    assert client.get_gene_disease_association("TCF7L2", DISEASE2) is None
    assert client.calls == [("APP", DISEASE2), ("TCF7L2", DISEASE2)]


@pytest.mark.parametrize(
    "trait_name,disease_name,expected",
    [
        ("Type 2 diabetes mellitus", DISEASE1, True),
        ("Type II diabetes and related traits", DISEASE1, True),
        ("T2DM nephropathy", DISEASE1, True),
        ("Alzheimer's disease (late onset)", DISEASE2, True),
        ("Body mass index", DISEASE1, False),
        ("Body mass index", DISEASE2, False),
    ],
)
def test_trait_matches_disease_loose_matching(trait_name, disease_name, expected):
    assert _trait_matches_disease(trait_name, disease_name) is expected


def test_to_dict_is_json_serializable():
    import json

    gwas = GwasAssociation("TCF7L2", "type 2 diabetes", "EFO_0001360", 1e-20, "GCST0001")
    disgenet = DisgenetAssociation("APP", DISEASE2, 0.8, 120)
    json.dumps(gwas.to_dict())
    json.dumps(disgenet.to_dict())
