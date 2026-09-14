import json

import pytest

from PipelineB.crossdb_client import GwasAssociation, StubGWASCatalogClient, _trait_matches_disease

DISEASE1 = "Type 2 Diabetes Mellitus"
DISEASE2 = "Alzheimer's Disease"


def test_stub_gwas_client_returns_configured_associations():
    assoc = GwasAssociation("TCF7L2", "type 2 diabetes mellitus", "EFO_0001360", 1e-20, "GCST0001")
    client = StubGWASCatalogClient({("TCF7L2", DISEASE1): [assoc]})

    result = client.get_gene_disease_associations("TCF7L2", DISEASE1)

    assert result == [assoc]
    assert client.calls == [("TCF7L2", DISEASE1)]


def test_stub_gwas_client_returns_empty_list_for_unconfigured_gene():
    client = StubGWASCatalogClient({})

    assert client.get_gene_disease_associations("UNKNOWNGENE", DISEASE1) == []


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
    gwas = GwasAssociation("TCF7L2", "type 2 diabetes", "EFO_0001360", 1e-20, "GCST0001")
    json.dumps(gwas.to_dict())
