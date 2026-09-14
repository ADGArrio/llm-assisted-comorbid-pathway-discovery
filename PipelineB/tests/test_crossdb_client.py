import json

import pytest

from PipelineB.crossdb_client import (
    _ASSOCIATIONS_BY_GENE_PATH,
    GWASCatalogClient,
    GwasAssociation,
    StubGWASCatalogClient,
    _trait_matches_disease,
)

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


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _efo_trait(trait):
    return {"trait": trait, "shortForm": "EFO_0001360"}


def test_gwas_client_follows_snp_associations_link_when_not_embedded(monkeypatch):
    """Regression test for the 2026-09-14 bug fix (see crossdb_client.py's module docstring):
    findByGene's SNP entries don't embed their associations inline -- get_gene_disease_associations
    must follow each SNP's own '_links.associations.href' and parse that response instead of
    silently returning [] for every gene."""
    search_payload = {
        "_embedded": {
            "singleNucleotidePolymorphisms": [
                {
                    "rsId": "rs7903146",
                    "_links": {"associations": {"href": "https://api.example/snps/rs7903146/associations"}},
                }
            ]
        }
    }
    associations_payload = {
        "_embedded": {
            "associations": [
                {
                    "pvalue": 1e-30,
                    "studyAccession": "GCST0001",
                    "efoTraits": [_efo_trait("Type 2 diabetes mellitus")],
                }
            ]
        }
    }

    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        if url.endswith(_ASSOCIATIONS_BY_GENE_PATH):
            return _FakeResponse(search_payload)
        assert url == "https://api.example/snps/rs7903146/associations"
        return _FakeResponse(associations_payload)

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    client = GWASCatalogClient()

    results = client.get_gene_disease_associations("TCF7L2", "Type 2 Diabetes Mellitus")

    assert len(calls) == 2  # search request + one follow-up per SNP
    assert len(results) == 1
    assert results[0].gene_symbol == "TCF7L2"
    assert results[0].trait_name == "Type 2 diabetes mellitus"
    assert results[0].study_accession == "GCST0001"


def test_gwas_client_uses_embedded_associations_when_present(monkeypatch):
    """If a SNP entry DOES embed its associations inline, use them directly -- no follow-up
    request needed."""
    search_payload = {
        "_embedded": {
            "singleNucleotidePolymorphisms": [
                {
                    "rsId": "rs7903146",
                    "_embedded": {
                        "associations": [
                            {"pvalue": 1e-30, "studyAccession": "GCST0001", "efoTraits": [_efo_trait("Alzheimer's disease")]}
                        ]
                    },
                }
            ]
        }
    }

    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        return _FakeResponse(search_payload)

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    client = GWASCatalogClient()

    results = client.get_gene_disease_associations("APOE", "Alzheimer's Disease")

    assert len(calls) == 1  # no follow-up needed
    assert len(results) == 1
    assert results[0].trait_name == "Alzheimer's disease"


def test_gwas_client_returns_empty_list_when_snp_has_no_associations_link(monkeypatch):
    search_payload = {
        "_embedded": {"singleNucleotidePolymorphisms": [{"rsId": "rs1", "_links": {}}]}
    }

    monkeypatch.setattr(
        "PipelineB.crossdb_client.requests.get", lambda url, params=None, timeout=None, headers=None: _FakeResponse(search_payload)
    )
    client = GWASCatalogClient()

    assert client.get_gene_disease_associations("SOMEGENE", "Type 2 Diabetes Mellitus") == []
