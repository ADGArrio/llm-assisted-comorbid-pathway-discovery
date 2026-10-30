"""Tests for the Open Targets client (PipelineB/opentargets_client.py) and its stub."""

from PipelineB.opentargets_client import GeneAssociation, StubOpenTargetsClient, _pick_uniprot_id


# Pick UniProt id prefers swissprot.
def test_pick_uniprot_id_prefers_swissprot():
    protein_ids = [
        {"id": "TREMBL123", "source": "uniprot_trembl"},
        {"id": "P12345", "source": "uniprot_swissprot"},
    ]
    assert _pick_uniprot_id(protein_ids) == "P12345"


# Pick UniProt id falls back to any UniProt source.
def test_pick_uniprot_id_falls_back_to_any_uniprot_source():
    protein_ids = [{"id": "Q99999", "source": "uniprot_trembl"}]
    assert _pick_uniprot_id(protein_ids) == "Q99999"


# Pick UniProt id returns none when empty.
def test_pick_uniprot_id_returns_none_when_empty():
    assert _pick_uniprot_id([]) is None


# Gene association to dict is JSON serializable.
def test_gene_association_to_dict_is_json_serializable():
    import json
    ga = GeneAssociation(gene_symbol="TCF7L2", ensembl_id="ENSG00000148737", uniprot_id="Q9NQB0", disease_assoc_score=0.71)
    json.dumps(ga.to_dict())


# Stub search disease returns configured EFO id.
def test_stub_search_disease_returns_configured_efo_id():
    client = StubOpenTargetsClient(
        disease_to_efo={"Type 2 Diabetes Mellitus": "EFO_0001360"},
        efo_to_targets={},
    )
    assert client.search_disease("Type 2 Diabetes Mellitus") == "EFO_0001360"
    assert client.search_calls == ["Type 2 Diabetes Mellitus"]


# Stub search disease raises for unknown disease.
def test_stub_search_disease_raises_for_unknown_disease():
    client = StubOpenTargetsClient(disease_to_efo={}, efo_to_targets={})
    try:
        client.search_disease("Not A Real Disease")
        assert False, "expected ValueError"
    except ValueError:
        pass


# Stub get associated targets filters by min score.
def test_stub_get_associated_targets_filters_by_min_score():
    targets = [
        GeneAssociation("HIGH", "ENSG1", "P1", 0.8),
        GeneAssociation("LOW", "ENSG2", "P2", 0.05),
    ]
    client = StubOpenTargetsClient(disease_to_efo={}, efo_to_targets={"EFO_X": targets})
    result = client.get_associated_targets("EFO_X", min_score=0.1)
    symbols = [t.gene_symbol for t in result]
    assert symbols == ["HIGH"]


# Get data version reads the release from the meta query.
def test_get_data_version_reads_the_release_from_the_meta_query(monkeypatch):
    from PipelineB.opentargets_client import OpenTargetsClient

    # Minimal stand-in for a requests.Response.
    class _Resp:
        # Mimic requests.Response.raise_for_status.
        def raise_for_status(self):
            pass

        # Return the canned JSON body.
        def json(self):
            return {"data": {"meta": {"apiVersion": {"x": "26", "y": "9", "z": "0"},
                                      "dataVersion": {"year": "26", "month": "09", "iteration": None}}}}

    monkeypatch.setattr("PipelineB.opentargets_client.requests.post", lambda *a, **k: _Resp())
    assert OpenTargetsClient().get_data_version() == {"api_version": "26.9.0", "data_version": "26.09"}


# Stub client reports a stub version.
def test_stub_client_reports_a_stub_version():
    assert StubOpenTargetsClient({}, {}).get_data_version()["data_version"] == "stub"
