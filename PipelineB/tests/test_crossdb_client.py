import json

import pytest

from PipelineB.crossdb_client import (
    DISEASE_GWAS_TRAITS,
    GWASCatalogClient,
    GwasAssociation,
    StubGWASCatalogClient,
    _MAX_ASSOCIATIONS_RETURNED,
    _trait_matches_disease,
)

DISEASE1 = "Type 2 Diabetes Mellitus"
DISEASE2 = "Alzheimer's Disease"
DISEASE3 = "Coronary Artery Disease"


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    # _get paces every request and backs off on 429 with real time.sleep() calls; record instead.
    sleeps = []
    monkeypatch.setattr("PipelineB.crossdb_client.time.sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


def test_stub_gwas_client_returns_configured_associations():
    assoc = GwasAssociation("TCF7L2", "type 2 diabetes mellitus", "EFO_0001360", 1e-20, "GCST0001")
    client = StubGWASCatalogClient({("TCF7L2", DISEASE1): [assoc]})
    assert client.get_gene_disease_associations("TCF7L2", DISEASE1) == [assoc]
    assert client.calls == [("TCF7L2", DISEASE1)]


def test_stub_gwas_client_returns_empty_list_for_unconfigured_gene():
    assert StubGWASCatalogClient({}).get_gene_disease_associations("UNKNOWNGENE", DISEASE1) == []


@pytest.mark.parametrize(
    "trait_name,disease_name,expected",
    [
        ("Type 2 diabetes mellitus", DISEASE1, True),
        ("type 2 diabetes mellitus", DISEASE1, True),
        ("Type II diabetes and related traits", DISEASE1, True),
        ("T2DM nephropathy", DISEASE1, True),
        ("Alzheimer disease", DISEASE2, True),
        ("Alzheimer's disease (late onset)", DISEASE2, True),
        ("coronary artery disorder", DISEASE3, True),
        ("Coronary artery disease", DISEASE3, True),
        ("cadmium measurement", DISEASE3, False),  # the old bare "cad" substring would have matched this
        ("Body mass index", DISEASE1, False),
        ("Body mass index", DISEASE2, False),
    ],
)
def test_trait_matches_disease(trait_name, disease_name, expected):
    assert _trait_matches_disease(trait_name, disease_name) is expected


def test_cad_trait_name_is_the_catalogs_current_label():
    # Verified live 2026-10-18: "coronary artery disease" returns 0 associations; this name returns 3,913.
    assert DISEASE_GWAS_TRAITS[DISEASE3] == ["coronary artery disorder"]


def test_to_dict_is_json_serializable():
    json.dumps(GwasAssociation("TCF7L2", "type 2 diabetes", "EFO_0001360", 1e-20, "GCST0001", "123").to_dict())


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return self._payload


def _assoc(aid, genes, trait, p=1e-9, acc="GCST1", pubmed=111):
    return {"association_id": aid, "p_value": p, "mapped_genes": genes, "accession_id": acc, "pubmed_id": pubmed,
            "efo_traits": [{"efo_id": "MONDO_0005148", "efo_trait": trait}]}


def _page(assocs, total_pages=1):
    return {"_embedded": {"associations": assocs}, "page": {"totalPages": total_pages}}


def _install(monkeypatch, responder):
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append((url, dict(params or {})))
        return responder(len(calls), params)

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    return calls


def test_query_uses_v2_associations_endpoint_with_gene_and_exact_trait(monkeypatch):
    calls = _install(monkeypatch, lambda n, p: _FakeResponse(_page([_assoc(1, ["INSR"], "type 2 diabetes mellitus")])))
    result = GWASCatalogClient().get_gene_disease_associations("INSR", DISEASE1)
    url, params = calls[0]
    assert url == "https://www.ebi.ac.uk/gwas/rest/api/v2/associations"
    assert params["mapped_gene"] == "INSR" and params["efo_trait"] == "type 2 diabetes mellitus"
    assert len(calls) == 1  # one request, not the v1 chain of per-SNP follow-ups
    assert result[0].study_accession == "GCST1" and result[0].pubmed_id == "111" and result[0].efo_id == "MONDO_0005148"


def test_returns_empty_list_when_no_associations(monkeypatch):
    _install(monkeypatch, lambda n, p: _FakeResponse(_page([])))
    client = GWASCatalogClient()
    assert client.get_gene_disease_associations("FGFR4", DISEASE3) == []
    assert client.last_check_rate_limited is False


def test_association_not_mapped_to_the_gene_is_ignored(monkeypatch):
    _install(monkeypatch, lambda n, p: _FakeResponse(_page([_assoc(1, ["OTHER"], "type 2 diabetes mellitus")])))
    assert GWASCatalogClient().get_gene_disease_associations("INSR", DISEASE1) == []


def test_association_with_a_non_matching_trait_is_ignored(monkeypatch):
    _install(monkeypatch, lambda n, p: _FakeResponse(_page([_assoc(1, ["INSR"], "body mass index")])))
    assert GWASCatalogClient().get_gene_disease_associations("INSR", DISEASE1) == []


def test_results_are_sorted_by_pvalue_deduped_and_capped(monkeypatch):
    assocs = [_assoc(i, ["APOE"], "Alzheimer disease", p=10.0 ** -(i + 1)) for i in range(15)] + [
        _assoc(3, ["APOE"], "Alzheimer disease", p=10.0 ** -4)  # repeat of association 3
    ]
    _install(monkeypatch, lambda n, p: _FakeResponse(_page(assocs)))
    result = GWASCatalogClient().get_gene_disease_associations("APOE", DISEASE2)
    assert len(result) == _MAX_ASSOCIATIONS_RETURNED
    assert [a.pvalue for a in result] == sorted(a.pvalue for a in result)
    assert result[0].pvalue == 10.0 ** -15


def test_walks_further_pages_when_the_catalog_reports_them(monkeypatch):
    pages = {0: _page([_assoc(1, ["APOE"], "Alzheimer disease")], total_pages=2),
             1: _page([_assoc(2, ["APOE"], "Alzheimer disease")], total_pages=2)}
    calls = _install(monkeypatch, lambda n, p: _FakeResponse(pages[p["page"]]))
    assert len(GWASCatalogClient().get_gene_disease_associations("APOE", DISEASE2)) == 2
    assert [c[1]["page"] for c in calls] == [0, 1]


@pytest.mark.parametrize("status", [404, 410])
def test_a_retired_endpoint_raises_instead_of_looking_like_no_evidence(monkeypatch, status):
    # Regression for the 2026-10-18 run: the v1 endpoint returned 410 Gone and must never become [].
    _install(monkeypatch, lambda n, p: _FakeResponse({}, status_code=status))
    with pytest.raises(RuntimeError, match="retired|changed"):
        GWASCatalogClient().get_gene_disease_associations("FGFR4", DISEASE3)


def test_retries_429_and_succeeds(monkeypatch):
    responses = [_FakeResponse({}, status_code=429), _FakeResponse(_page([_assoc(1, ["INSR"], "type 2 diabetes mellitus")]))]
    _install(monkeypatch, lambda n, p: responses.pop(0))
    assert len(GWASCatalogClient().get_gene_disease_associations("INSR", DISEASE1)) == 1
    assert responses == []


def test_degrades_and_flags_after_exhausting_429_retries(monkeypatch):
    _install(monkeypatch, lambda n, p: _FakeResponse({}, status_code=429))
    client = GWASCatalogClient()
    assert client.get_gene_disease_associations("INSR", DISEASE1) == []
    assert client.last_check_rate_limited is True


def test_flag_resets_on_a_clean_later_call(monkeypatch):
    mode = {"limited": True}
    _install(monkeypatch, lambda n, p: _FakeResponse({}, 429) if mode["limited"] else _FakeResponse(_page([])))
    client = GWASCatalogClient()
    client.get_gene_disease_associations("INSR", DISEASE1)
    assert client.last_check_rate_limited is True
    mode["limited"] = False
    client.get_gene_disease_associations("INSR", DISEASE1)
    assert client.last_check_rate_limited is False


def test_honors_retry_after_header(monkeypatch, _no_real_sleeping):
    limited = _FakeResponse({}, status_code=429)
    limited.headers = {"Retry-After": "7"}
    responses = [limited, _FakeResponse(_page([]))]
    _install(monkeypatch, lambda n, p: responses.pop(0))
    GWASCatalogClient().get_gene_disease_associations("INSR", DISEASE1)
    assert 7.0 in _no_real_sleeping


def test_retries_connection_failure_and_succeeds(monkeypatch):
    import requests as requests_module
    state = {"n": 0}

    def fake_get(url, params=None, timeout=None, headers=None):
        state["n"] += 1
        if state["n"] == 1:
            raise requests_module.exceptions.ConnectTimeout("timed out")
        return _FakeResponse(_page([]))

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    assert GWASCatalogClient().get_gene_disease_associations("INSR", DISEASE1) == []
    assert state["n"] == 2


def test_degrades_and_flags_after_exhausting_connection_retries(monkeypatch):
    import requests as requests_module

    def always_times_out(url, params=None, timeout=None, headers=None):
        raise requests_module.exceptions.ConnectTimeout("timed out")

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", always_times_out)
    client = GWASCatalogClient()
    assert client.get_gene_disease_associations("IRS1", DISEASE1) == []
    assert client.last_check_rate_limited is True
