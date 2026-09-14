import json

from PipelineB.tissue_client import EXPRESSION_THRESHOLD, GTExClient, StubGTExClient, check_tissue_expression


def test_gtex_expression_above_threshold_marks_expressed():
    gtex = StubGTExClient({"INSR|Pancreas": EXPRESSION_THRESHOLD + 5.0})

    result = check_tissue_expression("INSR", "Pancreas", gtex)

    assert result.source == "GTEx"
    assert result.expressed is True
    assert result.value == EXPRESSION_THRESHOLD + 5.0
    assert gtex.calls == [("INSR", "Pancreas")]


def test_gtex_expression_below_threshold_marks_not_expressed():
    gtex = StubGTExClient({"APOE|Brain_Cortex": 0.05})

    result = check_tissue_expression("APOE", "Brain_Cortex", gtex)

    assert result.source == "GTEx"
    assert result.expressed is False
    assert result.value == 0.05


def test_no_gtex_data_reports_no_data():
    gtex = StubGTExClient({})

    result = check_tissue_expression("MID", "Liver", gtex)

    assert result.source == "none"
    assert result.expressed is None
    assert result.value is None
    assert "No GTEx record" in result.note


def test_to_dict_is_json_serializable():
    gtex = StubGTExClient({"INSR|Pancreas": 5.0})
    result = check_tissue_expression("INSR", "Pancreas", gtex)
    json.dumps(result.to_dict())


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_gtex_client_resolves_gene_symbol_to_gencode_id_before_querying_expression(monkeypatch):
    """Regression test for the 2026-09-14 bug fix (see tissue_client.py's module docstring):
    medianGeneExpression requires a versioned gencodeId, not a plain gene symbol. GTExClient
    must resolve the symbol via /reference/gene first, then pass the resolved id -- not the
    symbol itself -- to /expression/medianGeneExpression."""
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params))
        if url.endswith("/reference/gene"):
            assert params["geneId"] == "INSR"
            return _FakeResponse({"data": [{"geneSymbol": "INSR", "gencodeId": "ENSG00000171105.16"}]})
        assert url.endswith("/expression/medianGeneExpression")
        assert params["gencodeId"] == "ENSG00000171105.16"  # the resolved id, not "INSR"
        return _FakeResponse({"data": [{"median": 42.0}]})

    monkeypatch.setattr("PipelineB.tissue_client.requests.get", fake_get)
    client = GTExClient()

    value = client.get_median_expression("INSR", "Pancreas")

    assert value == 42.0
    assert len(calls) == 2


def test_gtex_client_caches_gencode_id_resolution_across_tissues(monkeypatch):
    """The same gene is looked up once per relevant tissue (agent3_confirmation.py) -- resolving
    its gencodeId should only cost one /reference/gene request, not one per tissue."""
    resolve_calls = []

    def fake_get(url, params, timeout):
        if url.endswith("/reference/gene"):
            resolve_calls.append(params["geneId"])
            return _FakeResponse({"data": [{"geneSymbol": "IRS1", "gencodeId": "ENSG00000169213.12"}]})
        return _FakeResponse({"data": [{"median": 7.0}]})

    monkeypatch.setattr("PipelineB.tissue_client.requests.get", fake_get)
    client = GTExClient()

    client.get_median_expression("IRS1", "Pancreas")
    client.get_median_expression("IRS1", "Liver")

    assert resolve_calls == ["IRS1"]  # resolved once, reused for the second tissue


def test_gtex_client_returns_none_when_gene_symbol_does_not_resolve(monkeypatch):
    def fake_get(url, params, timeout):
        assert url.endswith("/reference/gene")
        return _FakeResponse({"data": []})

    monkeypatch.setattr("PipelineB.tissue_client.requests.get", fake_get)
    client = GTExClient()

    assert client.get_median_expression("NOTAREALGENE", "Pancreas") is None
