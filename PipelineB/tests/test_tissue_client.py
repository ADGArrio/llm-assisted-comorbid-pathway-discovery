"""Tests for the GTEx client (PipelineB/tissue_client.py) with a stubbed HTTP layer."""

import json

import pytest

from PipelineB.tissue_client import EXPRESSION_THRESHOLD, GTExClient, StubGTExClient, check_tissue_expression


# Autouse fixture: record time.sleep calls instead of sleeping, so pacing and backoff tests run instantly.
@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    # Record sleeps instead of waiting, so backoff-timing tests can still assert on them.
    sleeps = []
    monkeypatch.setattr("PipelineB.tissue_client.time.sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


# GTEx expression above threshold marks expressed.
def test_gtex_expression_above_threshold_marks_expressed():
    gtex = StubGTExClient({"INSR|Pancreas": EXPRESSION_THRESHOLD + 5.0})

    result = check_tissue_expression("INSR", "Pancreas", gtex)

    assert result.source == "GTEx"
    assert result.expressed is True
    assert result.value == EXPRESSION_THRESHOLD + 5.0
    assert gtex.calls == [("INSR", "Pancreas")]


# GTEx expression below threshold marks not expressed.
def test_gtex_expression_below_threshold_marks_not_expressed():
    gtex = StubGTExClient({"APOE|Brain_Cortex": 0.05})

    result = check_tissue_expression("APOE", "Brain_Cortex", gtex)

    assert result.source == "GTEx"
    assert result.expressed is False
    assert result.value == 0.05


# No GTEx data reports no data.
def test_no_gtex_data_reports_no_data():
    gtex = StubGTExClient({})

    result = check_tissue_expression("MID", "Liver", gtex)

    assert result.source == "none"
    assert result.expressed is None
    assert result.value is None
    assert "No GTEx record" in result.note


# To dict is JSON serializable.
def test_to_dict_is_json_serializable():
    gtex = StubGTExClient({"INSR|Pancreas": 5.0})
    result = check_tissue_expression("INSR", "Pancreas", gtex)
    json.dumps(result.to_dict())


# Minimal stand-in for a requests.Response.
class _FakeResponse:
    # Store the canned data.
    def __init__(self, payload, status_code=200, headers=None, text=""):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    # Mimic requests.Response.raise_for_status.
    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)

    # Return the canned JSON body.
    def json(self):
        return self._payload


# GTEx client resolves gene symbol to gencode id before querying expression.
def test_gtex_client_resolves_gene_symbol_to_gencode_id_before_querying_expression(monkeypatch):
    calls = []

    # Stand-in for the real GET request, returning canned data.
    def fake_get(url, params, timeout):
        calls.append((url, params))
        if url.endswith("/reference/gene"):
            assert params["geneId"] == "INSR"
            assert "geneIds" not in params
            return _FakeResponse({"data": [{"geneSymbol": "INSR", "gencodeId": "ENSG00000171105.16"}]})
        assert url.endswith("/expression/medianGeneExpression")
        assert params["gencodeId"] == "ENSG00000171105.16"  # the resolved id, not "INSR"
        assert params["tissueSiteDetailId"] == "Pancreas"
        assert "gencodeIds" not in params and "tissueSiteDetailIds" not in params
        return _FakeResponse({"data": [{"median": 42.0}]})

    monkeypatch.setattr("PipelineB.tissue_client.requests.get", fake_get)
    client = GTExClient()

    value = client.get_median_expression("INSR", "Pancreas")

    assert value == 42.0
    assert len(calls) == 2


# GTEx client caches gencode id resolution across tissues.
def test_gtex_client_caches_gencode_id_resolution_across_tissues(monkeypatch):
    resolve_calls = []

    # Stand-in for the real GET request, returning canned data.
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


# GTEx client defaults to GTEx v8 dataset.
def test_gtex_client_defaults_to_gtex_v8_dataset(monkeypatch):
    seen_dataset_ids = []

    # Stand-in for the real GET request, returning canned data.
    def fake_get(url, params, timeout):
        if url.endswith("/reference/gene"):
            return _FakeResponse({"data": [{"geneSymbol": "IRS1", "gencodeId": "ENSG00000169213.12"}]})
        seen_dataset_ids.append(params["datasetId"])
        return _FakeResponse({"data": [{"median": 2.26513}]})

    monkeypatch.setattr("PipelineB.tissue_client.requests.get", fake_get)
    client = GTExClient()

    assert client.dataset_id == "gtex_v8"
    client.get_median_expression("IRS1", "Pancreas")
    assert seen_dataset_ids == ["gtex_v8"]


# GTEx client returns none when gene symbol does not resolve.
def test_gtex_client_returns_none_when_gene_symbol_does_not_resolve(monkeypatch):
    # Stand-in for the real GET request, returning canned data.
    def fake_get(url, params, timeout):
        assert url.endswith("/reference/gene")
        return _FakeResponse({"data": []})

    monkeypatch.setattr("PipelineB.tissue_client.requests.get", fake_get)
    client = GTExClient()

    assert client.get_median_expression("NOTAREALGENE", "Pancreas") is None


# GTEx client retries 429 and succeeds.
def test_gtex_client_retries_429_and_succeeds(monkeypatch):
    responses = [
        _FakeResponse({}, status_code=429),
        _FakeResponse({"data": [{"geneSymbol": "INSR", "gencodeId": "ENSG00000171105.16"}]}),
    ]

    # Stand-in for the real GET request, returning canned data.
    def fake_get(url, params, timeout):
        return responses.pop(0)

    monkeypatch.setattr("PipelineB.tissue_client.requests.get", fake_get)
    client = GTExClient()

    assert client._resolve_gencode_id("INSR") == "ENSG00000171105.16"
    assert responses == []


# GTEx client degrades to none after exhausting retries.
def test_gtex_client_degrades_to_none_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(
        "PipelineB.tissue_client.requests.get", lambda url, params, timeout: _FakeResponse({}, status_code=429)
    )
    client = GTExClient()

    assert client.get_median_expression("INSR", "Pancreas") is None


# GTEx client honors retry after header.
def test_gtex_client_honors_retry_after_header(monkeypatch, _no_real_sleeping):
    responses = [
        _FakeResponse({}, status_code=429, headers={"Retry-After": "5"}),
        _FakeResponse({"data": [{"geneSymbol": "INSR", "gencodeId": "ENSG00000171105.16"}]}),
    ]

    monkeypatch.setattr("PipelineB.tissue_client.requests.get", lambda url, params, timeout: responses.pop(0))
    client = GTExClient()

    client._resolve_gencode_id("INSR")

    assert 5.0 in _no_real_sleeping  # waited the server-specified Retry-After, not the default backoff


# GTEx client degrades gracefully on non 429 HTTP error.
def test_gtex_client_degrades_gracefully_on_non_429_http_error(monkeypatch, capsys):
    monkeypatch.setattr(
        "PipelineB.tissue_client.requests.get",
        lambda url, params, timeout: _FakeResponse({}, status_code=422, text='{"detail":[{"loc":["query","geneIds"],"msg":"field required"}]}'),
    )
    client = GTExClient()

    result = client._resolve_gencode_id("IRS1")

    assert result is None
    printed = capsys.readouterr().out
    assert "422" in printed
    assert "geneIds" in printed  # the response body made it into the printed warning


# GTEx client get median expression degrades gracefully on non 429 HTTP error.
def test_gtex_client_get_median_expression_degrades_gracefully_on_non_429_http_error(monkeypatch):

    # Stand-in for the real GET request, returning canned data.
    def fake_get(url, params, timeout):
        if url.endswith("/reference/gene"):
            return _FakeResponse({"data": [{"geneSymbol": "IRS1", "gencodeId": "ENSG00000169213.12"}]})
        return _FakeResponse({}, status_code=500, text="Internal Server Error")

    monkeypatch.setattr("PipelineB.tissue_client.requests.get", fake_get)
    client = GTExClient()

    assert client.get_median_expression("IRS1", "Pancreas") is None
