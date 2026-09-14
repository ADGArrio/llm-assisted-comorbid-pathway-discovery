import json

import pytest

from PipelineB.tissue_client import EXPRESSION_THRESHOLD, GTExClient, StubGTExClient, check_tissue_expression


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    # GTExClient._get paces every request and backs off on 429 with real time.sleep() calls
    # (see tissue_client.py's rate-limit-hardening note) -- fine for a real run, but every test
    # in this file would otherwise burn real wall-clock time for no reason. Record calls instead
    # so backoff-timing tests can still assert on them.
    sleeps = []
    monkeypatch.setattr("PipelineB.tissue_client.time.sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


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
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_gtex_client_resolves_gene_symbol_to_gencode_id_before_querying_expression(monkeypatch):
    """Regression test for the 2026-09-14 bug fixes (see tissue_client.py's module docstring):
    medianGeneExpression requires a versioned gencodeId, not a plain gene symbol -- GTExClient
    must resolve the symbol via /reference/gene first, then pass the resolved id -- not the
    symbol itself -- to /expression/medianGeneExpression. And both requests must use GTEx's
    actual (plural) query param names -- geneIds/gencodeIds/tissueSiteDetailIds, not
    geneId/gencodeId/tissueSiteDetailId -- per bug fix #2: the singular names are silently
    ignored by the real API rather than erroring, which is why the first fix alone still came
    back with 0 results on a real run."""
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params))
        if url.endswith("/reference/gene"):
            assert params["geneIds"] == "INSR"
            assert "geneId" not in params
            return _FakeResponse({"data": [{"geneSymbol": "INSR", "gencodeId": "ENSG00000171105.16"}]})
        assert url.endswith("/expression/medianGeneExpression")
        assert params["gencodeIds"] == "ENSG00000171105.16"  # the resolved id, not "INSR"
        assert params["tissueSiteDetailIds"] == "Pancreas"
        assert "gencodeId" not in params and "tissueSiteDetailId" not in params
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
            resolve_calls.append(params["geneIds"])
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


def test_gtex_client_retries_429_and_succeeds(monkeypatch):
    """Regression test for the 2026-09-14 rate-limit hardening (see tissue_client.py's module
    docstring): a single transient 429 must not fail the request -- retry and use the next
    response, exactly like crossdb_client.py's GWASCatalogClient."""
    responses = [
        _FakeResponse({}, status_code=429),
        _FakeResponse({"data": [{"geneSymbol": "INSR", "gencodeId": "ENSG00000171105.16"}]}),
    ]

    def fake_get(url, params, timeout):
        return responses.pop(0)

    monkeypatch.setattr("PipelineB.tissue_client.requests.get", fake_get)
    client = GTExClient()

    assert client._resolve_gencode_id("INSR") == "ENSG00000171105.16"
    assert responses == []


def test_gtex_client_degrades_to_none_after_exhausting_retries(monkeypatch):
    """If GTEx is still rate-limiting after every retry, the whole pipeline must not crash --
    the gene/tissue check is treated as 'no data this time', same as crossdb_client.py's
    equivalent fix (which was itself a response to a real 429 crash)."""
    monkeypatch.setattr(
        "PipelineB.tissue_client.requests.get", lambda url, params, timeout: _FakeResponse({}, status_code=429)
    )
    client = GTExClient()

    assert client.get_median_expression("INSR", "Pancreas") is None


def test_gtex_client_honors_retry_after_header(monkeypatch, _no_real_sleeping):
    responses = [
        _FakeResponse({}, status_code=429, headers={"Retry-After": "5"}),
        _FakeResponse({"data": [{"geneSymbol": "INSR", "gencodeId": "ENSG00000171105.16"}]}),
    ]

    monkeypatch.setattr("PipelineB.tissue_client.requests.get", lambda url, params, timeout: responses.pop(0))
    client = GTExClient()

    client._resolve_gencode_id("INSR")

    assert 5.0 in _no_real_sleeping  # waited the server-specified Retry-After, not the default backoff
