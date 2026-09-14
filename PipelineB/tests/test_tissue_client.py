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
    def __init__(self, payload, status_code=200, headers=None, text=""):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)

    def json(self):
        return self._payload


def test_gtex_client_resolves_gene_symbol_to_gencode_id_before_querying_expression(monkeypatch):
    """Regression test for the 2026-09-14 bug fixes (see tissue_client.py's module docstring):
    medianGeneExpression requires a versioned gencodeId, not a plain gene symbol -- GTExClient
    must resolve the symbol via /reference/gene first, then pass the resolved id -- not the
    symbol itself -- to /expression/medianGeneExpression. Both requests must use GTEx's actual
    (singular) query param names -- geneId/gencodeId/tissueSiteDetailId -- per bug-fix #4: a
    live 422 confirmed the plural forms (geneIds/gencodeIds/tissueSiteDetailIds, briefly used in
    bug-fix #2 based on a misreading of a third-party R client's docs) are rejected by the real
    API with "field required: geneId", and the singular forms succeed."""
    calls = []

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


def test_gtex_client_defaults_to_gtex_v8_dataset(monkeypatch):
    """Regression test for bug-fix #4 (see tissue_client.py's module docstring): datasetId
    "gtex_v10" silently returns 200 {"data": []} for every query on the live API -- confirmed via
    a live diagnostic -- which was the actual root cause of this module's persistent 0/18 bug
    across three prior "fixes" that were all chasing the wrong thing (param name casing/plurality).
    "gtex_v8" is confirmed to return real data for the same query, so it must stay the default;
    this test exists so a future edit can't silently regress back to "gtex_v10"."""
    seen_dataset_ids = []

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


def test_gtex_client_degrades_gracefully_on_non_429_http_error(monkeypatch, capsys):
    """Regression test for bug-fix #3 (see tissue_client.py's module docstring): a real run hit
    a genuine 422 from /reference/gene. A broken request for one gene must not crash the whole
    pipeline -- it should degrade to 'unresolved this run' and print the response body (so the
    failure is diagnosable from a normal run's console output), not raise."""
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


def test_gtex_client_get_median_expression_degrades_gracefully_on_non_429_http_error(monkeypatch):
    """Same as above, but for the second request (medianGeneExpression) -- a gene that resolves
    fine but then fails on the expression lookup must also degrade, not crash."""

    def fake_get(url, params, timeout):
        if url.endswith("/reference/gene"):
            return _FakeResponse({"data": [{"geneSymbol": "IRS1", "gencodeId": "ENSG00000169213.12"}]})
        return _FakeResponse({}, status_code=500, text="Internal Server Error")

    monkeypatch.setattr("PipelineB.tissue_client.requests.get", fake_get)
    client = GTExClient()

    assert client.get_median_expression("IRS1", "Pancreas") is None
