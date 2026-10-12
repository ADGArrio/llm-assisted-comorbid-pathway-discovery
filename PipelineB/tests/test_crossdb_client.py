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


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    # GWASCatalogClient._get paces every request and backs off on 429 with real time.sleep()
    # calls (see crossdb_client.py's bug-fix #2 note) -- fine for a real run, but every test in
    # this file would otherwise burn real wall-clock time for no reason. Record calls instead so
    # backoff-timing tests can still assert on them.
    sleeps = []
    monkeypatch.setattr("PipelineB.crossdb_client.time.sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


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
        self.headers = {}

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


def test_gwas_client_stops_following_snps_once_a_match_is_found(monkeypatch):
    """Regression test for the 2026-09-14 bug-fix #2 (see crossdb_client.py's module
    docstring): a real run's Agent 3 step got 429'd by GWAS Catalog because it followed every
    matched gene's SNPs (up to 20) even after already confirming a hit. get_gene_disease_associations
    must stop following further SNPs for a gene as soon as one matching association is found."""
    search_payload = {
        "_embedded": {
            "singleNucleotidePolymorphisms": [
                {"rsId": "rs1", "_links": {"associations": {"href": "https://api.example/snps/rs1/associations"}}},
                {"rsId": "rs2", "_links": {"associations": {"href": "https://api.example/snps/rs2/associations"}}},
            ]
        }
    }
    match_payload = {
        "_embedded": {"associations": [{"pvalue": 1e-10, "studyAccession": "GCST0002", "efoTraits": [_efo_trait("Type 2 diabetes mellitus")]}]}
    }

    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        if url.endswith(_ASSOCIATIONS_BY_GENE_PATH):
            return _FakeResponse(search_payload)
        assert url == "https://api.example/snps/rs1/associations"  # rs2 must never be requested
        return _FakeResponse(match_payload)

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    client = GWASCatalogClient()

    results = client.get_gene_disease_associations("TCF7L2", "Type 2 Diabetes Mellitus")

    assert len(calls) == 2  # search + rs1's follow-up only -- rs2 skipped once rs1 matched
    assert len(results) == 1


def test_gwas_client_dedupes_repeated_snp_entries_on_a_page(monkeypatch):
    """Regression test for bug-fix #4 (see crossdb_client.py's module docstring): a live
    findByGene response for IRS1 listed every rsId twice in a row within a single page. A
    duplicate entry must not cost a second follow-up request for the same SNP."""
    search_payload = {
        "_embedded": {
            "singleNucleotidePolymorphisms": [
                {"rsId": "rs1", "_links": {"associations": {"href": "https://api.example/snps/rs1/associations"}}},
                {"rsId": "rs1", "_links": {"associations": {"href": "https://api.example/snps/rs1/associations"}}},
            ]
        }
    }
    empty_associations = {"_embedded": {"associations": []}}

    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        if url.endswith(_ASSOCIATIONS_BY_GENE_PATH):
            return _FakeResponse(search_payload)
        assert url == "https://api.example/snps/rs1/associations"
        return _FakeResponse(empty_associations)

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    client = GWASCatalogClient()

    client.get_gene_disease_associations("TCF7L2", "Type 2 Diabetes Mellitus")

    assert len(calls) == 2  # search + exactly one follow-up for rs1, not two


def test_gwas_client_paginates_to_find_a_match_on_a_later_page(monkeypatch):
    """Regression test for bug-fix #4: a real run's IRS1 check never found its known T2DM SNP
    because it only ever requested page 0 of findByGene, and that SNP was on a later page (6
    total pages existed for IRS1 in the live diagnostic that found this). If page 0 doesn't
    exhaust the distinct-SNP cap and more pages exist (per the response's own page.totalPages),
    get_gene_disease_associations must keep walking pages rather than stopping after page 0."""
    page0 = {
        "_embedded": {
            "singleNucleotidePolymorphisms": [
                {"rsId": "rs_no_match", "_links": {"associations": {"href": "https://api.example/snps/rs_no_match/associations"}}}
            ]
        },
        "page": {"size": 20, "totalElements": 21, "totalPages": 2, "number": 0},
    }
    page1 = {
        "_embedded": {
            "singleNucleotidePolymorphisms": [
                {"rsId": "rs_match", "_links": {"associations": {"href": "https://api.example/snps/rs_match/associations"}}}
            ]
        },
        "page": {"size": 20, "totalElements": 21, "totalPages": 2, "number": 1},
    }
    no_match_associations = {"_embedded": {"associations": []}}
    match_associations = {
        "_embedded": {"associations": [{"pvalue": 9e-12, "studyAccession": "GCST999", "efoTraits": [_efo_trait("Type 2 diabetes mellitus")]}]}
    }

    search_calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        if url.endswith(_ASSOCIATIONS_BY_GENE_PATH):
            search_calls.append(params)
            return _FakeResponse(page0 if params["page"] == 0 else page1)
        if url.endswith("rs_no_match/associations"):
            return _FakeResponse(no_match_associations)
        assert url.endswith("rs_match/associations")
        return _FakeResponse(match_associations)

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    client = GWASCatalogClient()

    results = client.get_gene_disease_associations("IRS1", "Type 2 Diabetes Mellitus")

    assert [c["page"] for c in search_calls] == [0, 1]  # walked into page 1 to find the match
    assert len(results) == 1
    assert results[0].study_accession == "GCST999"


def test_gwas_client_stops_paginating_after_max_search_pages_even_with_more_available(monkeypatch):
    """Safety cap: even if the live response keeps claiming more pages exist and no match is
    ever found, get_gene_disease_associations must not walk pages forever."""
    import PipelineB.crossdb_client as crossdb_client

    monkeypatch.setattr(crossdb_client, "_MAX_SEARCH_PAGES_PER_GENE", 2)

    def fake_get(url, params=None, timeout=None, headers=None):
        assert url.endswith(_ASSOCIATIONS_BY_GENE_PATH)
        page_number = params["page"]
        return _FakeResponse(
            {
                "_embedded": {"singleNucleotidePolymorphisms": []},
                "page": {"size": 20, "totalElements": 1000, "totalPages": 50, "number": page_number},
            }
        )

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    client = crossdb_client.GWASCatalogClient()

    results = client.get_gene_disease_associations("TCF7L2", "Type 2 Diabetes Mellitus")

    assert results == []  # gave up, not stuck in a loop -- reaching this line at all is the test


def test_gwas_client_follows_efo_traits_link_when_not_embedded(monkeypatch):
    """Regression test for bug-fix #5 (see crossdb_client.py's module docstring): a live
    diagnostic on IRS1's known T2DM SNP (rs2943641) found 32/32 real associations with an empty
    efoTraits list -- the actual trait data lives at each association's own
    `_links.efoTraits.href`, the same HAL "embed on request only" pattern bug-fix #1 already
    found one level up (SNP -> associations). _parse_association must follow it."""
    search_payload = {
        "_embedded": {
            "singleNucleotidePolymorphisms": [
                {"rsId": "rs2943641", "_links": {"associations": {"href": "https://api.example/snps/rs2943641/associations"}}}
            ]
        }
    }
    associations_payload = {
        "_embedded": {
            "associations": [
                {"pvalue": 9e-12, "_links": {"efoTraits": {"href": "https://api.example/associations/13684/efoTraits"}}}
            ]
        }
    }
    efo_traits_payload = {
        "_embedded": {"efoTraits": [{"trait": "type 2 diabetes mellitus", "shortForm": "MONDO_0005148"}]}
    }

    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        if url.endswith(_ASSOCIATIONS_BY_GENE_PATH):
            return _FakeResponse(search_payload)
        if url.endswith("rs2943641/associations"):
            return _FakeResponse(associations_payload)
        assert url == "https://api.example/associations/13684/efoTraits"
        return _FakeResponse(efo_traits_payload)

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    client = GWASCatalogClient()

    results = client.get_gene_disease_associations("IRS1", "Type 2 Diabetes Mellitus")

    assert len(calls) == 3  # search + snp associations + efoTraits follow-up
    assert len(results) == 1
    assert results[0].trait_name == "type 2 diabetes mellitus"
    assert results[0].efo_id == "MONDO_0005148"


def test_gwas_client_stops_fetching_efo_traits_once_a_match_is_found(monkeypatch):
    """A single busy SNP can have dozens of associations (rs2943641 had 32 in a live run) --
    following every one's efoTraits link after already matching one would be pure waste, and a
    real rate-limit risk (see bug-fix #3). Must stop as soon as one association matches."""
    search_payload = {
        "_embedded": {
            "singleNucleotidePolymorphisms": [
                {"rsId": "rs1", "_links": {"associations": {"href": "https://api.example/snps/rs1/associations"}}}
            ]
        }
    }
    associations_payload = {
        "_embedded": {
            "associations": [
                {"pvalue": 1e-10, "_links": {"efoTraits": {"href": "https://api.example/associations/1/efoTraits"}}},
                {"pvalue": 1e-8, "_links": {"efoTraits": {"href": "https://api.example/associations/2/efoTraits"}}},
            ]
        }
    }
    match_traits = {"_embedded": {"efoTraits": [{"trait": "type 2 diabetes mellitus", "shortForm": "MONDO_0005148"}]}}

    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        if url.endswith(_ASSOCIATIONS_BY_GENE_PATH):
            return _FakeResponse(search_payload)
        if url.endswith("rs1/associations"):
            return _FakeResponse(associations_payload)
        assert url == "https://api.example/associations/1/efoTraits"  # association 2 must never be requested
        return _FakeResponse(match_traits)

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    client = GWASCatalogClient()

    results = client.get_gene_disease_associations("TCF7L2", "Type 2 Diabetes Mellitus")

    assert len(calls) == 3  # search + snp associations + only the first association's efoTraits
    assert len(results) == 1


def test_gwas_client_flags_last_check_rate_limited_when_efo_traits_followup_exhausts_retries(monkeypatch):
    """Same degradation philosophy as the SNP-level and search-level follow-ups (bug-fix #3):
    if the efoTraits follow-up itself is still rate-limited after retries, that association is
    treated as 'no match here', not an error, and the check is flagged possibly_incomplete."""
    search_payload = {
        "_embedded": {
            "singleNucleotidePolymorphisms": [
                {"rsId": "rs1", "_links": {"associations": {"href": "https://api.example/snps/rs1/associations"}}}
            ]
        }
    }
    associations_payload = {
        "_embedded": {
            "associations": [
                {"pvalue": 1e-10, "_links": {"efoTraits": {"href": "https://api.example/associations/1/efoTraits"}}}
            ]
        }
    }

    def fake_get(url, params=None, timeout=None, headers=None):
        if url.endswith(_ASSOCIATIONS_BY_GENE_PATH):
            return _FakeResponse(search_payload)
        if url.endswith("rs1/associations"):
            return _FakeResponse(associations_payload)
        return _FakeResponse({}, status_code=429)

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    client = GWASCatalogClient()

    results = client.get_gene_disease_associations("TCF7L2", "Type 2 Diabetes Mellitus")

    assert results == []
    assert client.last_check_rate_limited is True


def test_gwas_client_retries_429_and_succeeds(monkeypatch, _no_real_sleeping):
    """A single transient 429 must not fail the request -- retry and use the second response."""
    search_payload = {"_embedded": {"singleNucleotidePolymorphisms": []}}
    responses = [_FakeResponse({}, status_code=429), _FakeResponse(search_payload, status_code=200)]

    def fake_get(url, params=None, timeout=None, headers=None):
        return responses.pop(0)

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    client = GWASCatalogClient()

    results = client.get_gene_disease_associations("TCF7L2", "Type 2 Diabetes Mellitus")

    assert results == []  # no SNPs in the (eventually successful) payload -- not an error
    assert responses == []  # both queued responses were consumed


def test_gwas_client_degrades_to_empty_list_after_exhausting_retries(monkeypatch):
    """If GWAS Catalog is still rate-limiting after every retry, the whole pipeline must not
    crash (this is exactly what happened in the real run that motivated this fix) -- the gene
    is treated as 'not checked this time', same shape as 'no data found'."""
    monkeypatch.setattr(
        "PipelineB.crossdb_client.requests.get",
        lambda url, params=None, timeout=None, headers=None: _FakeResponse({}, status_code=429),
    )
    client = GWASCatalogClient()

    assert client.get_gene_disease_associations("TCF7L2", "Type 2 Diabetes Mellitus") == []


def test_gwas_client_flags_last_check_rate_limited_when_search_exhausts_retries(monkeypatch):
    """Regression test for bug-fix #3 (see crossdb_client.py's module docstring): a real run
    reported '[] / not confirmed' for IRS1 x T2DM -- a gene with strong, well-replicated GWAS
    evidence -- after a long run of 429s. last_check_rate_limited lets a caller tell that apart
    from a clean negative."""
    monkeypatch.setattr(
        "PipelineB.crossdb_client.requests.get",
        lambda url, params=None, timeout=None, headers=None: _FakeResponse({}, status_code=429),
    )
    client = GWASCatalogClient()

    assert client.last_check_rate_limited is False  # nothing checked yet
    client.get_gene_disease_associations("IRS1", "Type 2 Diabetes Mellitus")
    assert client.last_check_rate_limited is True


def test_gwas_client_flags_last_check_rate_limited_when_snp_followup_exhausts_retries(monkeypatch):
    """Same as above, but the top-level search succeeds and the flag comes from a per-SNP
    associations follow-up exhausting retries instead (see _associations_for_snp)."""
    search_payload = {
        "_embedded": {
            "singleNucleotidePolymorphisms": [
                {"rsId": "rs1", "_links": {"associations": {"href": "https://api.example/snps/rs1/associations"}}}
            ]
        }
    }

    def fake_get(url, params=None, timeout=None, headers=None):
        if url.endswith(_ASSOCIATIONS_BY_GENE_PATH):
            return _FakeResponse(search_payload)
        return _FakeResponse({}, status_code=429)

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    client = GWASCatalogClient()

    results = client.get_gene_disease_associations("IRS1", "Type 2 Diabetes Mellitus")

    assert results == []
    assert client.last_check_rate_limited is True


def test_gwas_client_resets_last_check_rate_limited_on_a_clean_later_call(monkeypatch):
    """last_check_rate_limited must reflect only the most recent call -- a clean call after a
    rate-limited one must not leave a stale True flag behind for the caller to misread."""
    always_429 = {"on": True}  # every request 429s while True, so the first call exhausts retries

    def fake_get(url, params=None, timeout=None, headers=None):
        if always_429["on"]:
            return _FakeResponse({}, status_code=429)
        return _FakeResponse({"_embedded": {"singleNucleotidePolymorphisms": []}})

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    client = GWASCatalogClient()

    client.get_gene_disease_associations("IRS1", "Type 2 Diabetes Mellitus")
    assert client.last_check_rate_limited is True

    always_429["on"] = False
    client.get_gene_disease_associations("APOE", "Alzheimer's Disease")
    assert client.last_check_rate_limited is False


def test_gwas_client_honors_retry_after_header(monkeypatch, _no_real_sleeping):
    class _RateLimitedResponse(_FakeResponse):
        def __init__(self):
            super().__init__({}, status_code=429)
            self.headers = {"Retry-After": "7"}

    class _OkResponse(_FakeResponse):
        def __init__(self):
            super().__init__({"_embedded": {"singleNucleotidePolymorphisms": []}}, status_code=200)
            self.headers = {}

    responses = [_RateLimitedResponse(), _OkResponse()]
    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", lambda url, params=None, timeout=None, headers=None: responses.pop(0))
    client = GWASCatalogClient()

    client.get_gene_disease_associations("TCF7L2", "Type 2 Diabetes Mellitus")

    assert 7.0 in _no_real_sleeping  # waited the server-specified Retry-After, not the default backoff


def test_gwas_client_retries_connection_failure_and_succeeds(monkeypatch, _no_real_sleeping):
    """Regression test for bug-fix #6 (see crossdb_client.py's module docstring): a real
    T2DM->CAD run crashed with an uncaught ConnectTimeout -- a connection-level failure, not a
    429 response -- because only the 429 path retried. A single transient connection failure
    must not fail the request -- retry and use the response that eventually comes back."""
    import requests as requests_module

    search_payload = {"_embedded": {"singleNucleotidePolymorphisms": []}}
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None, headers=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests_module.exceptions.ConnectTimeout("connection timed out")
        return _FakeResponse(search_payload, status_code=200)

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", fake_get)
    client = GWASCatalogClient()

    results = client.get_gene_disease_associations("TCF7L2", "Type 2 Diabetes Mellitus")

    assert results == []  # no SNPs in the (eventually successful) payload -- not an error
    assert calls["n"] == 2  # first call raised, second call was the retry that succeeded


def test_gwas_client_degrades_and_flags_rate_limited_after_exhausting_connection_retries(monkeypatch):
    """If GWAS Catalog (or the network path to it) never responds at all, the whole pipeline
    must not crash -- this is exactly the uncaught ConnectTimeout that motivated bug-fix #6.
    The gene is treated as 'not checked this time' (same shape as a persistent 429), and
    last_check_rate_limited is set so callers can tell it apart from a clean negative."""
    import requests as requests_module

    def always_times_out(url, params=None, timeout=None, headers=None):
        raise requests_module.exceptions.ConnectTimeout("connection timed out")

    monkeypatch.setattr("PipelineB.crossdb_client.requests.get", always_times_out)
    monkeypatch.setattr("PipelineB.crossdb_client.time.sleep", lambda seconds: None)
    client = GWASCatalogClient()

    assert client.last_check_rate_limited is False  # nothing checked yet
    results = client.get_gene_disease_associations("IRS1", "Type 2 Diabetes Mellitus")

    assert results == []
    assert client.last_check_rate_limited is True
