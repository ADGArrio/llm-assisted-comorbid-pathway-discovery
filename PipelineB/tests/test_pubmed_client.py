import json

from PipelineB.pubmed_client import (
    Citation,
    EdgeCitation,
    PubMedClient,
    StubPubMedClient,
    _bioc_collections,
    _contains_both_genes,
    _first_paragraph_with_both_genes,
)


def test_citation_to_dict_is_json_serializable():
    import json
    c = Citation(pmid="12345", title="Some paper title")
    json.dumps(c.to_dict())


def test_stub_returns_configured_citations():
    client = StubPubMedClient(citations_by_gene={"TCF7L2": [Citation("1", "Paper A"), Citation("2", "Paper B")]})
    result = client.search_citations("TCF7L2", "Type 2 Diabetes Mellitus", retmax=3)
    assert [c.pmid for c in result] == ["1", "2"]
    assert client.calls == [("TCF7L2", "Type 2 Diabetes Mellitus", 3)]


def test_stub_respects_retmax():
    client = StubPubMedClient(citations_by_gene={"APP": [Citation("1", "A"), Citation("2", "B"), Citation("3", "C")]})
    result = client.search_citations("APP", "Alzheimer's Disease", retmax=2)
    assert len(result) == 2


def test_stub_returns_empty_list_for_unknown_gene():
    client = StubPubMedClient(citations_by_gene={})
    result = client.search_citations("UNKNOWNGENE", "Some Disease")
    assert result == []


# --- Edge-level literature validation (added 2026-09-27) ---
# See module docstring's "Edge-level literature validation" note and
# claude/committee_feedback_round2_coscientist.md section 3a.


def test_edge_citation_to_dict_is_json_serializable():
    import json
    c = EdgeCitation(
        pmid="999", title="A paper", full_text_checked=True, matched_text="PRKCA activates ARRB1 directly.",
        both_genes_confirmed=True,
    )
    json.dumps(c.to_dict())


def test_contains_both_genes_requires_word_boundary_match():
    # "APP" must not match inside "APPROXIMATELY" -- a bare substring check would wrongly confirm this.
    assert _contains_both_genes("PRKCA activates ARRB1 in this cascade.", "PRKCA", "ARRB1") is True
    assert _contains_both_genes("This is APPROXIMATELY the right dose.", "APP", "INSR") is False


def test_contains_both_genes_is_case_insensitive():
    assert _contains_both_genes("prkca activates arrb1.", "PRKCA", "ARRB1") is True


def test_first_paragraph_with_both_genes_returns_first_match_only():
    paragraphs = [
        "This paragraph only mentions PRKCA.",
        "This paragraph mentions both PRKCA and ARRB1 together.",
        "This paragraph also mentions both PRKCA and ARRB1.",
    ]
    result = _first_paragraph_with_both_genes(paragraphs, "PRKCA", "ARRB1")
    assert result == "This paragraph mentions both PRKCA and ARRB1 together."


def test_first_paragraph_with_both_genes_returns_none_if_no_paragraph_matches():
    paragraphs = ["Only PRKCA here.", "Only ARRB1 here."]
    assert _first_paragraph_with_both_genes(paragraphs, "PRKCA", "ARRB1") is None


def test_stub_search_edge_citations_returns_configured_results():
    edge_citation = EdgeCitation(
        pmid="111", title="PRKCA and ARRB1 interaction", full_text_checked=True,
        matched_text="PRKCA directly activates ARRB1.", both_genes_confirmed=True,
    )
    client = StubPubMedClient(citations_by_gene={}, edge_citations_by_pair={("PRKCA", "ARRB1"): [edge_citation]})

    result = client.search_edge_citations("PRKCA", "ARRB1", retmax=3)

    assert result == [edge_citation]
    assert client.edge_calls == [("PRKCA", "ARRB1", 3)]


def test_stub_search_edge_citations_lookup_is_order_insensitive():
    """The real query is symmetric in gene_a/gene_b, so a chain edge and its reverse must find
    the same configured stub result regardless of which gene is passed first."""
    edge_citation = EdgeCitation(
        pmid="111", title="t", full_text_checked=False, matched_text="abstract text", both_genes_confirmed=True
    )
    client = StubPubMedClient(citations_by_gene={}, edge_citations_by_pair={("PRKCA", "ARRB1"): [edge_citation]})

    result = client.search_edge_citations("ARRB1", "PRKCA")

    assert result == [edge_citation]


def test_stub_search_edge_citations_returns_empty_list_for_unknown_pair():
    client = StubPubMedClient(citations_by_gene={})
    result = client.search_edge_citations("GENEA", "GENEB")
    assert result == []


# --- Real BioC-PMC response shape regression (added 2026-09-28) ---
# A live run against the real BioC-PMC endpoint raised
# AttributeError: 'list' object has no attribute 'get' inside
# _get_pmc_full_text_paragraphs, because the real response's top-level JSON
# value is a LIST of BioCCollection objects, not a bare dict as originally
# assumed. These tests pin down the fix (_bioc_collections) against that
# real shape so this can't silently regress.
#
# Captured 2026-09-28 from a real request to
# https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC7096080/unicode
_REAL_BIOC_PMC_RESPONSE_SHAPE = [
    {
        "bioctype": "BioCCollection",
        "source": "PMC",
        "documents": [
            {
                "bioctype": "BioCDocument",
                "id": "PMC7096080",
                "passages": [
                    {"bioctype": "BioCPassage", "offset": 0, "text": "This paragraph only mentions PRKCA."},
                    {
                        "bioctype": "BioCPassage",
                        "offset": 100,
                        "text": "This paragraph mentions both PRKCA and ARRB1 together.",
                    },
                ],
            }
        ],
    }
]


def test_bioc_collections_handles_real_list_wrapped_shape():
    assert _bioc_collections(_REAL_BIOC_PMC_RESPONSE_SHAPE) == _REAL_BIOC_PMC_RESPONSE_SHAPE


def test_bioc_collections_ignores_non_dict_items_in_the_list():
    assert _bioc_collections([{"a": 1}, "not-a-dict", 42]) == [{"a": 1}]


def test_bioc_collections_still_accepts_a_bare_dict_as_a_fallback():
    # Not the real shape (see above), but accepted defensively in case a future API revision or
    # a different BioC-compatible mirror ever drops the list wrapper.
    bare_dict = _REAL_BIOC_PMC_RESPONSE_SHAPE[0]
    assert _bioc_collections(bare_dict) == [bare_dict]


def test_bioc_collections_returns_empty_list_for_unexpected_shapes():
    assert _bioc_collections(None) == []
    assert _bioc_collections("not json") == []
    assert _bioc_collections(123) == []


def test_get_pmc_full_text_paragraphs_parses_the_real_response_shape():
    """End-to-end through the real (non-stub) PubMedClient method, with only the two real
    network calls stubbed out -- this is what actually raised the live AttributeError."""
    client = PubMedClient(api_key="k", email="e@example.com")
    client._pmid_to_pmcid = lambda pmid: "PMC7096080"
    client._get_text = lambda url, params=None: json.dumps(_REAL_BIOC_PMC_RESPONSE_SHAPE)

    paragraphs = client._get_pmc_full_text_paragraphs("12345")

    assert paragraphs == [
        "This paragraph only mentions PRKCA.",
        "This paragraph mentions both PRKCA and ARRB1 together.",
    ]
