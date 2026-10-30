"""Tests for the PubMed client (PipelineB/pubmed_client.py) with stubbed HTTP responses."""

import json

from PipelineB.pubmed_client import (
    Citation,
    EdgeCitation,
    PubMedClient,
    StubPubMedClient,
    _abstracts_from_efetch_xml,
    _bioc_collections,
    citation_query_term,
    _contains_both_genes,
    _first_paragraph_with_both_genes,
)


# Citation to dict is JSON serializable.
def test_citation_to_dict_is_json_serializable():
    import json
    c = Citation(pmid="12345", title="Some paper title")
    json.dumps(c.to_dict())


# Stub returns configured citations.
def test_stub_returns_configured_citations():
    client = StubPubMedClient(citations_by_gene={"TCF7L2": [Citation("1", "Paper A"), Citation("2", "Paper B")]})
    result = client.search_citations("TCF7L2", "Type 2 Diabetes Mellitus", retmax=3)
    assert [c.pmid for c in result] == ["1", "2"]
    assert client.calls == [("TCF7L2", "Type 2 Diabetes Mellitus", 3)]


# Stub respects retmax.
def test_stub_respects_retmax():
    client = StubPubMedClient(citations_by_gene={"APP": [Citation("1", "A"), Citation("2", "B"), Citation("3", "C")]})
    result = client.search_citations("APP", "Alzheimer's Disease", retmax=2)
    assert len(result) == 2


# Stub returns empty list for unknown gene.
def test_stub_returns_empty_list_for_unknown_gene():
    client = StubPubMedClient(citations_by_gene={})
    result = client.search_citations("UNKNOWNGENE", "Some Disease")
    assert result == []


# --- Edge-level literature validation ---


# Edge citation to dict is JSON serializable.
def test_edge_citation_to_dict_is_json_serializable():
    import json
    c = EdgeCitation(
        pmid="999", title="A paper", full_text_checked=True, matched_text="PRKCA activates ARRB1 directly.",
        both_genes_confirmed=True,
    )
    json.dumps(c.to_dict())


# Contains both genes requires word boundary match.
def test_contains_both_genes_requires_word_boundary_match():
    # "APP" must not match inside "APPROXIMATELY" -- a bare substring check would wrongly confirm this.
    assert _contains_both_genes("PRKCA activates ARRB1 in this cascade.", "PRKCA", "ARRB1") is True
    assert _contains_both_genes("This is APPROXIMATELY the right dose.", "APP", "INSR") is False


# Contains both genes is case insensitive.
def test_contains_both_genes_is_case_insensitive():
    assert _contains_both_genes("prkca activates arrb1.", "PRKCA", "ARRB1") is True


# First paragraph with both genes returns first match only.
def test_first_paragraph_with_both_genes_returns_first_match_only():
    paragraphs = [
        "This paragraph only mentions PRKCA.",
        "This paragraph mentions both PRKCA and ARRB1 together.",
        "This paragraph also mentions both PRKCA and ARRB1.",
    ]
    result = _first_paragraph_with_both_genes(paragraphs, "PRKCA", "ARRB1")
    assert result == "This paragraph mentions both PRKCA and ARRB1 together."


# First paragraph with both genes returns none if no paragraph matches.
def test_first_paragraph_with_both_genes_returns_none_if_no_paragraph_matches():
    paragraphs = ["Only PRKCA here.", "Only ARRB1 here."]
    assert _first_paragraph_with_both_genes(paragraphs, "PRKCA", "ARRB1") is None


# Stub search edge citations returns configured results.
def test_stub_search_edge_citations_returns_configured_results():
    edge_citation = EdgeCitation(
        pmid="111", title="PRKCA and ARRB1 interaction", full_text_checked=True,
        matched_text="PRKCA directly activates ARRB1.", both_genes_confirmed=True,
    )
    client = StubPubMedClient(citations_by_gene={}, edge_citations_by_pair={("PRKCA", "ARRB1"): [edge_citation]})

    result = client.search_edge_citations("PRKCA", "ARRB1", retmax=3)

    assert result == [edge_citation]
    assert client.edge_calls == [("PRKCA", "ARRB1", 3)]


# Stub search edge citations lookup is order insensitive.
def test_stub_search_edge_citations_lookup_is_order_insensitive():
    edge_citation = EdgeCitation(
        pmid="111", title="t", full_text_checked=False, matched_text="abstract text", both_genes_confirmed=True
    )
    client = StubPubMedClient(citations_by_gene={}, edge_citations_by_pair={("PRKCA", "ARRB1"): [edge_citation]})

    result = client.search_edge_citations("ARRB1", "PRKCA")

    assert result == [edge_citation]


# Stub search edge citations returns empty list for unknown pair.
def test_stub_search_edge_citations_returns_empty_list_for_unknown_pair():
    client = StubPubMedClient(citations_by_gene={})
    result = client.search_edge_citations("GENEA", "GENEB")
    assert result == []


# --- Real BioC-PMC response shape ---
# The real response's top-level JSON value is a list of BioCCollection objects, not a bare dict; _bioc_collections must
# handle it.
# The fixture below is captured from a real BioC-PMC request.
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


# BioC collections handles real list wrapped shape.
def test_bioc_collections_handles_real_list_wrapped_shape():
    assert _bioc_collections(_REAL_BIOC_PMC_RESPONSE_SHAPE) == _REAL_BIOC_PMC_RESPONSE_SHAPE


# BioC collections ignores non dict items in the list.
def test_bioc_collections_ignores_non_dict_items_in_the_list():
    assert _bioc_collections([{"a": 1}, "not-a-dict", 42]) == [{"a": 1}]


# BioC collections still accepts a bare dict as a fallback.
def test_bioc_collections_still_accepts_a_bare_dict_as_a_fallback():
    # Not the real shape, but accepted defensively in case the API drops the list wrapper.
    bare_dict = _REAL_BIOC_PMC_RESPONSE_SHAPE[0]
    assert _bioc_collections(bare_dict) == [bare_dict]


# BioC collections returns empty list for unexpected shapes.
def test_bioc_collections_returns_empty_list_for_unexpected_shapes():
    assert _bioc_collections(None) == []
    assert _bioc_collections("not json") == []
    assert _bioc_collections(123) == []


# Get PMC full text paragraphs parses the real response shape.
def test_get_pmc_full_text_paragraphs_parses_the_real_response_shape():
    client = PubMedClient(api_key="k", email="e@example.com")
    client._pmid_to_pmcid = lambda pmid: "PMC7096080"
    client._get_text = lambda url, params=None: json.dumps(_REAL_BIOC_PMC_RESPONSE_SHAPE)

    paragraphs = client._get_pmc_full_text_paragraphs("12345")

    assert paragraphs == [
        "This paragraph only mentions PRKCA.",
        "This paragraph mentions both PRKCA and ARRB1 together.",
    ]


# --- Citation provenance and bibliographic metadata ---
# Every Citation records which gene and disease search produced it, plus the bibliographic fields in the esummary
# response.


# Citation query term uses title abstract field tags.
def test_citation_query_term_uses_title_abstract_field_tags():
    assert citation_query_term("INSR", "Alzheimer's Disease") == "INSR[Title/Abstract] AND Alzheimer's Disease[Title/Abstract]"


# Citation to dict includes every provenance and metadata field.
def test_citation_to_dict_includes_every_provenance_and_metadata_field():
    import json
    d = Citation("1", "Old-style citation").to_dict()  # pmid + title only still works
    assert set(d) == {
        "pmid", "title", "gene_symbol", "disease_name", "query_term", "search_rank", "search_total_hits",
        "pubdate", "epubdate", "journal", "authors", "doi", "pmcid", "pubtype", "url",
    }
    assert d["disease_name"] is None
    json.dumps(d)


# Stub tags returned citations with the search without mutating stored ones.
def test_stub_tags_returned_citations_with_the_search_without_mutating_stored_ones():
    stored = [Citation("1", "A"), Citation("2", "B")]
    client = StubPubMedClient(citations_by_gene={"APP": stored})
    result = client.search_citations("APP", "Alzheimer's Disease", retmax=2)

    assert [c.search_rank for c in result] == [1, 2]
    assert all(c.gene_symbol == "APP" and c.disease_name == "Alzheimer's Disease" for c in result)
    assert result[0].query_term == "APP[Title/Abstract] AND Alzheimer's Disease[Title/Abstract]"
    assert all(c.disease_name is None for c in stored)  # the configured objects are left alone


# Trimmed from a real esearch/esummary pair for INSR[Title/Abstract] AND Alzheimer's Disease[Title/Abstract] (retmax=2):
# real field names and shapes, authors shortened to two.
_REAL_ESEARCH_RESPONSE = {"esearchresult": {"count": "41", "retmax": "2", "idlist": ["42694370", "42250112"]}}
_REAL_ESUMMARY_RESPONSE = {
    "result": {
        "uids": ["42694370", "42250112"],
        "42694370": {
            "uid": "42694370",
            "title": "A paper about INSR in Alzheimer's disease.",
            "pubdate": "2026",
            "epubdate": "2026 Aug 20",
            "source": "Front Pharmacol",
            "fulljournalname": "Frontiers in pharmacology",
            "authors": [
                {"name": "Song J", "authtype": "Author", "clusterid": ""},
                {"name": "Yuan Y", "authtype": "Author", "clusterid": ""},
            ],
            "pubtype": ["Journal Article"],
            "articleids": [
                {"idtype": "pubmed", "idtypen": 1, "value": "42694370"},
                {"idtype": "pmc", "idtypen": 8, "value": "PMC13538836"},
                {"idtype": "pmcid", "idtypen": 5, "value": "pmc-id: PMC13538836;"},
                {"idtype": "doi", "idtypen": 3, "value": "10.3389/fphar.2026.1873606"},
            ],
        },
        # Second record deliberately sparse: no authors/ids/journal, as some real records are.
        "42250112": {"uid": "42250112", "title": "A sparse record."},
    }
}


# Build a PubMedClient whose esearch and esummary calls return the given responses and record the requests.
def _client_with_canned_responses(esearch, esummary):
    client = PubMedClient(api_key="k", email="e@example.com")
    client.requests_made = []

    # Stand-in for the real GET request, returning canned data.
    def fake_get(path, params):
        client.requests_made.append((path, params))
        return esearch if path == "esearch.fcgi" else esummary

    client._get = fake_get
    return client


# Search citations records provenance and bibliographic metadata.
def test_search_citations_records_provenance_and_bibliographic_metadata():
    client = _client_with_canned_responses(_REAL_ESEARCH_RESPONSE, _REAL_ESUMMARY_RESPONSE)
    result = client.search_citations("INSR", "Alzheimer's Disease", retmax=2)

    first = result[0]
    assert first.pmid == "42694370"
    assert first.gene_symbol == "INSR"
    assert first.disease_name == "Alzheimer's Disease"
    assert first.query_term == "INSR[Title/Abstract] AND Alzheimer's Disease[Title/Abstract]"
    assert first.search_rank == 1
    assert first.search_total_hits == 41  # all matching papers, not just the 2 returned
    assert first.pubdate == "2026" and first.epubdate == "2026 Aug 20"
    assert first.journal == "Frontiers in pharmacology"
    assert first.authors == ["Song J", "Yuan Y"]
    assert first.doi == "10.3389/fphar.2026.1873606"
    assert first.pmcid == "PMC13538836"
    assert first.pubtype == ["Journal Article"]
    assert first.url == "https://pubmed.ncbi.nlm.nih.gov/42694370/"

    # The query actually sent to PubMed is the recorded one.
    assert client.requests_made[0][1]["term"] == first.query_term


# Search citations tolerates a sparse esummary record.
def test_search_citations_tolerates_a_sparse_esummary_record():
    client = _client_with_canned_responses(_REAL_ESEARCH_RESPONSE, _REAL_ESUMMARY_RESPONSE)
    sparse = client.search_citations("INSR", "Alzheimer's Disease", retmax=2)[1]

    assert sparse.pmid == "42250112" and sparse.search_rank == 2
    assert sparse.title == "A sparse record."
    assert sparse.authors is None and sparse.doi is None and sparse.pmcid is None and sparse.journal is None
    assert sparse.url == "https://pubmed.ncbi.nlm.nih.gov/42250112/"


# Search citations returns empty list without calling esummary when nothing matches.
def test_search_citations_returns_empty_list_without_calling_esummary_when_nothing_matches():
    client = _client_with_canned_responses({"esearchresult": {"count": "0", "idlist": []}}, {})
    assert client.search_citations("NOHITS", "Some Disease") == []
    assert [path for path, _ in client.requests_made] == ["esearch.fcgi"]


# --- fetch_abstracts ---
# Trimmed from a real efetch XML response: a structured abstract with labelled sections and inline markup, and an
# article with no abstract.
_EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle><MedlineCitation><PMID Version="1">111</PMID><Article>
  <Abstract>
   <AbstractText Label="BACKGROUND">Insulin resistance is common.</AbstractText>
   <AbstractText Label="RESULTS">Knockdown of <i>INSR</i> reduced AKT signalling.</AbstractText>
  </Abstract></Article></MedlineCitation></PubmedArticle>
 <PubmedArticle><MedlineCitation><PMID Version="1">222</PMID><Article>
  <Abstract><AbstractText>An unlabeled abstract.</AbstractText></Abstract>
 </Article></MedlineCitation></PubmedArticle>
 <PubmedArticle><MedlineCitation><PMID Version="1">333</PMID><Article>
  <ArticleTitle>An editorial with no abstract</ArticleTitle>
 </Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


# Abstracts from efetch XML labels inline markup and missing.
def test_abstracts_from_efetch_xml_labels_inline_markup_and_missing():
    out = _abstracts_from_efetch_xml(_EFETCH_XML)
    assert out["111"] == "BACKGROUND: Insulin resistance is common. RESULTS: Knockdown of INSR reduced AKT signalling."
    assert out["222"] == "An unlabeled abstract."
    assert "333" not in out


# Abstracts from efetch XML bad XML returns empty.
def test_abstracts_from_efetch_xml_bad_xml_returns_empty():
    assert _abstracts_from_efetch_xml("<html>rate limited") == {}


# Fetch abstracts batches dedupes and tolerates failed batch.
def test_fetch_abstracts_batches_dedupes_and_tolerates_failed_batch():
    client = PubMedClient()
    calls = []

    # Stand-in for the real text GET request, returning canned data.
    def fake_get_text(url, params=None):
        calls.append(params["id"])
        return _EFETCH_XML if len(calls) == 1 else None  # second batch fails -> just absent

    client._get_text = fake_get_text
    out = client.fetch_abstracts(["111", "222", "111", "444"], batch_size=2)
    assert calls == ["111,222", "444"]
    assert set(out) == {"111", "222"}


# Search citations sends the sort only when one is configured.
def test_search_citations_sends_the_sort_only_when_one_is_configured(monkeypatch):
    from PipelineB.pubmed_client import PubMedClient

    seen = []

    # Stand-in for the real GET request, returning canned data.
    def fake_get(self, path, params):
        seen.append((path, dict(params)))
        return {"esearchresult": {"idlist": [], "count": "0"}}

    monkeypatch.setattr(PubMedClient, "_get", fake_get)
    PubMedClient().search_citations("INSR", "Alzheimer's Disease")
    PubMedClient(sort="relevance").search_citations("INSR", "Alzheimer's Disease")
    assert "sort" not in seen[0][1] and seen[1][1]["sort"] == "relevance"
