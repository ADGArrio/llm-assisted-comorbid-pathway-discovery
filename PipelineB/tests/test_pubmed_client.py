from PipelineB.pubmed_client import Citation, StubPubMedClient


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
