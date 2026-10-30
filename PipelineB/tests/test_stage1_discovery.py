"""Tests for Stage 1 Discovery (PipelineB/stage1_discovery.py) with stub clients."""

from PipelineB.stage1_discovery import Stage1DiscoveryResult, run_stage1_discovery
from PipelineB.opentargets_client import GeneAssociation, StubOpenTargetsClient
from PipelineB.pubmed_client import Citation, StubPubMedClient

DISEASE1 = "Type 2 Diabetes Mellitus"
DISEASE2 = "Alzheimer's Disease"

DISEASE1_TARGETS = [
    GeneAssociation("TCF7L2", "ENSG_TCF7L2", "Q9NQB0", 0.71),
    GeneAssociation("INSR", "ENSG_INSR", "P06213", 0.55),
    GeneAssociation("LOWSCORE1", "ENSG_LOW1", "P00001", 0.02),
]
DISEASE2_TARGETS = [
    GeneAssociation("APP", "ENSG_APP", "P05067", 0.90),
    # Deliberately overlaps with a disease1 gene, to exercise the
    # "don't double-cite a gene already looked up" path in cite_genes.
    GeneAssociation("INSR", "ENSG_INSR", "P06213", 0.40),
    GeneAssociation("LOWSCORE2", "ENSG_LOW2", "P00002", 0.01),
]


# Build stub Open Targets and PubMed clients for the two diseases.
def _build_stub_clients():
    ot_client = StubOpenTargetsClient(
        disease_to_efo={DISEASE1: "EFO_T2DM", DISEASE2: "EFO_AD"},
        efo_to_targets={"EFO_T2DM": DISEASE1_TARGETS, "EFO_AD": DISEASE2_TARGETS},
    )
    pubmed_client = StubPubMedClient(
        citations_by_gene={
            "TCF7L2": [Citation("111", "TCF7L2 in T2DM")],
            "INSR": [Citation("222", "INSR signaling")],
            "APP": [Citation("333", "APP processing in AD")],
        }
    )
    return ot_client, pubmed_client


# Run stage1 discovery end to end with stubs.
def test_run_stage1_discovery_end_to_end_with_stubs():
    ot_client, pubmed_client = _build_stub_clients()
    result = run_stage1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)

    assert result.disease1_efo_id == "EFO_T2DM"
    assert result.disease2_efo_id == "EFO_AD"


# Run stage1 discovery filters low score targets.
def test_run_stage1_discovery_filters_low_score_targets():
    ot_client, pubmed_client = _build_stub_clients()
    result = run_stage1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)

    d1_symbols = [t.gene_symbol for t in result.disease1_targets]
    d2_symbols = [t.gene_symbol for t in result.disease2_targets]
    assert "LOWSCORE1" not in d1_symbols
    assert "LOWSCORE2" not in d2_symbols
    assert "TCF7L2" in d1_symbols
    assert "APP" in d2_symbols


# Run stage1 discovery cites genes from both diseases.
def test_run_stage1_discovery_cites_genes_from_both_diseases():
    ot_client, pubmed_client = _build_stub_clients()
    result = run_stage1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)

    assert result.citations["TCF7L2"][0].pmid == "111"
    assert result.citations["APP"][0].pmid == "333"


# Run stage1 discovery searches shared gene against both diseases.
def test_run_stage1_discovery_searches_shared_gene_against_both_diseases():
    ot_client, pubmed_client = _build_stub_clients()
    result = run_stage1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)

    # INSR is in both diseases' top-N, so it is searched once against each disease.
    insr_calls = [c for c in pubmed_client.calls if c[0] == "INSR"]
    assert [c[1] for c in insr_calls] == [DISEASE1, DISEASE2]

    # Both searches' citations live under the one gene key, each tagged with its own disease.
    insr_citations = result.citations["INSR"]
    assert [c.disease_name for c in insr_citations] == [DISEASE1, DISEASE2]
    assert all(c.gene_symbol == "INSR" for c in insr_citations)

    # And disease 2 keeps its full set of genes (no lost slot): APP and INSR.
    d2_searched = [s["gene_symbol"] for s in result.citation_searches if s["disease_name"] == DISEASE2]
    assert d2_searched == ["APP", "INSR"]


# Run stage1 discovery to dict is JSON serializable.
def test_run_stage1_discovery_to_dict_is_json_serializable():
    import json
    ot_client, pubmed_client = _build_stub_clients()
    result = run_stage1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)
    json.dumps(result.to_dict())


# Run stage1 discovery tags every citation with its search.
def test_run_stage1_discovery_tags_every_citation_with_its_search():
    ot_client, pubmed_client = _build_stub_clients()
    result = run_stage1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)

    for gene, cites in result.citations.items():
        for c in cites:
            assert c.gene_symbol == gene
            assert c.disease_name in (DISEASE1, DISEASE2)
            assert c.query_term == f"{gene}[Title/Abstract] AND {c.disease_name}[Title/Abstract]"
            assert c.search_rank == 1

    # Which disease each search used is recorded correctly, not just present.
    assert result.citations["TCF7L2"][0].disease_name == DISEASE1
    assert result.citations["APP"][0].disease_name == DISEASE2


# Run stage1 discovery records every search including zero hit ones.
def test_run_stage1_discovery_records_every_search_including_zero_hit_ones():
    # A gene with no PubMed hits leaves no Citation behind, so without citation_searches there
    # would be no record that it was searched at all.
    targets_d1 = [
        GeneAssociation("TCF7L2", "ENSG_TCF7L2", "Q9NQB0", 0.71),
        GeneAssociation("INSR", "ENSG_INSR", "P06213", 0.55),
        GeneAssociation("NOHITS", "ENSG_NOHITS", "P00003", 0.50),
    ]
    ot_client = StubOpenTargetsClient(
        disease_to_efo={DISEASE1: "EFO_T2DM", DISEASE2: "EFO_AD"},
        efo_to_targets={"EFO_T2DM": targets_d1, "EFO_AD": DISEASE2_TARGETS},
    )
    pubmed_client = StubPubMedClient(
        citations_by_gene={"TCF7L2": [Citation("111", "t")], "INSR": [Citation("222", "i")], "APP": [Citation("333", "a")]}
    )
    result = run_stage1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)

    d1 = [s for s in result.citation_searches if s["disease_name"] == DISEASE1]
    assert [s["gene_symbol"] for s in d1] == ["TCF7L2", "INSR", "NOHITS"]  # in score order
    assert [s["gene_rank_in_disease"] for s in d1] == [1, 2, 3]
    assert all(s["disease_efo_id"] == "EFO_T2DM" for s in d1)

    nohits = d1[2]
    assert nohits["n_returned"] == 0
    assert nohits["total_hits"] == 0
    assert nohits["query_term"] == "NOHITS[Title/Abstract] AND Type 2 Diabetes Mellitus[Title/Abstract]"
    assert "NOHITS" in result.citations and result.citations["NOHITS"] == []

    # 3 disease-1 searches + 2 disease-2 searches (APP, INSR); LOWSCORE genes are filtered earlier.
    assert len(result.citation_searches) == 5


# Stage1 result without citation searches still works.
def test_stage1_result_without_citation_searches_still_works():
    # Results built by hand, or loaded from older saved runs, omit citation_searches.
    result = Stage1DiscoveryResult(
        disease1=DISEASE1, disease2=DISEASE2, disease1_efo_id="EFO_T2DM", disease2_efo_id="EFO_AD",
        disease1_targets=[], disease2_targets=[], citations={"A": [Citation("1", "old-style, no provenance")]},
    )
    d = result.to_dict()
    assert d["citation_searches"] == []
    assert d["citations"]["A"][0]["disease_name"] is None
    assert d["citations"]["A"][0]["pmid"] == "1"


# Result records data versions and never fails on a version lookup error.
def test_result_records_data_versions_and_never_fails_on_a_version_lookup_error():
    from PipelineB.stage1_discovery import _data_versions

    # Open Targets stand-in whose version lookup succeeds.
    class _Good:
        # Output: a fixed version dict.
        def get_data_version(self):
            return {"data_version": "26.09"}

    # Open Targets stand-in whose version lookup fails.
    class _Bad:
        # Raise to simulate a failed lookup.
        def get_data_version(self):
            raise RuntimeError("down")

    # PubMed stand-in that only carries a sort order.
    class _Pubmed:
        sort = "relevance"

    good = _data_versions(_Good(), _Pubmed())
    assert good["open_targets"] == {"data_version": "26.09"} and good["pubmed_sort"] == "relevance"
    assert "retrieved_at_utc" in good
    bad = _data_versions(_Bad(), object())
    assert "RuntimeError" in bad["open_targets"]["error"] and "recency" in bad["pubmed_sort"]
    assert "open_targets" not in _data_versions(object(), object())


# Stage1 result round trips through to dict and from dict.
def test_stage1_result_round_trips_through_to_dict_and_from_dict():
    from PipelineB.stage1_discovery import Stage1DiscoveryResult
    from PipelineB.opentargets_client import GeneAssociation
    from PipelineB.pubmed_client import Citation
    original = Stage1DiscoveryResult(
        disease1="D1", disease2="D2", disease1_efo_id="EFO_1", disease2_efo_id="EFO_2",
        disease1_targets=[GeneAssociation("INSR", "ENSG1", "P06213", 0.8)],
        disease2_targets=[GeneAssociation("APP", "ENSG2", None, 0.9)],
        citations={"INSR": [Citation(pmid="1", title="t", gene_symbol="INSR", disease_name="D1", search_rank=1)]},
        citation_searches=[{"gene_symbol": "INSR"}], data_versions={"open_targets": "26.09"},
    )
    assert Stage1DiscoveryResult.from_dict(original.to_dict()).to_dict() == original.to_dict()
