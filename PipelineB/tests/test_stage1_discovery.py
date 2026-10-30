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


def test_run_stage1_discovery_end_to_end_with_stubs():
    ot_client, pubmed_client = _build_stub_clients()
    result = run_stage1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)

    assert result.disease1_efo_id == "EFO_T2DM"
    assert result.disease2_efo_id == "EFO_AD"


def test_run_stage1_discovery_filters_low_score_targets():
    ot_client, pubmed_client = _build_stub_clients()
    result = run_stage1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)

    d1_symbols = [t.gene_symbol for t in result.disease1_targets]
    d2_symbols = [t.gene_symbol for t in result.disease2_targets]
    assert "LOWSCORE1" not in d1_symbols
    assert "LOWSCORE2" not in d2_symbols
    assert "TCF7L2" in d1_symbols
    assert "APP" in d2_symbols


def test_run_stage1_discovery_cites_genes_from_both_diseases():
    ot_client, pubmed_client = _build_stub_clients()
    result = run_stage1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)

    assert result.citations["TCF7L2"][0].pmid == "111"
    assert result.citations["APP"][0].pmid == "333"


def test_run_stage1_discovery_searches_shared_gene_against_both_diseases():
    ot_client, pubmed_client = _build_stub_clients()
    result = run_stage1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)

    # INSR is in BOTH diseases' top-N, so (changed 2026-10-15; it used to be skipped on the
    # disease-2 side) it is searched once against each disease.
    insr_calls = [c for c in pubmed_client.calls if c[0] == "INSR"]
    assert [c[1] for c in insr_calls] == [DISEASE1, DISEASE2]

    # Both searches' citations live under the one gene key, each tagged with its own disease.
    insr_citations = result.citations["INSR"]
    assert [c.disease_name for c in insr_citations] == [DISEASE1, DISEASE2]
    assert all(c.gene_symbol == "INSR" for c in insr_citations)

    # And disease 2 keeps its full set of genes (no lost slot): APP and INSR.
    d2_searched = [s["gene_symbol"] for s in result.citation_searches if s["disease_name"] == DISEASE2]
    assert d2_searched == ["APP", "INSR"]


def test_run_stage1_discovery_to_dict_is_json_serializable():
    import json
    ot_client, pubmed_client = _build_stub_clients()
    result = run_stage1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)
    json.dumps(result.to_dict())


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


def test_stage1_result_without_citation_searches_still_works():
    # Results built by hand, or loaded from a run saved before citation_searches existed, omit it.
    result = Stage1DiscoveryResult(
        disease1=DISEASE1, disease2=DISEASE2, disease1_efo_id="EFO_T2DM", disease2_efo_id="EFO_AD",
        disease1_targets=[], disease2_targets=[], citations={"A": [Citation("1", "old-style, no provenance")]},
    )
    d = result.to_dict()
    assert d["citation_searches"] == []
    assert d["citations"]["A"][0]["disease_name"] is None
    assert d["citations"]["A"][0]["pmid"] == "1"


def test_result_records_data_versions_and_never_fails_on_a_version_lookup_error():
    from PipelineB.stage1_discovery import _data_versions

    class _Good:
        def get_data_version(self):
            return {"data_version": "26.09"}

    class _Bad:
        def get_data_version(self):
            raise RuntimeError("down")

    class _Pubmed:
        sort = "relevance"

    good = _data_versions(_Good(), _Pubmed())
    assert good["open_targets"] == {"data_version": "26.09"} and good["pubmed_sort"] == "relevance"
    assert "retrieved_at_utc" in good
    bad = _data_versions(_Bad(), object())
    assert "RuntimeError" in bad["open_targets"]["error"] and "recency" in bad["pubmed_sort"]
    assert "open_targets" not in _data_versions(object(), object())


def test_stage1_result_round_trips_through_to_dict_and_from_dict():
    # added 2026-10-18 so Stages 2-4 can be re-run on a saved Stage 1 output (scripts/run_all --reuse-saved-stage1)
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
