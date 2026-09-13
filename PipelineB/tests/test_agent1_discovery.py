from PipelineB.agent1_discovery import run_agent1_discovery
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


def test_run_agent1_discovery_end_to_end_with_stubs():
    ot_client, pubmed_client = _build_stub_clients()
    result = run_agent1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)

    assert result.disease1_efo_id == "EFO_T2DM"
    assert result.disease2_efo_id == "EFO_AD"


def test_run_agent1_discovery_filters_low_score_targets():
    ot_client, pubmed_client = _build_stub_clients()
    result = run_agent1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)

    d1_symbols = [t.gene_symbol for t in result.disease1_targets]
    d2_symbols = [t.gene_symbol for t in result.disease2_targets]
    assert "LOWSCORE1" not in d1_symbols
    assert "LOWSCORE2" not in d2_symbols
    assert "TCF7L2" in d1_symbols
    assert "APP" in d2_symbols


def test_run_agent1_discovery_cites_genes_from_both_diseases():
    ot_client, pubmed_client = _build_stub_clients()
    result = run_agent1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)

    assert result.citations["TCF7L2"][0].pmid == "111"
    assert result.citations["APP"][0].pmid == "333"


def test_run_agent1_discovery_does_not_double_cite_shared_gene():
    ot_client, pubmed_client = _build_stub_clients()
    run_agent1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)

    # INSR appears in both disease1 and disease2's target lists; cite_genes
    # should only call search_citations for it once (via disease1, since
    # that's processed first), not once per disease.
    insr_calls = [c for c in pubmed_client.calls if c[0] == "INSR"]
    assert len(insr_calls) == 1
    assert insr_calls[0][1] == DISEASE1


def test_run_agent1_discovery_to_dict_is_json_serializable():
    import json
    ot_client, pubmed_client = _build_stub_clients()
    result = run_agent1_discovery(DISEASE1, DISEASE2, ot_client=ot_client, pubmed_client=pubmed_client, min_score=0.1)
    json.dumps(result.to_dict())
