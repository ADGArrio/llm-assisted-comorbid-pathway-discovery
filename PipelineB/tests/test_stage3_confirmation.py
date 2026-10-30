"""Tests for Stage 3 Confirmation (PipelineB/stage3_confirmation.py) with stub clients."""

import json

from PipelineB.stage1_discovery import Stage1DiscoveryResult
from PipelineB.stage2_judging import Stage2JudgingResult, ChainCandidate, ChainEdge
from PipelineB.stage3_confirmation import run_stage3_confirmation
from PipelineB.crossdb_client import GwasAssociation, StubGWASCatalogClient
from PipelineB.opentargets_client import GeneAssociation
from PipelineB.pubmed_client import EdgeCitation, StubPubMedClient
from PipelineB.tissue_client import StubGTExClient

DISEASE1 = "Type 2 Diabetes Mellitus"
DISEASE2 = "Alzheimer's Disease"


# Build a Stage 1 result for the two diseases.
def _stage1_result():
    return Stage1DiscoveryResult(
        disease1=DISEASE1,
        disease2=DISEASE2,
        disease1_efo_id="EFO_T2DM",
        disease2_efo_id="EFO_AD",
        disease1_targets=[GeneAssociation("INSR", "ENSG_INSR", "P06213", 0.7)],
        disease2_targets=[GeneAssociation("APP", "ENSG_APP", "P05067", 0.6)],
        citations={},
    )


# Build a one-edge chain from INSR to APP.
def _single_edge_chain():
    edge = ChainEdge("INSR", "P06213", "APP", "P05067", "activates", "P06213->P05067", 0.7, 0.6)
    return ChainCandidate(edges=[edge], rationale="test rationale")


# Build a two-edge chain with one bridge gene.
def _two_hop_chain():
    edge1 = ChainEdge("INSR", "P06213", "MID", "PM", "activates", "P06213->PM", 0.7, None)
    edge2 = ChainEdge("MID", "PM", "APP", "P05067", "inhibits", "PM->P05067", None, 0.6)
    return ChainCandidate(edges=[edge1, edge2], rationale="bridge rationale")


# Build a chain whose middle hop is a metabolite.
def _metabolite_bridge_chain():
    # INSR --produces--> Cholesterol (metabolite middle hop) --activates--> APP.
    # Endpoints are still real genes (INSR, APP); only the middle node is metabolite-kind.
    edge1 = ChainEdge(
        "INSR",
        "P06213",
        "Cholesterol",
        None,
        "produces",
        "INSR->Cholesterol",
        0.7,
        None,
        source_kind="gene",
        target_kind="metabolite",
        source_hmdb_id=None,
        target_hmdb_id="HMDB0000067",
    )
    edge2 = ChainEdge(
        "Cholesterol",
        None,
        "APP",
        "P05067",
        "activates",
        "Cholesterol->APP",
        None,
        0.6,
        source_kind="metabolite",
        target_kind="gene",
        source_hmdb_id="HMDB0000067",
        target_hmdb_id=None,
    )
    return ChainCandidate(edges=[edge1, edge2], rationale="metabolite bridge rationale")


# Build a three-edge chain with two bridge genes.
def _three_hop_chain():
    # INSR (disease-1 endpoint) -> PRKCA (bridge) -> ARRB1 (bridge) -> APP (disease-2 endpoint): two bridge genes, so
    # exactly one edge (PRKCA-ARRB1) has both ends being bridge genes.
    edge1 = ChainEdge("INSR", "P06213", "PRKCA", "P_PRKCA", "activates", "INSR->PRKCA", 0.7, None)
    edge2 = ChainEdge("PRKCA", "P_PRKCA", "ARRB1", "P_ARRB1", "activates", "PRKCA->ARRB1", None, None)
    edge3 = ChainEdge("ARRB1", "P_ARRB1", "APP", "P05067", "activates", "ARRB1->APP", None, 0.6)
    return ChainCandidate(edges=[edge1, edge2, edge3], rationale="three-hop bridge rationale")


# Wrap chains in a Stage 2 result.
def _stage2_result(chains):
    return Stage2JudgingResult(disease1=DISEASE1, disease2=DISEASE2, chains=chains, candidate_path_count=len(chains), shared_genes=[])


# Empty chains returns empty list.
def test_empty_chains_returns_empty_list():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([])

    result = run_stage3_confirmation(stage1_result, stage2_result, gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}))

    assert result == []


# Single edge chain checks both endpoints against own disease tissues.
def test_single_edge_chain_checks_both_endpoints_against_own_disease_tissues():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_single_edge_chain()])
    gtex = StubGTExClient({"INSR|Pancreas": 10.0, "APP|Brain_Cortex": 8.0})
    gwas = StubGWASCatalogClient({})

    results = run_stage3_confirmation(stage1_result, stage2_result, gtex_client=gtex, gwas_client=gwas)

    assert len(results) == 1
    confirmation = results[0]
    tissues_checked = {(t.gene_symbol, t.tissue) for t in confirmation.tissue_checks}
    # INSR (disease1 endpoint) checked against T2DM's tissues, not AD's
    assert ("INSR", "Pancreas") in tissues_checked
    assert ("INSR", "Brain_Cortex") not in tissues_checked
    # APP (disease2 endpoint) checked against AD's tissues, not T2DM's
    assert ("APP", "Brain_Cortex") in tissues_checked
    assert ("APP", "Pancreas") not in tissues_checked
    assert confirmation.endpoint_checks[0].gene_symbol == "INSR"
    assert confirmation.endpoint_checks[0].disease_name == DISEASE1
    assert confirmation.endpoint_checks[1].gene_symbol == "APP"
    assert confirmation.endpoint_checks[1].disease_name == DISEASE2


# Multi hop chain checks middle gene against both diseases tissues.
def test_multi_hop_chain_checks_middle_gene_against_both_diseases_tissues():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_two_hop_chain()])
    gtex = StubGTExClient({})
    gwas = StubGWASCatalogClient({})

    results = run_stage3_confirmation(stage1_result, stage2_result, gtex_client=gtex, gwas_client=gwas)

    tissues_checked = {(t.gene_symbol, t.tissue) for t in results[0].tissue_checks}
    # MID is a pure bridge gene -- checked against BOTH diseases' tissues
    assert ("MID", "Pancreas") in tissues_checked  # T2DM tissue
    assert ("MID", "Brain_Cortex") in tissues_checked  # AD tissue
    # endpoints are still INSR (disease1) and APP (disease2), not MID
    endpoint_genes = {e.gene_symbol for e in results[0].endpoint_checks}
    assert endpoint_genes == {"INSR", "APP"}


# Metabolite middle hop excluded from tissue checks and listed in chain metabolites.
def test_metabolite_middle_hop_excluded_from_tissue_checks_and_listed_in_chain_metabolites():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_metabolite_bridge_chain()])
    gtex = StubGTExClient({})
    gwas = StubGWASCatalogClient({})

    results = run_stage3_confirmation(stage1_result, stage2_result, gtex_client=gtex, gwas_client=gwas)

    assert len(results) == 1
    confirmation = results[0]
    # the metabolite middle hop is never gene-centric tissue-checked -- only the two real genes are
    checked_genes = {t.gene_symbol for t in confirmation.tissue_checks}
    assert "Cholesterol" not in checked_genes
    assert checked_genes == {"INSR", "APP"}
    # it's surfaced separately instead
    assert confirmation.chain_metabolites == ["Cholesterol"]
    # endpoints are still the two real genes, cross-db checked as usual
    endpoint_genes = {e.gene_symbol for e in confirmation.endpoint_checks}
    assert endpoint_genes == {"INSR", "APP"}


# Endpoint confirmed true when GWAS has evidence.
def test_endpoint_confirmed_true_when_gwas_has_evidence():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_single_edge_chain()])
    gwas_hit = GwasAssociation("INSR", "type 2 diabetes", "EFO_0001360", 1e-15, "GCST001")
    gwas = StubGWASCatalogClient({("INSR", DISEASE1): [gwas_hit]})

    results = run_stage3_confirmation(stage1_result, stage2_result, gtex_client=StubGTExClient({}), gwas_client=gwas)

    insr_check = next(e for e in results[0].endpoint_checks if e.gene_symbol == "INSR")
    assert insr_check.confirmed is True


# Endpoint not confirmed when no GWAS evidence.
def test_endpoint_not_confirmed_when_no_gwas_evidence():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_single_edge_chain()])
    gwas = StubGWASCatalogClient({})

    results = run_stage3_confirmation(stage1_result, stage2_result, gtex_client=StubGTExClient({}), gwas_client=gwas)

    assert all(not e.confirmed for e in results[0].endpoint_checks)


# GWAS client called for both endpoints.
def test_gwas_client_called_for_both_endpoints():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_single_edge_chain()])
    gwas_hit = GwasAssociation("INSR", "type 2 diabetes", "EFO_0001360", 1e-15, "GCST001")
    gwas = StubGWASCatalogClient({("INSR", DISEASE1): [gwas_hit]})

    run_stage3_confirmation(stage1_result, stage2_result, gtex_client=StubGTExClient({}), gwas_client=gwas)

    assert len(gwas.calls) == 2  # both endpoints checked


# Endpoint check flags possibly incomplete from GWAS client.
def test_endpoint_check_flags_possibly_incomplete_from_gwas_client():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_single_edge_chain()])
    gwas = StubGWASCatalogClient({})
    gwas.last_check_rate_limited = True  # simulates the real client's post-call state

    results = run_stage3_confirmation(stage1_result, stage2_result, gtex_client=StubGTExClient({}), gwas_client=gwas)

    assert all(e.possibly_incomplete is True for e in results[0].endpoint_checks)
    assert all(e.confirmed is False for e in results[0].endpoint_checks)  # still no evidence -- just not final


# Endpoint check possibly incomplete defaults to false.
def test_endpoint_check_possibly_incomplete_defaults_to_false():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_single_edge_chain()])
    gwas = StubGWASCatalogClient({})

    results = run_stage3_confirmation(stage1_result, stage2_result, gtex_client=StubGTExClient({}), gwas_client=gwas)

    assert all(e.possibly_incomplete is False for e in results[0].endpoint_checks)


# To dict is JSON serializable.
def test_to_dict_is_json_serializable():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_single_edge_chain()])
    results = run_stage3_confirmation(
        stage1_result, stage2_result, gtex_client=StubGTExClient({"INSR|Pancreas": 5.0}), gwas_client=StubGWASCatalogClient({})
    )
    json.dumps(results[0].to_dict())


# --- Edge-level literature validation ---


# Single edge chain gets no literature checks no bridge edges exist.
def test_single_edge_chain_gets_no_literature_checks_no_bridge_edges_exist():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_single_edge_chain()])
    pubmed = StubPubMedClient(citations_by_gene={})

    results = run_stage3_confirmation(
        stage1_result, stage2_result, gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=pubmed
    )

    assert results[0].edge_literature_checks == []
    assert pubmed.edge_calls == []  # no PubMed query made at all


# Two hop chain gets no literature checks every edge touches an endpoint.
def test_two_hop_chain_gets_no_literature_checks_every_edge_touches_an_endpoint():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_two_hop_chain()])
    pubmed = StubPubMedClient(citations_by_gene={})

    results = run_stage3_confirmation(
        stage1_result, stage2_result, gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=pubmed
    )

    assert results[0].edge_literature_checks == []
    assert pubmed.edge_calls == []


# Three hop chain checks exactly the one bridge to bridge edge.
def test_three_hop_chain_checks_exactly_the_one_bridge_to_bridge_edge():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_three_hop_chain()])
    edge_citation = EdgeCitation(
        pmid="555", title="PRKCA-ARRB1 interaction", full_text_checked=True,
        matched_text="PRKCA directly activates ARRB1 via GPCR desensitization.", both_genes_confirmed=True,
    )
    pubmed = StubPubMedClient(citations_by_gene={}, edge_citations_by_pair={("PRKCA", "ARRB1"): [edge_citation]})

    results = run_stage3_confirmation(
        stage1_result, stage2_result, gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=pubmed
    )

    checks = results[0].edge_literature_checks
    assert len(checks) == 1
    assert checks[0].gene_a == "PRKCA"
    assert checks[0].gene_b == "ARRB1"
    assert checks[0].supported is True
    assert checks[0].any_full_text_checked is True
    assert pubmed.edge_calls == [("PRKCA", "ARRB1", 3)]


# Edge literature check not supported when no confirming citation.
def test_edge_literature_check_not_supported_when_no_confirming_citation():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_three_hop_chain()])
    unconfirmed_citation = EdgeCitation(
        pmid="777", title="A tangentially related paper", full_text_checked=False,
        matched_text="This abstract discusses PRKCA but not ARRB1.", both_genes_confirmed=False,
    )
    pubmed = StubPubMedClient(citations_by_gene={}, edge_citations_by_pair={("PRKCA", "ARRB1"): [unconfirmed_citation]})

    results = run_stage3_confirmation(
        stage1_result, stage2_result, gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=pubmed
    )

    assert results[0].edge_literature_checks[0].supported is False
    assert results[0].edge_literature_checks[0].any_full_text_checked is False


# Metabolite involved edges are skipped from literature checks.
def test_metabolite_involved_edges_are_skipped_from_literature_checks():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_metabolite_bridge_chain()])
    pubmed = StubPubMedClient(citations_by_gene={})

    results = run_stage3_confirmation(
        stage1_result, stage2_result, gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=pubmed
    )

    assert results[0].edge_literature_checks == []
    assert pubmed.edge_calls == []


# Edge literature check to dict is JSON serializable.
def test_edge_literature_check_to_dict_is_json_serializable():
    stage1_result = _stage1_result()
    stage2_result = _stage2_result([_three_hop_chain()])
    edge_citation = EdgeCitation(
        pmid="555", title="t", full_text_checked=True, matched_text="both genes here", both_genes_confirmed=True
    )
    pubmed = StubPubMedClient(citations_by_gene={}, edge_citations_by_pair={("PRKCA", "ARRB1"): [edge_citation]})

    results = run_stage3_confirmation(
        stage1_result, stage2_result, gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=pubmed
    )

    json.dumps(results[0].to_dict())


# --- Independence of the evidence ---

# Build a one-association GWAS result for a gene.
def _gwas_hit(gene, trait="trait"):
    return [GwasAssociation(gene, trait, "EFO_1", 1e-9, "GCST1")]


# Endpoint GWAS checks are flagged as not independent of selection.
def test_endpoint_gwas_checks_are_flagged_as_not_independent_of_selection():
    result = run_stage3_confirmation(
        _stage1_result(), _stage2_result([_two_hop_chain()]),
        gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({("INSR", DISEASE1): _gwas_hit("INSR")}),
        pubmed_client=StubPubMedClient({}),
    )[0]
    assert all(e.independent_of_selection is False for e in result.endpoint_checks)
    assert all(e["independent_of_selection"] is False for e in result.to_dict()["endpoint_checks"])


# Bridge genes are GWAS checked against both diseases and flagged independent.
def test_bridge_genes_are_gwas_checked_against_both_diseases_and_flagged_independent():
    gwas = StubGWASCatalogClient({("PRKCA", DISEASE2): _gwas_hit("PRKCA")})
    result = run_stage3_confirmation(
        _stage1_result(), _stage2_result([_three_hop_chain()]),
        gtex_client=StubGTExClient({}), gwas_client=gwas, pubmed_client=StubPubMedClient({}),
    )[0]
    assert {(c.gene_symbol, c.disease_name) for c in result.bridge_gwas_checks} == {
        ("PRKCA", DISEASE1), ("PRKCA", DISEASE2), ("ARRB1", DISEASE1), ("ARRB1", DISEASE2),
    }
    assert all(c.independent_of_selection for c in result.bridge_gwas_checks)
    assert [c.confirmed for c in result.bridge_gwas_checks if c.gene_symbol == "PRKCA" and c.disease_name == DISEASE2] == [True]
    assert ("PRKCA", DISEASE1) in gwas.calls and ("ARRB1", DISEASE2) in gwas.calls


# Single edge chain has no bridge GWAS checks.
def test_single_edge_chain_has_no_bridge_gwas_checks():
    result = run_stage3_confirmation(
        _stage1_result(), _stage2_result([_single_edge_chain()]),
        gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=StubPubMedClient({}),
    )[0]
    assert result.bridge_gwas_checks == []


# Build a Stage 2 result whose bfs_stats carries a pool of candidate bridge genes.
def _stage2_with_pool(chains, pool):
    r = _stage2_result(chains)
    r.bfs_stats = {"candidate_bridge_genes": pool}
    return r


# Null baseline samples from the pool excluding chain genes and computes rates.
def test_null_baseline_samples_from_the_pool_excluding_chain_genes_and_computes_rates():
    pool = ["PRKCA", "ARRB1"] + [f"G{i}" for i in range(10)]
    # every pool gene expressed in every tissue except G0..G4, which have no GTEx record at all
    expression = {f"G{i}|{t}": 5.0 for i in range(5, 10) for t in
                  ["Pancreas", "Liver", "Muscle_Skeletal", "Adipose_Subcutaneous", "Brain_Cortex", "Brain_Hippocampus"]}
    gwas = StubGWASCatalogClient({("G7", DISEASE1): _gwas_hit("G7"), ("G8", DISEASE2): _gwas_hit("G8")})
    results = run_stage3_confirmation(
        _stage1_result(), _stage2_with_pool([_three_hop_chain()], pool),
        gtex_client=StubGTExClient(expression), gwas_client=gwas, pubmed_client=StubPubMedClient({}),
        null_baseline_size=10,
    )
    null = results[0].null_baseline
    assert null is results[0].to_dict()["null_baseline"] or null == results[0].to_dict()["null_baseline"]
    assert sorted(null["genes"]) == sorted(f"G{i}" for i in range(10))  # chain genes PRKCA/ARRB1/INSR/APP excluded
    assert null["pool_size"] == 10 and null["n_genes"] == 10
    assert null["tissue_checks_with_data"] == 30 and null["tissue_expressed_rate"] == 1.0  # no-record checks excluded
    assert null["gwas"][DISEASE1] == {"genes_checked": 10, "genes_with_hit": 1, "hit_rate": 0.1}
    assert null["gwas"][DISEASE2]["genes_with_hit"] == 1
    assert null["chains"]["bridge_genes"] == ["ARRB1", "PRKCA"]


# Null baseline is reproducible and skippable.
def test_null_baseline_is_reproducible_and_skippable():
    pool = [f"G{i}" for i in range(50)]
    kwargs = dict(gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=StubPubMedClient({}))
    a = run_stage3_confirmation(_stage1_result(), _stage2_with_pool([_two_hop_chain()], pool), null_baseline_size=7, **kwargs)[0]
    b = run_stage3_confirmation(_stage1_result(), _stage2_with_pool([_two_hop_chain()], pool), null_baseline_size=7, **kwargs)[0]
    assert a.null_baseline["genes"] == b.null_baseline["genes"] and len(a.null_baseline["genes"]) == 7
    skipped = run_stage3_confirmation(_stage1_result(), _stage2_with_pool([_two_hop_chain()], pool), null_baseline_size=0, **kwargs)[0]
    assert skipped.null_baseline is None


# No baseline when stage2 result has no bridge pool.
def test_no_baseline_when_stage2_result_has_no_bridge_pool():
    r = run_stage3_confirmation(
        _stage1_result(), _stage2_result([_two_hop_chain()]),
        gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=StubPubMedClient({}),
    )[0]
    assert r.null_baseline is None


# Rate limited null checks are inconclusive not negative.
def test_rate_limited_null_checks_are_inconclusive_not_negative():
    # GWAS stub that reports gene G0 as rate-limited.
    class Limited(StubGWASCatalogClient):
        # Flag the lookup as rate-limited for G0 and return no associations.
        def get_gene_disease_associations(self, gene_symbol, disease_name):
            self.last_check_rate_limited = gene_symbol == "G0"
            return []

    pool = ["G0", "G1"]
    r = run_stage3_confirmation(
        _stage1_result(), _stage2_with_pool([_single_edge_chain()], pool),
        gtex_client=StubGTExClient({}), gwas_client=Limited({}), pubmed_client=StubPubMedClient({}), null_baseline_size=2,
    )[0]
    assert r.null_baseline["gwas_checks_inconclusive"] == 2  # G0 x 2 diseases
    assert r.null_baseline["gwas"][DISEASE1]["genes_checked"] == 1


# Result with baseline is JSON serializable.
def test_result_with_baseline_is_json_serializable():
    r = run_stage3_confirmation(
        _stage1_result(), _stage2_with_pool([_three_hop_chain()], ["A", "B", "C"]),
        gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=StubPubMedClient({}), null_baseline_size=3,
    )
    json.dumps([x.to_dict() for x in r])
