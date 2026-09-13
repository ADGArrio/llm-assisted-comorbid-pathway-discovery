from PipelineB.agent1_discovery import Agent1DiscoveryResult
from PipelineB.agent2_chain import Agent2ChainResult, ChainCandidate, ChainEdge
from PipelineB.agent3_confirmation import run_agent3_confirmation
from PipelineB.crossdb_client import DisgenetAssociation, GwasAssociation, StubDisGeNETClient, StubGWASCatalogClient
from PipelineB.opentargets_client import GeneAssociation
from PipelineB.tissue_client import StubGTExClient, StubHPAClient

DISEASE1 = "Type 2 Diabetes Mellitus"
DISEASE2 = "Alzheimer's Disease"


def _agent1_result():
    return Agent1DiscoveryResult(
        disease1=DISEASE1,
        disease2=DISEASE2,
        disease1_efo_id="EFO_T2DM",
        disease2_efo_id="EFO_AD",
        disease1_targets=[GeneAssociation("INSR", "ENSG_INSR", "P06213", 0.7)],
        disease2_targets=[GeneAssociation("APP", "ENSG_APP", "P05067", 0.6)],
        citations={},
    )


def _single_edge_chain():
    edge = ChainEdge("INSR", "P06213", "APP", "P05067", "activates", "P06213->P05067", 0.7, 0.6)
    return ChainCandidate(edges=[edge], rationale="test rationale")


def _two_hop_chain():
    edge1 = ChainEdge("INSR", "P06213", "MID", "PM", "activates", "P06213->PM", 0.7, None)
    edge2 = ChainEdge("MID", "PM", "APP", "P05067", "inhibits", "PM->P05067", None, 0.6)
    return ChainCandidate(edges=[edge1, edge2], rationale="bridge rationale")


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


def _agent2_result(chains):
    return Agent2ChainResult(disease1=DISEASE1, disease2=DISEASE2, chains=chains, candidate_path_count=len(chains), shared_genes=[])


def test_empty_chains_returns_empty_list():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([])

    result = run_agent3_confirmation(
        agent1_result, agent2_result, gtex_client=StubGTExClient({}), hpa_client=StubHPAClient({}), gwas_client=StubGWASCatalogClient({})
    )

    assert result == []


def test_single_edge_chain_checks_both_endpoints_against_own_disease_tissues():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_single_edge_chain()])
    gtex = StubGTExClient({"INSR|Pancreas": 10.0, "APP|Brain_Cortex": 8.0})
    hpa = StubHPAClient({})
    gwas = StubGWASCatalogClient({})

    results = run_agent3_confirmation(agent1_result, agent2_result, gtex_client=gtex, hpa_client=hpa, gwas_client=gwas)

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


def test_multi_hop_chain_checks_middle_gene_against_both_diseases_tissues():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_two_hop_chain()])
    gtex = StubGTExClient({})
    hpa = StubHPAClient({})
    gwas = StubGWASCatalogClient({})

    results = run_agent3_confirmation(agent1_result, agent2_result, gtex_client=gtex, hpa_client=hpa, gwas_client=gwas)

    tissues_checked = {(t.gene_symbol, t.tissue) for t in results[0].tissue_checks}
    # MID is a pure bridge gene -- checked against BOTH diseases' tissues
    assert ("MID", "Pancreas") in tissues_checked  # T2DM tissue
    assert ("MID", "Brain_Cortex") in tissues_checked  # AD tissue
    # endpoints are still INSR (disease1) and APP (disease2), not MID
    endpoint_genes = {e.gene_symbol for e in results[0].endpoint_checks}
    assert endpoint_genes == {"INSR", "APP"}


def test_metabolite_middle_hop_excluded_from_tissue_checks_and_listed_in_chain_metabolites():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_metabolite_bridge_chain()])
    gtex = StubGTExClient({})
    hpa = StubHPAClient({})
    gwas = StubGWASCatalogClient({})

    results = run_agent3_confirmation(agent1_result, agent2_result, gtex_client=gtex, hpa_client=hpa, gwas_client=gwas)

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


def test_endpoint_confirmed_true_when_gwas_has_evidence_even_without_disgenet():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_single_edge_chain()])
    gwas_hit = GwasAssociation("INSR", "type 2 diabetes", "EFO_0001360", 1e-15, "GCST001")
    gwas = StubGWASCatalogClient({("INSR", DISEASE1): [gwas_hit]})

    results = run_agent3_confirmation(
        agent1_result, agent2_result, gtex_client=StubGTExClient({}), hpa_client=StubHPAClient({}), gwas_client=gwas, disgenet_client=None
    )

    insr_check = next(e for e in results[0].endpoint_checks if e.gene_symbol == "INSR")
    assert insr_check.confirmed is True
    assert insr_check.disgenet_association is None


def test_endpoint_confirmed_true_when_only_disgenet_has_evidence():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_single_edge_chain()])
    disgenet_hit = DisgenetAssociation("APP", DISEASE2, 0.9, 200)
    disgenet = StubDisGeNETClient({("APP", DISEASE2): disgenet_hit})
    gwas = StubGWASCatalogClient({})  # no GWAS hits at all

    results = run_agent3_confirmation(
        agent1_result,
        agent2_result,
        gtex_client=StubGTExClient({}),
        hpa_client=StubHPAClient({}),
        gwas_client=gwas,
        disgenet_client=disgenet,
    )

    app_check = next(e for e in results[0].endpoint_checks if e.gene_symbol == "APP")
    assert app_check.confirmed is True
    assert app_check.gwas_associations == []


def test_endpoint_not_confirmed_when_neither_source_has_evidence():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_single_edge_chain()])
    gwas = StubGWASCatalogClient({})
    disgenet = StubDisGeNETClient({})

    results = run_agent3_confirmation(
        agent1_result,
        agent2_result,
        gtex_client=StubGTExClient({}),
        hpa_client=StubHPAClient({}),
        gwas_client=gwas,
        disgenet_client=disgenet,
    )

    assert all(not e.confirmed for e in results[0].endpoint_checks)


def test_disgenet_client_none_still_runs_gwas_only():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_single_edge_chain()])
    gwas_hit = GwasAssociation("INSR", "type 2 diabetes", "EFO_0001360", 1e-15, "GCST001")
    gwas = StubGWASCatalogClient({("INSR", DISEASE1): [gwas_hit]})

    results = run_agent3_confirmation(
        agent1_result, agent2_result, gtex_client=StubGTExClient({}), hpa_client=StubHPAClient({}), gwas_client=gwas, disgenet_client=None
    )

    assert len(gwas.calls) == 2  # both endpoints checked
    assert all(e.disgenet_association is None for e in results[0].endpoint_checks)


def test_to_dict_is_json_serializable():
    import json

    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_single_edge_chain()])
    results = run_agent3_confirmation(
        agent1_result,
        agent2_result,
        gtex_client=StubGTExClient({"INSR|Pancreas": 5.0}),
        hpa_client=StubHPAClient({}),
        gwas_client=StubGWASCatalogClient({}),
    )
    json.dumps(results[0].to_dict())
