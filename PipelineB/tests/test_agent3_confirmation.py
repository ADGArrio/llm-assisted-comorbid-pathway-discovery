import json

from PipelineB.agent1_discovery import Agent1DiscoveryResult
from PipelineB.agent2_chain import Agent2ChainResult, ChainCandidate, ChainEdge
from PipelineB.agent3_confirmation import run_agent3_confirmation
from PipelineB.crossdb_client import GwasAssociation, StubGWASCatalogClient
from PipelineB.opentargets_client import GeneAssociation
from PipelineB.pubmed_client import EdgeCitation, StubPubMedClient
from PipelineB.tissue_client import StubGTExClient

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


def _three_hop_chain():
    # INSR (disease1 endpoint) -> PRKCA (bridge) -> ARRB1 (bridge) -> APP (disease2 endpoint).
    # Modeled on the real T2DM->CAD Pathway 2 chain (INSR->PRKCA->ARRB1->PDE4D) -- two bridge
    # genes, so exactly one edge (PRKCA-ARRB1) has both ends being bridge genes. See
    # agent3_confirmation.py's module docstring, "Edge-level literature validation" note.
    edge1 = ChainEdge("INSR", "P06213", "PRKCA", "P_PRKCA", "activates", "INSR->PRKCA", 0.7, None)
    edge2 = ChainEdge("PRKCA", "P_PRKCA", "ARRB1", "P_ARRB1", "activates", "PRKCA->ARRB1", None, None)
    edge3 = ChainEdge("ARRB1", "P_ARRB1", "APP", "P05067", "activates", "ARRB1->APP", None, 0.6)
    return ChainCandidate(edges=[edge1, edge2, edge3], rationale="three-hop bridge rationale")


def _agent2_result(chains):
    return Agent2ChainResult(disease1=DISEASE1, disease2=DISEASE2, chains=chains, candidate_path_count=len(chains), shared_genes=[])


def test_empty_chains_returns_empty_list():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([])

    result = run_agent3_confirmation(agent1_result, agent2_result, gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}))

    assert result == []


def test_single_edge_chain_checks_both_endpoints_against_own_disease_tissues():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_single_edge_chain()])
    gtex = StubGTExClient({"INSR|Pancreas": 10.0, "APP|Brain_Cortex": 8.0})
    gwas = StubGWASCatalogClient({})

    results = run_agent3_confirmation(agent1_result, agent2_result, gtex_client=gtex, gwas_client=gwas)

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
    gwas = StubGWASCatalogClient({})

    results = run_agent3_confirmation(agent1_result, agent2_result, gtex_client=gtex, gwas_client=gwas)

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
    gwas = StubGWASCatalogClient({})

    results = run_agent3_confirmation(agent1_result, agent2_result, gtex_client=gtex, gwas_client=gwas)

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


def test_endpoint_confirmed_true_when_gwas_has_evidence():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_single_edge_chain()])
    gwas_hit = GwasAssociation("INSR", "type 2 diabetes", "EFO_0001360", 1e-15, "GCST001")
    gwas = StubGWASCatalogClient({("INSR", DISEASE1): [gwas_hit]})

    results = run_agent3_confirmation(agent1_result, agent2_result, gtex_client=StubGTExClient({}), gwas_client=gwas)

    insr_check = next(e for e in results[0].endpoint_checks if e.gene_symbol == "INSR")
    assert insr_check.confirmed is True


def test_endpoint_not_confirmed_when_no_gwas_evidence():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_single_edge_chain()])
    gwas = StubGWASCatalogClient({})

    results = run_agent3_confirmation(agent1_result, agent2_result, gtex_client=StubGTExClient({}), gwas_client=gwas)

    assert all(not e.confirmed for e in results[0].endpoint_checks)


def test_gwas_client_called_for_both_endpoints():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_single_edge_chain()])
    gwas_hit = GwasAssociation("INSR", "type 2 diabetes", "EFO_0001360", 1e-15, "GCST001")
    gwas = StubGWASCatalogClient({("INSR", DISEASE1): [gwas_hit]})

    run_agent3_confirmation(agent1_result, agent2_result, gtex_client=StubGTExClient({}), gwas_client=gwas)

    assert len(gwas.calls) == 2  # both endpoints checked


def test_endpoint_check_flags_possibly_incomplete_from_gwas_client():
    """Regression test for crossdb_client.py's bug-fix #3: an endpoint check whose GWAS Catalog
    lookup was rate-limited before it could finish must be flagged possibly_incomplete=True, so
    'no hit' downstream (main.py's readable report, the saved JSON) isn't misread as a confirmed
    negative -- see agent3_confirmation.py's EndpointCrossDbCheck docstring."""
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_single_edge_chain()])
    gwas = StubGWASCatalogClient({})
    gwas.last_check_rate_limited = True  # simulates the real client's post-call state

    results = run_agent3_confirmation(agent1_result, agent2_result, gtex_client=StubGTExClient({}), gwas_client=gwas)

    assert all(e.possibly_incomplete is True for e in results[0].endpoint_checks)
    assert all(e.confirmed is False for e in results[0].endpoint_checks)  # still no evidence -- just not final


def test_endpoint_check_possibly_incomplete_defaults_to_false():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_single_edge_chain()])
    gwas = StubGWASCatalogClient({})

    results = run_agent3_confirmation(agent1_result, agent2_result, gtex_client=StubGTExClient({}), gwas_client=gwas)

    assert all(e.possibly_incomplete is False for e in results[0].endpoint_checks)


def test_to_dict_is_json_serializable():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_single_edge_chain()])
    results = run_agent3_confirmation(
        agent1_result, agent2_result, gtex_client=StubGTExClient({"INSR|Pancreas": 5.0}), gwas_client=StubGWASCatalogClient({})
    )
    json.dumps(results[0].to_dict())


# --- Edge-level literature validation (added 2026-09-27) ---
# See module docstring's "Edge-level literature validation" note and
# claude/committee_feedback_round2_coscientist.md section 3a.


def test_single_edge_chain_gets_no_literature_checks_no_bridge_edges_exist():
    """A 1-hop chain has no edge with both ends being bridge genes (its one edge's both ends ARE
    the two endpoints) -- zero literature checks is the correct, expected scoping outcome, not a
    bug. See the module docstring's stated real consequence of this scoping decision."""
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_single_edge_chain()])
    pubmed = StubPubMedClient(citations_by_gene={})

    results = run_agent3_confirmation(
        agent1_result, agent2_result, gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=pubmed
    )

    assert results[0].edge_literature_checks == []
    assert pubmed.edge_calls == []  # no PubMed query made at all


def test_two_hop_chain_gets_no_literature_checks_every_edge_touches_an_endpoint():
    """A 2-hop chain has exactly one bridge gene (MID) -- both its edges touch an endpoint
    (INSR-MID, MID-APP), so neither edge has BOTH ends being bridge genes. Still zero checks,
    same scoping rationale as the 1-hop case above."""
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_two_hop_chain()])
    pubmed = StubPubMedClient(citations_by_gene={})

    results = run_agent3_confirmation(
        agent1_result, agent2_result, gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=pubmed
    )

    assert results[0].edge_literature_checks == []
    assert pubmed.edge_calls == []


def test_three_hop_chain_checks_exactly_the_one_bridge_to_bridge_edge():
    """INSR->PRKCA->ARRB1->APP: PRKCA and ARRB1 are both bridge genes, so the PRKCA-ARRB1 edge
    (and only that edge) gets a literature check -- INSR-PRKCA and ARRB1-APP each touch an
    endpoint and are skipped, per the confirmed scoping decision."""
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_three_hop_chain()])
    edge_citation = EdgeCitation(
        pmid="555", title="PRKCA-ARRB1 interaction", full_text_checked=True,
        matched_text="PRKCA directly activates ARRB1 via GPCR desensitization.", both_genes_confirmed=True,
    )
    pubmed = StubPubMedClient(citations_by_gene={}, edge_citations_by_pair={("PRKCA", "ARRB1"): [edge_citation]})

    results = run_agent3_confirmation(
        agent1_result, agent2_result, gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=pubmed
    )

    checks = results[0].edge_literature_checks
    assert len(checks) == 1
    assert checks[0].gene_a == "PRKCA"
    assert checks[0].gene_b == "ARRB1"
    assert checks[0].supported is True
    assert checks[0].any_full_text_checked is True
    assert pubmed.edge_calls == [("PRKCA", "ARRB1", 3)]


def test_edge_literature_check_not_supported_when_no_confirming_citation():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_three_hop_chain()])
    unconfirmed_citation = EdgeCitation(
        pmid="777", title="A tangentially related paper", full_text_checked=False,
        matched_text="This abstract discusses PRKCA but not ARRB1.", both_genes_confirmed=False,
    )
    pubmed = StubPubMedClient(citations_by_gene={}, edge_citations_by_pair={("PRKCA", "ARRB1"): [unconfirmed_citation]})

    results = run_agent3_confirmation(
        agent1_result, agent2_result, gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=pubmed
    )

    assert results[0].edge_literature_checks[0].supported is False
    assert results[0].edge_literature_checks[0].any_full_text_checked is False


def test_metabolite_involved_edges_are_skipped_from_literature_checks():
    """An edge with a metabolite end has no gene-pair literature search to run, even if the gene
    end happens to be a bridge gene -- see _edge_literature_checks' gene-kind check."""
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_metabolite_bridge_chain()])
    pubmed = StubPubMedClient(citations_by_gene={})

    results = run_agent3_confirmation(
        agent1_result, agent2_result, gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=pubmed
    )

    assert results[0].edge_literature_checks == []
    assert pubmed.edge_calls == []


def test_edge_literature_check_to_dict_is_json_serializable():
    agent1_result = _agent1_result()
    agent2_result = _agent2_result([_three_hop_chain()])
    edge_citation = EdgeCitation(
        pmid="555", title="t", full_text_checked=True, matched_text="both genes here", both_genes_confirmed=True
    )
    pubmed = StubPubMedClient(citations_by_gene={}, edge_citations_by_pair={("PRKCA", "ARRB1"): [edge_citation]})

    results = run_agent3_confirmation(
        agent1_result, agent2_result, gtex_client=StubGTExClient({}), gwas_client=StubGWASCatalogClient({}), pubmed_client=pubmed
    )

    json.dumps(results[0].to_dict())
