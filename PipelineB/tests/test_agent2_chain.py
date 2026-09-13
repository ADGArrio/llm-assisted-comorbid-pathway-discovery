from PipelineB.agent1_discovery import Agent1DiscoveryResult
from PipelineB.agent2_chain import NoChainFoundError, run_agent2_chain
from PipelineB.omnipath_client import Interaction, StubOmniPathClient
from PipelineB.opentargets_client import GeneAssociation
from PipelineB.pubmed_client import Citation
from common.llm_client import StubLLMClient

DISEASE1 = "Type 2 Diabetes Mellitus"
DISEASE2 = "Alzheimer's Disease"


class RaisingLLMClient:
    """Fails the test if Agent 2 calls the LLM when it shouldn't have
    (i.e. when there was only one shortest candidate path)."""

    def generate(self, prompt, temperature=0.3):
        raise AssertionError("LLM should not be called when only one shortest path exists")


def _agent1_result(disease1_targets, disease2_targets, citations=None):
    return Agent1DiscoveryResult(
        disease1=DISEASE1,
        disease2=DISEASE2,
        disease1_efo_id="EFO_T2DM",
        disease2_efo_id="EFO_AD",
        disease1_targets=disease1_targets,
        disease2_targets=disease2_targets,
        citations=citations or {},
    )


def test_single_shortest_path_skips_llm_entirely():
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("TCF7L2", "ENSG1", "Q9NQB0", 0.71)],
        disease2_targets=[GeneAssociation("APP", "ENSG2", "P05067", 0.44)],
    )
    omnipath = StubOmniPathClient([Interaction("TCF7L2", "Q9NQB0", "APP", "P05067", "inhibits", "Q9NQB0->P05067")])

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())

    assert result.candidate_path_count == 1
    assert len(result.chains) == 1
    assert result.chains[0].rationale is None
    assert result.chains[0].has_unconfirmed_direction is False
    assert result.chains[0].to_dict()["has_unconfirmed_direction"] is False
    edges = result.chains[0].edges
    assert len(edges) == 1
    assert edges[0].source_gene == "TCF7L2"
    assert edges[0].target_gene == "APP"
    assert edges[0].interaction_type == "inhibits"
    assert edges[0].source_disease_assoc_score == 0.71
    assert edges[0].target_disease_assoc_score == 0.44


def test_single_shortest_path_with_unknown_direction_gets_a_caveat_even_without_llm():
    """Per the user's explicit instruction: an OmniPath edge with no
    curated activate/inhibit flag must never ship silently, even in the
    single-candidate case that normally bypasses the LLM entirely
    because "there's nothing to adjudicate." The caveat is a mandatory,
    code-level backstop -- it does not depend on the LLM ever being
    consulted."""
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("TCF7L2", "ENSG1", "Q9NQB0", 0.71)],
        disease2_targets=[GeneAssociation("APP", "ENSG2", "P05067", 0.44)],
    )
    omnipath = StubOmniPathClient([Interaction("TCF7L2", "Q9NQB0", "APP", "P05067", "unknown", "Q9NQB0->P05067")])

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())

    assert len(result.chains) == 1
    chain = result.chains[0]
    assert chain.has_unconfirmed_direction is True
    assert chain.to_dict()["has_unconfirmed_direction"] is True
    assert chain.rationale is not None
    assert "not established" in chain.rationale


def _two_path_setup():
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6), GeneAssociation("B", "ENSGB", "PB", 0.5)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8), GeneAssociation("Y", "ENSGY", "PY", 0.7)],
    )
    omnipath = StubOmniPathClient(
        [
            Interaction("A", "PA", "X", "PX", "activates", "PA->PX"),
            Interaction("B", "PB", "Y", "PY", "inhibits", "PB->PY"),
        ]
    )
    return agent1_result, omnipath


def test_llm_can_select_a_single_candidate_among_several():
    agent1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response="2: Because B-Y is the more established pathway.")

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.candidate_path_count == 2
    assert len(result.chains) == 1
    assert result.chains[0].rationale == "Because B-Y is the more established pathway."
    assert result.chains[0].edges[0].source_gene == "B"


def test_llm_can_select_multiple_plausible_candidates():
    agent1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response=(
            "1: Supported by direct literature evidence for A-X.\n"
            "2: Also plausible based on known B-Y signaling."
        )
    )

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 2
    sources = {c.edges[0].source_gene for c in result.chains}
    assert sources == {"A", "B"}
    rationales = {c.rationale for c in result.chains}
    assert "Supported by direct literature evidence for A-X." in rationales
    assert "Also plausible based on known B-Y signaling." in rationales


def test_llm_endorsed_chain_with_unknown_edge_gets_caveat_appended():
    """Same backstop as the single-candidate case above, but exercised
    through the multi-candidate LLM-selection path: the LLM's own
    rationale is kept, with the mandatory caveat appended -- the code
    doesn't trust the LLM to have already covered it, even though the
    prompt now also asks it to."""
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6), GeneAssociation("B", "ENSGB", "PB", 0.5)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8), GeneAssociation("Y", "ENSGY", "PY", 0.7)],
    )
    omnipath = StubOmniPathClient(
        [
            Interaction("A", "PA", "X", "PX", "unknown", "PA->PX"),
            Interaction("B", "PB", "Y", "PY", "inhibits", "PB->PY"),
        ]
    )
    llm = StubLLMClient(fixed_response="1: Plausible based on known A-X signaling despite the unclear direction.")

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    chain = result.chains[0]
    assert chain.has_unconfirmed_direction is True
    assert chain.rationale.startswith("Plausible based on known A-X signaling despite the unclear direction.")
    assert "not established" in chain.rationale


def test_llm_selecting_none_returns_empty_chains_without_raising():
    agent1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response="NONE")

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.candidate_path_count == 2
    assert result.chains == []


def test_llm_response_with_no_parseable_lines_and_not_none_raises():
    agent1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response="I refuse to answer in the requested format.")

    try:
        run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "no parseable" in str(e)


def test_llm_response_ignores_out_of_range_lines_but_keeps_valid_ones():
    agent1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response="99: out of range, should be dropped\n1: This one is valid.")

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    assert result.chains[0].rationale == "This one is valid."


def test_citations_for_path_genes_are_included_in_the_llm_prompt():
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6), GeneAssociation("B", "ENSGB", "PB", 0.5)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8), GeneAssociation("Y", "ENSGY", "PY", 0.7)],
        citations={"A": [Citation("111", "A causes X via some mechanism")]},
    )
    omnipath = StubOmniPathClient(
        [
            Interaction("A", "PA", "X", "PX", "activates", "PA->PX"),
            Interaction("B", "PB", "Y", "PY", "inhibits", "PB->PY"),
        ]
    )
    llm = StubLLMClient(fixed_response="1: matches the cited literature.")

    run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert "A causes X via some mechanism" in llm.last_prompt
    assert "1. A --activates--> X" in llm.last_prompt


def test_unknown_direction_edge_is_marked_in_the_llm_prompt():
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6), GeneAssociation("B", "ENSGB", "PB", 0.5)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8), GeneAssociation("Y", "ENSGY", "PY", 0.7)],
    )
    omnipath = StubOmniPathClient(
        [
            Interaction("A", "PA", "X", "PX", "unknown", "PA->PX"),
            Interaction("B", "PB", "Y", "PY", "inhibits", "PB->PY"),
        ]
    )
    llm = StubLLMClient(fixed_response="2: B-Y is well established.")

    run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert "A --UNKNOWN DIRECTION--> X" in llm.last_prompt
    assert "B --inhibits--> Y" in llm.last_prompt


def test_no_path_raises_no_chain_found_error():
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8)],
    )
    omnipath = StubOmniPathClient([Interaction("A", "PA", "Z", "PZ", "activates", "PA->PZ")])

    try:
        run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())
        assert False, "expected NoChainFoundError"
    except NoChainFoundError:
        pass


def test_shared_gene_reported_but_not_treated_as_a_zero_edge_path():
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6), GeneAssociation("APOE", "ENSGE", "PE", 0.5)],
        disease2_targets=[GeneAssociation("APOE", "ENSGE", "PE", 0.55)],
    )
    omnipath = StubOmniPathClient([Interaction("A", "PA", "APOE", "PE", "activates", "PA->PE")])

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())

    assert result.shared_genes == ["APOE"]
    assert result.candidate_path_count == 1
    assert result.chains[0].edges[0].target_gene == "APOE"


def test_multi_hop_path_found_when_no_direct_edge_exists():
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8)],
    )
    omnipath = StubOmniPathClient(
        [
            Interaction("A", "PA", "MID", "PM", "activates", "PA->PM"),
            Interaction("MID", "PM", "X", "PX", "inhibits", "PM->PX"),
        ]
    )

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())

    assert result.candidate_path_count == 1
    edges = result.chains[0].edges
    assert [e.source_gene for e in edges] == ["A", "MID"]
    assert [e.target_gene for e in edges] == ["MID", "X"]
    assert edges[0].target_disease_assoc_score is None  # MID has no disease2 score
    assert edges[1].source_disease_assoc_score is None  # MID has no disease1 score


def test_default_max_hops_is_five():
    from PipelineB.agent2_chain import DEFAULT_MAX_HOPS

    assert DEFAULT_MAX_HOPS == 5


def test_to_dict_is_json_serializable():
    import json

    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("TCF7L2", "ENSG1", "Q9NQB0", 0.71)],
        disease2_targets=[GeneAssociation("APP", "ENSG2", "P05067", 0.44)],
    )
    omnipath = StubOmniPathClient([Interaction("TCF7L2", "Q9NQB0", "APP", "P05067", "inhibits", "Q9NQB0->P05067")])
    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())
    json.dumps(result.to_dict())
