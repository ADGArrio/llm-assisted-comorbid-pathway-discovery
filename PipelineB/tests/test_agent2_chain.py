from PipelineB.agent1_discovery import Agent1DiscoveryResult
from PipelineB.agent2_chain import NoChainFoundError, run_agent2_chain
from PipelineB.metalinks_client import MetaboliteEdge, StubMetalinksClient
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
    assert edges[0].source_kind == "gene"
    assert edges[0].target_kind == "gene"


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
    llm = StubLLMClient(fixed_response="2: PLAUSIBLE - Because B-Y is the more established pathway.")

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.candidate_path_count == 2
    assert len(result.chains) == 1
    assert result.chains[0].rationale == "Because B-Y is the more established pathway."
    assert result.chains[0].edges[0].source_gene == "B"


def test_llm_can_select_multiple_plausible_candidates():
    agent1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response=(
            "1: PLAUSIBLE - Supported by direct literature evidence for A-X.\n"
            "2: PLAUSIBLE - Also plausible based on known B-Y signaling."
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
    llm = StubLLMClient(
        fixed_response="1: PLAUSIBLE - Plausible based on known A-X signaling despite the unclear direction."
    )

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    chain = result.chains[0]
    assert chain.has_unconfirmed_direction is True
    assert chain.rationale.startswith("Plausible based on known A-X signaling despite the unclear direction.")
    assert "not established" in chain.rationale


def test_llm_selecting_bare_none_returns_empty_chains_with_no_reason():
    """A bare "NONE" with no reason text (an older or non-compliant LLM
    response) is tolerated, not treated as a formatting error --
    rejection_reason is just None in that case."""
    agent1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response="NONE")

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.candidate_path_count == 2
    assert result.chains == []
    assert result.rejection_reason is None


def test_llm_none_response_with_reason_is_captured():
    agent1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response="NONE: neither candidate names a specific enough mechanism for every step."
    )

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.chains == []
    assert result.rejection_reason == "neither candidate names a specific enough mechanism for every step."


def test_rejection_reason_is_none_when_a_chain_is_endorsed():
    agent1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response="1: PLAUSIBLE - Supported by direct literature evidence for A-X.")

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    assert result.rejection_reason is None


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
    llm = StubLLMClient(
        fixed_response="99: PLAUSIBLE - out of range, should be dropped\n1: PLAUSIBLE - This one is valid."
    )

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    assert result.chains[0].rationale == "This one is valid."


def test_candidate_discussed_without_a_verdict_word_is_not_endorsed():
    """Regression test for a real bug, fixed 2026-09-14 (see agent2_chain.py's
    module docstring, "Explicit per-candidate verdicts required"): the model
    used to write one line per candidate whether or not it judged it
    plausible, and the old parser treated any "<number>: <text>" line as an
    endorsement -- so a candidate whose own sentence said the mechanism
    "isn't specifically detailed" got endorsed anyway. A line with no
    explicit PLAUSIBLE/NOT PLAUSIBLE verdict word must not count."""
    agent1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response=(
            "1: The mechanism for A activating X is not specifically detailed in the provided "
            "literature or general knowledge as a direct, named biochemical pathway.\n"
            "2: PLAUSIBLE - B directly phosphorylates Y, activating its kinase domain."
        )
    )

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    assert result.chains[0].edges[0].source_gene == "B"


def test_llm_marks_some_candidates_not_plausible_only_plausible_ones_are_endorsed():
    agent1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response=(
            "1: NOT PLAUSIBLE - no specific mechanism is established for A-X.\n"
            "2: PLAUSIBLE - B directly phosphorylates Y."
        )
    )

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    assert result.chains[0].edges[0].source_gene == "B"
    assert result.rejection_reason is None  # something WAS endorsed -- no blanket rejection reason


def test_all_candidates_marked_not_plausible_produces_a_joined_rejection_reason():
    agent1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response=(
            "1: NOT PLAUSIBLE - no specific mechanism is established for A-X.\n"
            "2: NOT PLAUSIBLE - no specific mechanism is established for B-Y."
        )
    )

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.chains == []
    assert "candidate 1" in result.rejection_reason
    assert "candidate 2" in result.rejection_reason
    assert "A-X" in result.rejection_reason
    assert "B-Y" in result.rejection_reason


def test_verdict_lines_using_a_period_after_the_number_are_still_parsed():
    """Regression test for a real crash, fixed 2026-09-14 (see agent2_chain.py's
    module docstring, "Verdict-line separator widened"): a real run's LLM wrote
    "1. NOT PLAUSIBLE - ..." (an ordinary numbered-list period) instead of the
    prompt's own "1: NOT PLAUSIBLE - ..." example, on all 10 of its lines, and
    the old regex -- which required a literal colon -- matched none of them,
    raising RuntimeError instead of correctly reporting a blanket rejection.
    This response (verbatim shape, trimmed to two candidates) must now parse
    as two NOT PLAUSIBLE verdicts, not raise."""
    agent1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response=(
            "1. NOT PLAUSIBLE - The interaction between A and X lacks a specific, named "
            "mechanism beyond a general activation relationship.\n"
            "2. NOT PLAUSIBLE - The interaction between B and Y lacks a specific, named "
            "mechanism beyond a general activation relationship."
        )
    )

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.chains == []
    assert "candidate 1" in result.rejection_reason
    assert "candidate 2" in result.rejection_reason


def test_verdict_lines_using_a_right_paren_after_the_number_are_still_parsed():
    agent1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response=(
            "1) PLAUSIBLE - A directly activates X via a well-characterized cascade.\n"
            "2) NOT PLAUSIBLE - no specific mechanism established for B-Y."
        )
    )

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    assert result.chains[0].edges[0].source_gene == "A"


def test_bfs_does_not_stop_exploring_once_max_paths_shallow_hits_are_found():
    """Regression test for a real bug, fixed 2026-09-14 (see agent2_chain.py's
    module docstring, "max_paths cutting off exploration too early"): on a real
    run, 10 distinct depth-1 goal-gene hits filled max_candidate_paths (10)
    before depth 2 was ever reached, so a deeper, more specific candidate never
    got a chance to be found at all -- even after the goal-gene-expansion fix
    that was supposed to let the search continue past a hit. Here 11 distinct
    depth-1 hits exist (one more than the default cap) alongside a single
    depth-2 hit reachable only through a non-goal bridge gene; the depth-2
    candidate must still be found and must survive the max_paths trim (which
    prefers longer/deeper candidates over shallow ones when there are too
    many)."""
    shallow_goals = [GeneAssociation(f"G{i}", f"ENSG{i}", f"P{i}", 0.5) for i in range(1, 12)]  # 11 of them
    deep_goal = GeneAssociation("DEEP", "ENSG_DEEP", "P_DEEP", 0.9)
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6)],
        disease2_targets=shallow_goals + [deep_goal],
    )
    shallow_interactions = [
        Interaction("A", "PA", f"G{i}", f"P{i}", "activates", f"PA->P{i}") for i in range(1, 12)
    ]
    bridge_interactions = [
        Interaction("A", "PA", "MID", "P_MID", "activates", "PA->P_MID"),
        Interaction("MID", "P_MID", "DEEP", "P_DEEP", "inhibits", "P_MID->P_DEEP"),
    ]
    omnipath = StubOmniPathClient(shallow_interactions + bridge_interactions)
    llm = StubLLMClient(fixed_response="NONE: none of these have a specific enough mechanism.")

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.candidate_path_count == 10  # trimmed down from 12 real candidates found
    assert "MID --inhibits--> DEEP" in llm.last_prompt  # the depth-2 candidate survived the trim


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
    llm = StubLLMClient(fixed_response="1: PLAUSIBLE - matches the cited literature.")

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
    llm = StubLLMClient(fixed_response="2: PLAUSIBLE - B-Y is well established.")

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


def test_bfs_continues_past_first_successful_depth_to_find_other_goals_deeper():
    """Regression test for a real bug, fixed 2026-09-14 (see
    agent2_chain.py's module docstring, "BFS depth-exhaustion bug"): the
    search used to stop the moment ANY goal gene was reached, even at
    depth 1, so a different goal gene only reachable at depth 2+ was
    silently never found -- regardless of max_hops. Here A->X is a real
    depth-1 edge, and B->MID->Y is a real depth-2 path to a *different*
    goal gene; both must show up as candidates."""
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6), GeneAssociation("B", "ENSGB", "PB", 0.5)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8), GeneAssociation("Y", "ENSGY", "PY", 0.7)],
    )
    omnipath = StubOmniPathClient(
        [
            Interaction("A", "PA", "X", "PX", "activates", "PA->PX"),  # depth 1: reaches goal X
            Interaction("B", "PB", "MID", "PM", "activates", "PB->PM"),  # depth 1: bridge, not a goal
            Interaction("MID", "PM", "Y", "PY", "inhibits", "PM->PY"),  # depth 2: reaches goal Y
        ]
    )
    llm = StubLLMClient(fixed_response="1: PLAUSIBLE - A directly activates X via a well-characterized cascade.")

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.candidate_path_count == 2
    assert "1. A --activates--> X" in llm.last_prompt
    assert "B --activates--> MID -> MID --inhibits--> Y" in llm.last_prompt


def test_bfs_does_not_find_the_same_goal_gene_twice_at_different_depths():
    """A goal gene reached at depth 1 must not also show up as a second,
    longer candidate if a deeper route to that same gene also exists --
    found_targets should keep only its first (shortest) path."""
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8)],
    )
    omnipath = StubOmniPathClient(
        [
            Interaction("A", "PA", "X", "PX", "activates", "PA->PX"),  # depth 1: direct hit on X
            Interaction("A", "PA", "MID", "PM", "activates", "PA->PM"),  # depth 1: bridge
            Interaction("MID", "PM", "X", "PX", "inhibits", "PM->PX"),  # depth 2: a LONGER route to the same X
        ]
    )

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())

    assert result.candidate_path_count == 1
    assert len(result.chains[0].edges) == 1


def test_bfs_keeps_searching_past_a_goal_gene_for_a_more_specific_one_deeper():
    """Regression test for a real bug, fixed 2026-09-14 (see agent2_chain.py's
    module docstring, "Goal-gene expansion"): a gene can independently clear
    Open Targets' disease2 threshold (a "hit") while being a weak, generic,
    or upstream player -- with a much more specific, better-known mechanism
    sitting one or two hops further down the exact same real chain. Modeled
    directly on a real T2DM->AD run: INSR->PIK3R1 is a genuine 1-hop
    candidate and PIK3R1 independently clears the AD threshold, but the
    textbook continuation PIK3R1->AKT1->GSK3B->MAPT (MAPT = tau, also an AD
    gene) must still be found by continuing the search past PIK3R1, not
    abandoned just because PIK3R1 alone already counted as a hit."""
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("INSR", "ENSG_INSR", "P_INSR", 0.9)],
        disease2_targets=[
            GeneAssociation("PIK3R1", "ENSG_PIK3R1", "P_PIK3R1", 0.3),
            GeneAssociation("MAPT", "ENSG_MAPT", "P_MAPT", 0.85),
        ],
    )
    omnipath = StubOmniPathClient(
        [
            Interaction("INSR", "P_INSR", "PIK3R1", "P_PIK3R1", "activates", "INSR->PIK3R1"),
            Interaction("PIK3R1", "P_PIK3R1", "AKT1", "P_AKT1", "activates", "PIK3R1->AKT1"),
            Interaction("AKT1", "P_AKT1", "GSK3B", "P_GSK3B", "inhibits", "AKT1->GSK3B"),
            Interaction("GSK3B", "P_GSK3B", "MAPT", "P_MAPT", "activates", "GSK3B->MAPT"),
        ]
    )
    llm = StubLLMClient(
        fixed_response=(
            "1: PLAUSIBLE - INSR directly activates PIK3R1, the p85 regulatory subunit of PI3K.\n"
            "2: PLAUSIBLE - The well-established insulin-signaling cascade through AKT1 and GSK3B "
            "phosphorylating tau."
        )
    )

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.candidate_path_count == 2
    lengths = sorted(len(c.edges) for c in result.chains)
    assert lengths == [1, 4]
    long_chain = next(c for c in result.chains if len(c.edges) == 4)
    assert [e.target_gene for e in long_chain.edges] == ["PIK3R1", "AKT1", "GSK3B", "MAPT"]


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


# --- Metabolite (MetalinksDB) integration ---


def test_no_metalinks_client_means_protein_only_search_even_if_a_bridge_would_exist():
    """Without metalinks_client (the default), Agent 2 must behave
    exactly like the pre-metabolite pipeline -- metabolite support is
    opt-in, not a silent default that could change results for every
    existing caller."""
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8)],
    )
    omnipath = StubOmniPathClient([])  # no protein-only path at all

    try:
        run_agent2_chain(agent1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())
        assert False, "expected NoChainFoundError"
    except NoChainFoundError:
        pass


def test_metabolite_bridge_found_when_metalinks_client_is_given():
    """A -> Glucose -> X: no protein-only path exists in OmniPath at
    all, but MetalinksDB provides a gene->metabolite->gene bridge. Single
    candidate at this depth, so the LLM is bypassed just like a pure
    protein single-candidate chain would be."""
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8)],
    )
    omnipath = StubOmniPathClient([])
    metalinks = StubMetalinksClient(
        gene_to_metabolite=[MetaboliteEdge("gene", "A", "metabolite", "Glucose", "produces", "HMDB0000122", "PA", "A->Glucose")],
        metabolite_to_gene=[MetaboliteEdge("metabolite", "Glucose", "gene", "X", "activates", "HMDB0000122", "PX", "Glucose->X")],
    )

    result = run_agent2_chain(
        agent1_result, omnipath_client=omnipath, metalinks_client=metalinks, llm_client=RaisingLLMClient()
    )

    assert result.candidate_path_count == 1
    edges = result.chains[0].edges
    assert len(edges) == 2
    assert edges[0].source_gene == "A"
    assert edges[0].source_kind == "gene"
    assert edges[0].target_gene == "Glucose"
    assert edges[0].target_kind == "metabolite"
    assert edges[0].interaction_type == "produces"
    assert edges[0].source_disease_assoc_score == 0.6
    assert edges[0].target_disease_assoc_score is None  # metabolite end never gets a disease score
    assert edges[1].source_gene == "Glucose"
    assert edges[1].source_kind == "metabolite"
    assert edges[1].target_gene == "X"
    assert edges[1].target_kind == "gene"
    assert edges[1].interaction_type == "activates"
    assert edges[1].target_disease_assoc_score == 0.8


def test_metabolite_bridge_and_protein_path_of_equal_length_both_surface_as_candidates():
    """A 2-hop metabolite bridge (A->Cholesterol->X) and a 2-hop
    protein-only path (B->MID->Y) are both discovered at the same BFS
    depth, so both go to the LLM as separate candidates -- exactly like
    two competing protein-only paths would."""
    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6), GeneAssociation("B", "ENSGB", "PB", 0.5)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8), GeneAssociation("Y", "ENSGY", "PY", 0.7)],
    )
    omnipath = StubOmniPathClient(
        [
            Interaction("B", "PB", "MID", "PM", "activates", "PB->PM"),
            Interaction("MID", "PM", "Y", "PY", "inhibits", "PM->PY"),
        ]
    )
    metalinks = StubMetalinksClient(
        gene_to_metabolite=[MetaboliteEdge("gene", "A", "metabolite", "Cholesterol", "produces", "HMDB1", "PA", "A->Chol")],
        metabolite_to_gene=[MetaboliteEdge("metabolite", "Cholesterol", "gene", "X", "activates", "HMDB1", "PX", "Chol->X")],
    )
    llm = StubLLMClient(fixed_response="1: PLAUSIBLE - The metabolite route is well supported.")

    result = run_agent2_chain(agent1_result, omnipath_client=omnipath, metalinks_client=metalinks, llm_client=llm)

    assert result.candidate_path_count == 2
    assert "[metabolite] Cholesterol" in llm.last_prompt
    assert len(result.chains) == 1
    edges = result.chains[0].edges
    assert edges[0].source_gene == "A"
    assert edges[0].target_gene == "Cholesterol"
    assert edges[0].target_kind == "metabolite"
    assert edges[1].source_kind == "metabolite"
    assert edges[1].target_gene == "X"


def test_metabolite_edge_to_dict_includes_kind_and_hmdb_fields():
    import json

    agent1_result = _agent1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8)],
    )
    omnipath = StubOmniPathClient([])
    metalinks = StubMetalinksClient(
        gene_to_metabolite=[MetaboliteEdge("gene", "A", "metabolite", "Glucose", "produces", "HMDB0000122", "PA", "A->Glucose")],
        metabolite_to_gene=[MetaboliteEdge("metabolite", "Glucose", "gene", "X", "activates", "HMDB0000122", "PX", "Glucose->X")],
    )

    result = run_agent2_chain(
        agent1_result, omnipath_client=omnipath, metalinks_client=metalinks, llm_client=RaisingLLMClient()
    )
    edge_dict = result.chains[0].edges[0].to_dict()

    assert edge_dict["target_kind"] == "metabolite"
    assert edge_dict["target_hmdb_id"] == "HMDB0000122"
    json.dumps(result.to_dict())
