from PipelineB.stage1_discovery import Stage1DiscoveryResult
from PipelineB.stage2_judging import (
    EdgeProvenanceError,
    NoChainFoundError,
    _RawEdge,
    _assert_chain_edges_come_from_bfs,
    _assert_selected_paths_are_code_found,
    run_stage2_judging as _run_stage2_judging,
)
from PipelineB.metalinks_client import MetaboliteEdge, StubMetalinksClient
from PipelineB.omnipath_client import Interaction, StubOmniPathClient
from PipelineB.opentargets_client import GeneAssociation
from PipelineB.pubmed_client import Citation
from common.llm_client import StubLLMClient



def run_stage2_judging(*args, **kwargs):
    """The tests below were written for the single-pass behaviour (one judge call, one call per debate, no
    per-start-gene cap). The 2026-10-18 hardening made two judge passes, swapped-order debates and a
    per-start-gene cap the defaults, so these tests pin the legacy settings explicitly; the new behaviour is
    tested in test_stage2_hardening.py."""
    kwargs.setdefault("judge_passes", 1)
    kwargs.setdefault("swap_debate_order", False)
    kwargs.setdefault("max_per_start_gene", None)
    return _run_stage2_judging(*args, **kwargs)


DISEASE1 = "Type 2 Diabetes Mellitus"
DISEASE2 = "Alzheimer's Disease"


class RaisingLLMClient:
    """Fails the test if Stage 2 calls the LLM when it shouldn't have
    (i.e. when there was only one shortest candidate path)."""

    def generate(self, prompt, temperature=0.3):
        raise AssertionError("LLM should not be called when only one shortest path exists")


def _stage1_result(disease1_targets, disease2_targets, citations=None):
    return Stage1DiscoveryResult(
        disease1=DISEASE1,
        disease2=DISEASE2,
        disease1_efo_id="EFO_T2DM",
        disease2_efo_id="EFO_AD",
        disease1_targets=disease1_targets,
        disease2_targets=disease2_targets,
        citations=citations or {},
    )


def test_single_shortest_path_skips_llm_entirely():
    stage1_result = _stage1_result(
        disease1_targets=[GeneAssociation("TCF7L2", "ENSG1", "Q9NQB0", 0.71)],
        disease2_targets=[GeneAssociation("APP", "ENSG2", "P05067", 0.44)],
    )
    omnipath = StubOmniPathClient([Interaction("TCF7L2", "Q9NQB0", "APP", "P05067", "inhibits", "Q9NQB0->P05067")])

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())

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
    stage1_result = _stage1_result(
        disease1_targets=[GeneAssociation("TCF7L2", "ENSG1", "Q9NQB0", 0.71)],
        disease2_targets=[GeneAssociation("APP", "ENSG2", "P05067", 0.44)],
    )
    omnipath = StubOmniPathClient([Interaction("TCF7L2", "Q9NQB0", "APP", "P05067", "unknown", "Q9NQB0->P05067")])

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())

    assert len(result.chains) == 1
    chain = result.chains[0]
    assert chain.has_unconfirmed_direction is True
    assert chain.to_dict()["has_unconfirmed_direction"] is True
    assert chain.rationale is not None
    assert "not established" in chain.rationale


def _two_path_setup():
    stage1_result = _stage1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6), GeneAssociation("B", "ENSGB", "PB", 0.5)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8), GeneAssociation("Y", "ENSGY", "PY", 0.7)],
    )
    omnipath = StubOmniPathClient(
        [
            Interaction("A", "PA", "X", "PX", "activates", "PA->PX"),
            Interaction("B", "PB", "Y", "PY", "inhibits", "PB->PY"),
        ]
    )
    return stage1_result, omnipath


def test_llm_can_select_a_single_candidate_among_several():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response="2: PLAUSIBLE - Because B-Y is the more established pathway.")

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.candidate_path_count == 2
    assert len(result.chains) == 1
    assert result.chains[0].rationale == "Because B-Y is the more established pathway."
    assert result.chains[0].edges[0].source_gene == "B"


def test_llm_can_select_multiple_plausible_candidates():
    # enable_debate_tournament=False: this test is about judge independent selection
    # specifically (both candidates endorsed independently) -- the pairwise debate
    # tournament is exercised by its own dedicated tests below, since it needs a second,
    # differently-shaped LLM response (a "WINNER: A/B ... RATIONALE: ..." debate verdict, not
    # another PLAUSIBLE/NOT PLAUSIBLE line) that this test's single fixed_response doesn't provide.
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response=(
            "1: PLAUSIBLE - Supported by direct literature evidence for A-X.\n"
            "2: PLAUSIBLE - Also plausible based on known B-Y signaling."
        )
    )

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm, enable_debate_tournament=False)

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
    stage1_result = _stage1_result(
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

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    chain = result.chains[0]
    assert chain.has_unconfirmed_direction is True
    assert chain.rationale.startswith("Plausible based on known A-X signaling despite the unclear direction.")
    assert "not established" in chain.rationale


def test_llm_selecting_bare_none_returns_empty_chains_with_no_reason():
    """A bare "NONE" with no reason text (an older or non-compliant LLM
    response) is tolerated, not treated as a formatting error --
    rejection_reason is just None in that case."""
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response="NONE")

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.candidate_path_count == 2
    assert result.chains == []
    assert result.rejection_reason is None


def test_llm_none_response_with_reason_is_captured():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response="NONE: neither candidate names a specific enough mechanism for every step."
    )

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.chains == []
    assert result.rejection_reason == "neither candidate names a specific enough mechanism for every step."


def test_rejection_reason_is_none_when_a_chain_is_endorsed():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response="1: PLAUSIBLE - Supported by direct literature evidence for A-X.")

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    assert result.rejection_reason is None


def test_llm_response_with_no_parseable_lines_and_not_none_raises():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response="I refuse to answer in the requested format.")

    try:
        run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "no parseable" in str(e)


def test_llm_response_ignores_out_of_range_lines_but_keeps_valid_ones():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response="99: PLAUSIBLE - out of range, should be dropped\n1: PLAUSIBLE - This one is valid."
    )

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    assert result.chains[0].rationale == "This one is valid."


def test_candidate_discussed_without_a_verdict_word_is_not_endorsed():
    """Regression test for a real bug, fixed 2026-09-14 (see stage2_judging.py's
    module docstring, "Explicit per-candidate verdicts required"): the model
    used to write one line per candidate whether or not it judged it
    plausible, and the old parser treated any "<number>: <text>" line as an
    endorsement -- so a candidate whose own sentence said the mechanism
    "isn't specifically detailed" got endorsed anyway. A line with no
    explicit PLAUSIBLE/NOT PLAUSIBLE verdict word must not count."""
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response=(
            "1: The mechanism for A activating X is not specifically detailed in the provided "
            "literature or general knowledge as a direct, named biochemical pathway.\n"
            "2: PLAUSIBLE - B directly phosphorylates Y, activating its kinase domain."
        )
    )

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    assert result.chains[0].edges[0].source_gene == "B"


def test_llm_marks_some_candidates_not_plausible_only_plausible_ones_are_endorsed():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response=(
            "1: NOT PLAUSIBLE - no specific mechanism is established for A-X.\n"
            "2: PLAUSIBLE - B directly phosphorylates Y."
        )
    )

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    assert result.chains[0].edges[0].source_gene == "B"
    assert result.rejection_reason is None  # something WAS endorsed -- no blanket rejection reason


def test_all_candidates_marked_not_plausible_produces_a_joined_rejection_reason():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response=(
            "1: NOT PLAUSIBLE - no specific mechanism is established for A-X.\n"
            "2: NOT PLAUSIBLE - no specific mechanism is established for B-Y."
        )
    )

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.chains == []
    assert "candidate 1" in result.rejection_reason
    assert "candidate 2" in result.rejection_reason
    assert "A-X" in result.rejection_reason
    assert "B-Y" in result.rejection_reason


def test_verdict_lines_using_a_period_after_the_number_are_still_parsed():
    """Regression test for a real crash, fixed 2026-09-14 (see stage2_judging.py's
    module docstring, "Verdict-line separator widened"): a real run's LLM wrote
    "1. NOT PLAUSIBLE - ..." (an ordinary numbered-list period) instead of the
    prompt's own "1: NOT PLAUSIBLE - ..." example, on all 10 of its lines, and
    the old regex -- which required a literal colon -- matched none of them,
    raising RuntimeError instead of correctly reporting a blanket rejection.
    This response (verbatim shape, trimmed to two candidates) must now parse
    as two NOT PLAUSIBLE verdicts, not raise."""
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response=(
            "1. NOT PLAUSIBLE - The interaction between A and X lacks a specific, named "
            "mechanism beyond a general activation relationship.\n"
            "2. NOT PLAUSIBLE - The interaction between B and Y lacks a specific, named "
            "mechanism beyond a general activation relationship."
        )
    )

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.chains == []
    assert "candidate 1" in result.rejection_reason
    assert "candidate 2" in result.rejection_reason


def test_verdict_lines_using_a_right_paren_after_the_number_are_still_parsed():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response=(
            "1) PLAUSIBLE - A directly activates X via a well-characterized cascade.\n"
            "2) NOT PLAUSIBLE - no specific mechanism established for B-Y."
        )
    )

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    assert result.chains[0].edges[0].source_gene == "A"


def test_bfs_does_not_stop_exploring_once_max_paths_shallow_hits_are_found():
    """Regression test for a real bug, fixed 2026-09-14 (see stage2_judging.py's
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
    stage1_result = _stage1_result(
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

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.candidate_path_count == 10  # trimmed down from 12 real candidates found
    assert "MID --inhibits--> DEEP" in llm.prompts[0]  # the depth-2 candidate survived the trim


def test_frontier_truncation_protects_disease_scored_genes_over_alphabetical_order():
    """Regression test for a real bug, fixed 2026-09-14 (see stage2_judging.py's
    module docstring, "Frontier truncation prioritized"): a real run showed
    per-depth frontiers of 599-3308 nodes against the safety cap in effect at
    the time, and the old truncation kept only the alphabetically-first N --
    a real, silent sampling bias with no relation to biological relevance. A
    gene like "ZZZGOAL" (disease-associated, but sorting near the end of the
    alphabet) could be dropped from a crowded frontier before its own
    outgoing edges were ever queried, even though continuing past it was
    exactly the point of the goal-gene-expansion fix. Here ZZZGOAL is
    reachable at depth 1 alongside more unscored bridge genes than
    `_MAX_FRONTIER_SIZE` allows (enough to force truncation regardless of the
    current cap value) that all sort before it, and "DEEPER_GOAL" is
    reachable only through ZZZGOAL's own depth-2 outgoing edge. DEEPER_GOAL
    must still be found, proving ZZZGOAL survived truncation despite its
    alphabetical position."""
    from PipelineB.stage2_judging import _MAX_FRONTIER_SIZE

    bridge_genes = [f"BRIDGE{i:06d}" for i in range(_MAX_FRONTIER_SIZE + 50)]
    stage1_result = _stage1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6)],
        disease2_targets=[
            GeneAssociation("ZZZGOAL", "ENSG_Z", "P_Z", 0.5),
            GeneAssociation("DEEPER_GOAL", "ENSG_D", "P_D", 0.9),
        ],
    )
    interactions = [Interaction("A", "PA", g, f"P_{g}", "activates", f"PA->P_{g}") for g in bridge_genes]
    interactions.append(Interaction("A", "PA", "ZZZGOAL", "P_Z", "activates", "PA->P_Z"))
    interactions.append(Interaction("ZZZGOAL", "P_Z", "DEEPER_GOAL", "P_D", "activates", "P_Z->P_D"))
    omnipath = StubOmniPathClient(interactions)
    llm = StubLLMClient(fixed_response="NONE: nothing specific enough.")

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert "ZZZGOAL --activates--> DEEPER_GOAL" in llm.prompts[0]


def test_citations_for_path_genes_are_included_in_the_llm_prompt():
    stage1_result = _stage1_result(
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

    run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert "A causes X via some mechanism" in llm.prompts[0]
    assert "1. A --activates--> X" in llm.prompts[0]


def test_unknown_direction_edge_is_marked_in_the_llm_prompt():
    stage1_result = _stage1_result(
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

    run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert "A --UNKNOWN DIRECTION--> X" in llm.prompts[0]
    assert "B --inhibits--> Y" in llm.prompts[0]


def test_no_path_raises_no_chain_found_error():
    stage1_result = _stage1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8)],
    )
    omnipath = StubOmniPathClient([Interaction("A", "PA", "Z", "PZ", "activates", "PA->PZ")])

    try:
        run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())
        assert False, "expected NoChainFoundError"
    except NoChainFoundError:
        pass


def test_shared_gene_reported_but_not_treated_as_a_zero_edge_path():
    stage1_result = _stage1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6), GeneAssociation("APOE", "ENSGE", "PE", 0.5)],
        disease2_targets=[GeneAssociation("APOE", "ENSGE", "PE", 0.55)],
    )
    omnipath = StubOmniPathClient([Interaction("A", "PA", "APOE", "PE", "activates", "PA->PE")])

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())

    assert result.shared_genes == ["APOE"]
    assert result.candidate_path_count == 1
    assert result.chains[0].edges[0].target_gene == "APOE"


def test_multi_hop_path_found_when_no_direct_edge_exists():
    stage1_result = _stage1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8)],
    )
    omnipath = StubOmniPathClient(
        [
            Interaction("A", "PA", "MID", "PM", "activates", "PA->PM"),
            Interaction("MID", "PM", "X", "PX", "inhibits", "PM->PX"),
        ]
    )

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())

    assert result.candidate_path_count == 1
    edges = result.chains[0].edges
    assert [e.source_gene for e in edges] == ["A", "MID"]
    assert [e.target_gene for e in edges] == ["MID", "X"]
    assert edges[0].target_disease_assoc_score is None  # MID has no disease2 score
    assert edges[1].source_disease_assoc_score is None  # MID has no disease1 score


def test_bfs_continues_past_first_successful_depth_to_find_other_goals_deeper():
    """Regression test for a real bug, fixed 2026-09-14 (see
    stage2_judging.py's module docstring, "BFS depth-exhaustion bug"): the
    search used to stop the moment ANY goal gene was reached, even at
    depth 1, so a different goal gene only reachable at depth 2+ was
    silently never found -- regardless of max_hops. Here A->X is a real
    depth-1 edge, and B->MID->Y is a real depth-2 path to a *different*
    goal gene; both must show up as candidates."""
    stage1_result = _stage1_result(
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

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert result.candidate_path_count == 2
    assert "1. A --activates--> X" in llm.prompts[0]
    assert "B --activates--> MID -> MID --inhibits--> Y" in llm.prompts[0]


def test_bfs_does_not_find_the_same_goal_gene_twice_at_different_depths():
    """A goal gene reached at depth 1 must not also show up as a second,
    longer candidate if a deeper route to that same gene also exists --
    found_targets should keep only its first (shortest) path."""
    stage1_result = _stage1_result(
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

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())

    assert result.candidate_path_count == 1
    assert len(result.chains[0].edges) == 1


def test_bfs_keeps_searching_past_a_goal_gene_for_a_more_specific_one_deeper():
    """Regression test for a real bug, fixed 2026-09-14 (see stage2_judging.py's
    module docstring, "Goal-gene expansion"): a gene can independently clear
    Open Targets' disease2 threshold (a "hit") while being a weak, generic,
    or upstream player -- with a much more specific, better-known mechanism
    sitting one or two hops further down the exact same real chain. Modeled
    directly on a real T2DM->AD run: INSR->PIK3R1 is a genuine 1-hop
    candidate and PIK3R1 independently clears the AD threshold, but the
    textbook continuation PIK3R1->AKT1->GSK3B->MAPT (MAPT = tau, also an AD
    gene) must still be found by continuing the search past PIK3R1, not
    abandoned just because PIK3R1 alone already counted as a hit."""
    stage1_result = _stage1_result(
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

    # enable_debate_tournament=False: this test is about BFS goal-gene-expansion behavior, not
    # the debate tournament -- see the comment on test_llm_can_select_multiple_plausible_candidates.
    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm, enable_debate_tournament=False)

    assert result.candidate_path_count == 2
    lengths = sorted(len(c.edges) for c in result.chains)
    assert lengths == [1, 4]
    long_chain = next(c for c in result.chains if len(c.edges) == 4)
    assert [e.target_gene for e in long_chain.edges] == ["PIK3R1", "AKT1", "GSK3B", "MAPT"]


def test_default_max_hops_is_five():
    from PipelineB.stage2_judging import DEFAULT_MAX_HOPS

    assert DEFAULT_MAX_HOPS == 5


def test_to_dict_is_json_serializable():
    import json

    stage1_result = _stage1_result(
        disease1_targets=[GeneAssociation("TCF7L2", "ENSG1", "Q9NQB0", 0.71)],
        disease2_targets=[GeneAssociation("APP", "ENSG2", "P05067", 0.44)],
    )
    omnipath = StubOmniPathClient([Interaction("TCF7L2", "Q9NQB0", "APP", "P05067", "inhibits", "Q9NQB0->P05067")])
    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())
    json.dumps(result.to_dict())


# --- Metabolite (MetalinksDB) integration ---


def test_no_metalinks_client_means_protein_only_search_even_if_a_bridge_would_exist():
    """Without metalinks_client (the default), Stage 2 must behave
    exactly like the pre-metabolite pipeline -- metabolite support is
    opt-in, not a silent default that could change results for every
    existing caller."""
    stage1_result = _stage1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8)],
    )
    omnipath = StubOmniPathClient([])  # no protein-only path at all

    try:
        run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=RaisingLLMClient())
        assert False, "expected NoChainFoundError"
    except NoChainFoundError:
        pass


def test_metabolite_bridge_found_when_metalinks_client_is_given():
    """A -> Glucose -> X: no protein-only path exists in OmniPath at
    all, but MetalinksDB provides a gene->metabolite->gene bridge. Single
    candidate at this depth, so the LLM is bypassed just like a pure
    protein single-candidate chain would be."""
    stage1_result = _stage1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8)],
    )
    omnipath = StubOmniPathClient([])
    metalinks = StubMetalinksClient(
        gene_to_metabolite=[MetaboliteEdge("gene", "A", "metabolite", "Glucose", "produces", "HMDB0000122", "PA", "A->Glucose")],
        metabolite_to_gene=[MetaboliteEdge("metabolite", "Glucose", "gene", "X", "activates", "HMDB0000122", "PX", "Glucose->X")],
    )

    result = run_stage2_judging(
        stage1_result, omnipath_client=omnipath, metalinks_client=metalinks, llm_client=RaisingLLMClient()
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
    stage1_result = _stage1_result(
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

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, metalinks_client=metalinks, llm_client=llm)

    assert result.candidate_path_count == 2
    assert "[metabolite] Cholesterol" in llm.prompts[0]
    assert len(result.chains) == 1
    edges = result.chains[0].edges
    assert edges[0].source_gene == "A"
    assert edges[0].target_gene == "Cholesterol"
    assert edges[0].target_kind == "metabolite"
    assert edges[1].source_kind == "metabolite"
    assert edges[1].target_gene == "X"


def test_metabolite_edge_to_dict_includes_kind_and_hmdb_fields():
    import json

    stage1_result = _stage1_result(
        disease1_targets=[GeneAssociation("A", "ENSGA", "PA", 0.6)],
        disease2_targets=[GeneAssociation("X", "ENSGX", "PX", 0.8)],
    )
    omnipath = StubOmniPathClient([])
    metalinks = StubMetalinksClient(
        gene_to_metabolite=[MetaboliteEdge("gene", "A", "metabolite", "Glucose", "produces", "HMDB0000122", "PA", "A->Glucose")],
        metabolite_to_gene=[MetaboliteEdge("metabolite", "Glucose", "gene", "X", "activates", "HMDB0000122", "PX", "Glucose->X")],
    )

    result = run_stage2_judging(
        stage1_result, omnipath_client=omnipath, metalinks_client=metalinks, llm_client=RaisingLLMClient()
    )
    edge_dict = result.chains[0].edges[0].to_dict()

    assert edge_dict["target_kind"] == "metabolite"
    assert edge_dict["target_hmdb_id"] == "HMDB0000122"
    json.dumps(result.to_dict())


# --- Pairwise debate ranking ---
# See stage2_judging.py's module docstring, "Pairwise debate ranking" note, and
# claude/committee_feedback_round2_coscientist.md section 3b.


def _three_path_setup():
    stage1_result = _stage1_result(
        disease1_targets=[
            GeneAssociation("A", "ENSGA", "PA", 0.6),
            GeneAssociation("B", "ENSGB", "PB", 0.5),
            GeneAssociation("C", "ENSGC", "PC", 0.4),
        ],
        disease2_targets=[
            GeneAssociation("X", "ENSGX", "PX", 0.8),
            GeneAssociation("Y", "ENSGY", "PY", 0.7),
            GeneAssociation("Z", "ENSGZ", "PZ", 0.6),
        ],
    )
    omnipath = StubOmniPathClient(
        [
            Interaction("A", "PA", "X", "PX", "activates", "PA->PX"),
            Interaction("B", "PB", "Y", "PY", "inhibits", "PB->PY"),
            Interaction("C", "PC", "Z", "PZ", "activates", "PC->PZ"),
        ]
    )
    return stage1_result, omnipath


def test_debate_tournament_runs_for_two_endorsed_candidates_and_updates_elo():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        responses=[
            "1: PLAUSIBLE - Supported by direct literature evidence for A-X.\n"
            "2: PLAUSIBLE - Also plausible based on known B-Y signaling.",
            "WINNER: B\nRATIONALE: B-Y is a far better-characterized mechanism than A-X.",
        ]
    )

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 2
    assert len(result.debate_log) == 1
    debate = result.debate_log[0]
    assert "B-Y is a far better-characterized mechanism" in debate.rationale
    winner_chain = result.chains[debate.winner_index]
    loser_chain = result.chains[debate.loser_index]
    assert winner_chain.edges[0].source_gene == "B"  # B-Y was declared the winner
    assert winner_chain.elo_rating > loser_chain.elo_rating
    assert winner_chain.elo_rating > 1200.0
    assert loser_chain.elo_rating < 1200.0
    # winner sorts first
    assert result.chains[0].edges[0].source_gene == "B"


def test_single_endorsed_candidate_skips_tournament_entirely():
    """A StubLLMClient given only ONE queued response would raise if the tournament tried a
    second call -- proving the tournament is genuinely skipped, not just quietly reusing
    something, when there's only one candidate to compare against nothing."""
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(responses=["1: NOT PLAUSIBLE - weak.\n2: PLAUSIBLE - Because B-Y is the more established pathway."])

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 1
    assert result.chains[0].elo_rating is None
    assert result.debate_log == []


def test_debate_tournament_disabled_via_flag():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        fixed_response=(
            "1: PLAUSIBLE - Supported by direct literature evidence for A-X.\n"
            "2: PLAUSIBLE - Also plausible based on known B-Y signaling."
        )
    )

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm, enable_debate_tournament=False)

    assert len(result.chains) == 2
    assert result.debate_log == []
    assert all(c.elo_rating is None for c in result.chains)


def test_round_robin_tournament_with_three_candidates_runs_all_three_debates():
    stage1_result, omnipath = _three_path_setup()
    llm = StubLLMClient(
        responses=[
            "1: PLAUSIBLE - A-X mechanism.\n2: PLAUSIBLE - B-Y mechanism.\n3: PLAUSIBLE - C-Z mechanism.",
            "WINNER: A\nRATIONALE: A beats B.",  # candidates 0 vs 1
            "WINNER: A\nRATIONALE: A beats C.",  # candidates 0 vs 2
            "WINNER: B\nRATIONALE: B beats C.",  # candidates 1 vs 2
        ]
    )

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    assert len(result.chains) == 3
    assert len(result.debate_log) == 3  # full round-robin over 3 candidates = 3 pairs
    # A won both its debates -- highest Elo, sorts first
    assert result.chains[0].edges[0].source_gene == "A"
    assert result.chains[0].elo_rating > result.chains[1].elo_rating > result.chains[2].elo_rating


def test_unparseable_debate_response_is_a_flagged_draw_not_a_crash():
    # Changed 2026-10-18: this used to raise RuntimeError and abort the whole run (a real run died on a "RATIONE:" typo).
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(
        responses=[
            "1: PLAUSIBLE - Supported by direct literature evidence for A-X.\n"
            "2: PLAUSIBLE - Also plausible based on known B-Y signaling.",
            "I decline to pick a winner.",
            "Still declining.",  # the retry (added 2026-10-15) also fails -> a draw flagged `unparsed`
        ]
    )

    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)

    d = result.debate_log[0]
    assert d.draw is True and d.unparsed is True and "no parseable" in d.rationale
    assert result.bfs_stats["debates_unparsed"] == 1
    assert [c.elo_rating for c in result.chains] == [1200.0, 1200.0]  # a draw between equals moves nobody


# =====================================================================
# 2026-10-15 hardening: the LLM can only SELECT code-found paths (edge provenance), plus parser/retry
# tolerance and the new bfs_stats. "Adversarial" LLM replies below try to smuggle in ungrounded content;
# none of it may change the edges in the result.
# =====================================================================

def _edge_set(result):
    return {(e.source_gene, e.target_gene, e.interaction_type, e.omnipath_edge_id) for c in result.chains for e in c.edges}


_BFS_EDGES_OF_TWO_PATH_SETUP = {("A", "X", "activates", "PA->PX"), ("B", "Y", "inhibits", "PB->PY")}


def test_adversarial_llm_inventing_candidates_and_genes_cannot_add_an_edge():
    """The model endorses a candidate that does not exist (#7, #99), and its rationale asserts a gene
    interaction OmniPath never returned. Only the real candidate #1 may appear, with its real edge."""
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response=(
        "1: PLAUSIBLE - A activates X, and X also activates the made-up gene ZZZ9 via a known pathway.\n"
        "7: PLAUSIBLE - A --activates--> ZZZ9 --inhibits--> APP is textbook biology.\n"
        "99: PLAUSIBLE - TCF7L2 inhibits APP."
    ))
    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm, enable_debate_tournament=False)
    assert _edge_set(result) == {("A", "X", "activates", "PA->PX")}
    assert len(result.chains) == 1
    # The rationale text IS the model's free text (not checked here) -- only the edges are guaranteed.
    assert "ZZZ9" in result.chains[0].rationale


def test_every_chain_edge_is_a_bfs_edge_even_when_the_llm_endorses_everything():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(responses=["1: PLAUSIBLE - ok.\n2: PLAUSIBLE - ok.", "WINNER: A\nRATIONALE: because."])
    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)
    assert _edge_set(result) <= _BFS_EDGES_OF_TWO_PATH_SETUP and len(result.chains) == 2


def test_provenance_guard_rejects_a_selected_path_the_bfs_never_found():
    foreign = [_RawEdge("gene", "A", "PA", None, "gene", "ZZZ9", "PZ", None, "activates", "PA->PZ")]
    real = [_RawEdge("gene", "A", "PA", None, "gene", "X", "PX", None, "activates", "PA->PX")]
    _assert_selected_paths_are_code_found([(real, None)], [real])  # fine
    try:
        _assert_selected_paths_are_code_found([(foreign, "made up")], [real])
        assert False, "expected EdgeProvenanceError"
    except EdgeProvenanceError:
        pass


def test_provenance_guard_rejects_a_chain_edge_not_in_the_bfs_edges():
    from PipelineB.stage2_judging import ChainCandidate, ChainEdge
    real = [_RawEdge("gene", "A", "PA", None, "gene", "X", "PX", None, "activates", "PA->PX")]
    tampered = ChainCandidate(edges=[ChainEdge("A", "PA", "X", "PX", "inhibits", "PA->PX", None, None)], rationale=None)
    try:
        _assert_chain_edges_come_from_bfs([tampered], [real])
        assert False, "expected EdgeProvenanceError"
    except EdgeProvenanceError:
        pass


def test_duplicate_verdict_lines_for_one_candidate_yield_one_chain():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response="1: PLAUSIBLE - first.\n1: PLAUSIBLE - again.\n2: NOT PLAUSIBLE - no.")
    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)
    assert len(result.chains) == 1 and result.chains[0].rationale == "first."


def test_first_verdict_wins_when_the_model_contradicts_itself():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response="1: NOT PLAUSIBLE - weak.\n1: PLAUSIBLE - changed my mind.\n2: NOT PLAUSIBLE - weak.")
    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)
    assert result.chains == []


def test_markdown_and_candidate_prefix_verdict_lines_are_parsed():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response="**1: PLAUSIBLE** - bold reply.\n- Candidate 2: NOT PLAUSIBLE - weak.")
    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)
    assert len(result.chains) == 1 and result.chains[0].edges[0].source_gene == "A"
    assert result.chains[0].rationale == "bold reply."  # asterisks stripped from the rationale too


def test_judge_unparseable_reply_is_retried_once():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(responses=["Sorry, no.", "2: PLAUSIBLE - fine.\n1: NOT PLAUSIBLE - weak."])
    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)
    assert len(llm.prompts) == 2 and len(result.chains) == 1


def test_debate_unparseable_reply_is_retried_once():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(responses=[
        "1: PLAUSIBLE - ok.\n2: PLAUSIBLE - ok.", "I cannot decide.", "WINNER: B\nRATIONALE: B-Y has better support.",
    ])
    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm)
    assert len(llm.prompts) == 3 and result.chains[0].edges[0].source_gene == "B"


def test_bfs_stats_report_the_funnel_and_data_provenance():
    stage1_result, omnipath = _two_path_setup()
    llm = StubLLMClient(fixed_response="1: PLAUSIBLE - ok.\n2: NOT PLAUSIBLE - no.")
    result = run_stage2_judging(stage1_result, omnipath_client=omnipath, llm_client=llm, max_candidate_paths=10)
    st = result.bfs_stats
    assert st["n_paths_before_trim"] == 2 and st["n_paths_after_trim"] == 2 and st["n_judge_endorsed"] == 1
    assert st["paths_by_hops"] == {"1": 2} and st["n_paths_with_unknown_direction"] == 0
    assert st["n_start_genes"] == 2 and st["n_goal_genes"] == 2 and st["n_distinct_goal_genes"] == 2
    assert st["layers"][0]["depth"] == 1 and st["layers"][0]["new_goal_paths"] == 2
    assert st["max_hops"] == 5 and st["metabolites_included"] is False and "retrieved_at_utc" in st
    assert result.to_dict()["bfs_stats"]["n_paths_before_trim"] == 2


def test_bfs_stats_count_paths_that_the_max_paths_trim_drops():
    targets1 = [GeneAssociation(f"S{i}", f"E{i}", f"PS{i}", 0.5) for i in range(5)]
    targets2 = [GeneAssociation(f"T{i}", f"F{i}", f"PT{i}", 0.5) for i in range(5)]
    omnipath = StubOmniPathClient([Interaction(f"S{i}", f"PS{i}", f"T{i}", f"PT{i}", "activates", f"PS{i}->PT{i}") for i in range(5)])
    llm = StubLLMClient(fixed_response="1: PLAUSIBLE - ok.")
    result = run_stage2_judging(_stage1_result(targets1, targets2), omnipath_client=omnipath, llm_client=llm,
                              max_candidate_paths=2, enable_debate_tournament=False)
    assert result.candidate_path_count == 2
    assert result.bfs_stats["n_paths_before_trim"] == 5 and result.bfs_stats["n_paths_after_trim"] == 2


def test_result_saved_before_bfs_stats_existed_still_serialises():
    from PipelineB.stage2_judging import Stage2JudgingResult
    old_style = Stage2JudgingResult(disease1="a", disease2="b", chains=[], candidate_path_count=0, shared_genes=[])
    assert old_style.to_dict()["bfs_stats"] == {}
