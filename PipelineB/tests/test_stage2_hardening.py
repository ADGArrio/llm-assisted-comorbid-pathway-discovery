"""Tests for Stage 2's hardened behaviour: trim policy, two-order judge, control pass, debate order swapping, evidence record, bridge citations and
unparseable-reply handling.
"""
import pytest

from PipelineB.stage1_discovery import Stage1DiscoveryResult
from PipelineB.stage2_judging import (
    _RawEdge,
    _bfs_shortest_paths,
    _off_path_genes,
    _trim_candidates,
    _update_elo,
    _update_elo_draw,
    run_stage2_judging,  # the REAL defaults: two judge passes, swapped debates, per-start-gene cap
)
from PipelineB.omnipath_client import Interaction, StubOmniPathClient
from PipelineB.opentargets_client import GeneAssociation
from PipelineB.pubmed_client import Citation
from common.llm_client import StubLLMClient


# Build a gene-to-gene _RawEdge.
def _edge(src, dst, typ="activates"):
    return _RawEdge("gene", src, None, None, "gene", dst, None, None, typ, f"{src}->{dst}")


# Build a path of edges through the given gene names.
def _path(start, *rest, typ="activates"):
    nodes = [start, *rest]
    return [_edge(a, b, typ) for a, b in zip(nodes, nodes[1:])]


# ---------------------------------------------------------------- trim policy
# Trim keeps longest first and prefers signed among equal lengths.
def test_trim_keeps_longest_first_and_prefers_signed_among_equal_lengths():
    found = [_path("A", "B", typ="unknown"), _path("C", "D"), _path("E", "F", "G"), _path("H", "I")]
    kept = _trim_candidates(found, 3)
    assert [len(p) for p in kept] == [2, 1, 1]
    assert kept[1][0].source_id == "C" and kept[2][0].source_id == "H"  # the signed 1-hops, not the unknown one


# Trim equal rank keeps discovery order.
def test_trim_equal_rank_keeps_discovery_order():
    found = [_path("A", "B"), _path("C", "D"), _path("E", "F")]
    assert [p[0].source_id for p in _trim_candidates(found, 2)] == ["A", "C"]


# Per start gene cap rebalances the kept set.
def test_per_start_gene_cap_rebalances_the_kept_set():
    found = [_path("INSR", f"T{i}", f"U{i}") for i in range(5)] + [_path("PPARG", "V", "W"), _path("IRS1", "X", "Y")]
    kept = _trim_candidates(found, 4, max_per_start_gene=2)
    starts = [p[0].source_id for p in kept]
    assert starts.count("INSR") == 2 and "PPARG" in starts and "IRS1" in starts


# Cap never leaves a slot empty.
def test_cap_never_leaves_a_slot_empty():
    found = [_path("INSR", f"T{i}", f"U{i}") for i in range(5)]
    kept = _trim_candidates(found, 4, max_per_start_gene=1)
    assert len(kept) == 4 and {p[0].source_id for p in kept} == {"INSR"}  # backfilled from the skipped ones


# Cap zero or none means no cap.
def test_cap_zero_or_none_means_no_cap():
    found = [_path("INSR", f"T{i}", f"U{i}") for i in range(5)]
    assert len(_trim_candidates(found, 3, 0)) == len(_trim_candidates(found, 3, None)) == 3


# BFS applies the cap and records the policy.
def test_bfs_applies_the_cap_and_records_the_policy():
    interactions = [Interaction("S1", "", f"G{i}", "", "activates", f"S1->G{i}") for i in range(4)]
    interactions += [Interaction("S2", "", "G9", "", "activates", "S2->G9")]
    stats = {}
    paths = _bfs_shortest_paths(
        StubOmniPathClient(interactions), None, ["S1", "S2"], {"G0", "G1", "G2", "G3", "G9"}, 2, 3,
        stats=stats, max_per_start_gene=2,
    )
    assert sorted(p[0].source_id for p in paths) == ["S1", "S1", "S2"]
    assert stats["trim_policy"]["max_per_start_gene"] == 2
    assert stats["n_paths_before_trim"] == 5 and stats["n_paths_after_trim"] == 3
    assert stats["n_distinct_start_genes_after_trim"] == 2
    assert stats["any_layer_truncated"] is False


# ---------------------------------------------------------------- Elo
# Elo draw between equals does not move ratings.
def test_elo_draw_between_equals_does_not_move_ratings():
    assert _update_elo_draw(1200.0, 1200.0) == (1200.0, 1200.0)


# Elo draw moves the stronger side down and conserves points.
def test_elo_draw_moves_the_stronger_side_down_and_conserves_points():
    a, b = _update_elo_draw(1300.0, 1200.0)
    assert a < 1300.0 and b > 1200.0
    assert a + b == pytest.approx(2500.0)


# Elo win still conserves points.
def test_elo_win_still_conserves_points():
    w, l = _update_elo(1200.0, 1200.0)
    assert w + l == pytest.approx(2400.0) and w > l


# ---------------------------------------------------------------- end-to-end with stubs
DISEASE1, DISEASE2 = "Type 2 Diabetes Mellitus", "Alzheimer's Disease"


# Build a Stage 1 result and stub OmniPath client with two candidate paths.
def _two_candidate_setup(citations=None):
    a1 = Stage1DiscoveryResult(
        disease1=DISEASE1, disease2=DISEASE2, disease1_efo_id="E1", disease2_efo_id="E2",
        disease1_targets=[GeneAssociation("A", "e1", "P1", 0.9), GeneAssociation("B", "e2", "P2", 0.8)],
        disease2_targets=[GeneAssociation("X", "e3", "P3", 0.7), GeneAssociation("Y", "e4", "P4", 0.6)],
        citations=citations or {},
    )
    omnipath = StubOmniPathClient([
        Interaction("A", "P1", "X", "P3", "activates", "P1->P3", curation_effort=31, n_sources=12),
        Interaction("B", "P2", "Y", "P4", "unknown", "P2->P4", curation_effort=2, n_sources=1),
    ])
    return a1, omnipath


FORWARD_BOTH = "1: PLAUSIBLE - A drives X.\n2: PLAUSIBLE - B reaches Y."
REVERSED_BOTH = "1: PLAUSIBLE - B reaches Y.\n2: PLAUSIBLE - A drives X."


# Judge is judged twice and votes are recorded.
def test_judge_is_judged_twice_and_votes_are_recorded():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[
        "1: PLAUSIBLE - A drives X.\n2: NOT PLAUSIBLE - weak.",   # forward: A->X yes, B->Y no
        "1: NOT PLAUSIBLE - weak.\n2: PLAUSIBLE - A drives X.",   # reversed list is [B->Y, A->X]: A->X yes again
    ])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm)
    assert len(llm.prompts) == 2 and len(result.chains) == 1
    assert result.chains[0].judge_votes == 2
    assert result.chains[0].edges[0].source_gene == "A"
    assert result.bfs_stats["judge_order_disagreements"] == 0
    assert result.bfs_stats["judge_endorsed_both"] == 1
    # the second prompt lists the candidates in reversed order
    assert llm.prompts[0].index("B --") > llm.prompts[0].index("A --")
    assert llm.prompts[1].index("A --") > llm.prompts[1].index("B --")


# A verdict that flips with order is kept by union and counted as a disagreement.
def test_a_verdict_that_flips_with_order_is_kept_by_union_and_counted_as_a_disagreement():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[
        "1: PLAUSIBLE - A drives X.\n2: NOT PLAUSIBLE - weak.",   # forward: only A->X
        "1: PLAUSIBLE - B reaches Y.\n2: NOT PLAUSIBLE - weak.",  # reversed: only B->Y (first listed)
        "WINNER: A\nRATIONALE: first.",                          # debate A->X vs B->Y
        "WINNER: B\nRATIONALE: first again.",                    # swapped: A is B->Y, B is A->X -> A->X wins both
    ])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm)
    assert len(result.chains) == 2
    assert sorted(c.judge_votes for c in result.chains) == [1, 1]
    assert result.bfs_stats["judge_order_disagreements"] == 2


# Min votes two requires both orders to agree.
def test_min_votes_two_requires_both_orders_to_agree():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[
        "1: PLAUSIBLE - A drives X.\n2: PLAUSIBLE - B reaches Y.",
        "1: NOT PLAUSIBLE - weak.\n2: PLAUSIBLE - A drives X.",
    ])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm, judge_min_votes=2)
    assert [c.edges[0].source_gene for c in result.chains] == ["A"]


# Consistent swapped debate is decisive.
def test_consistent_swapped_debate_is_decisive():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[
        FORWARD_BOTH, REVERSED_BOTH,
        "WINNER: A\nRATIONALE: A->X is better.",   # A = candidate A->X
        "WINNER: B\nRATIONALE: still A->X.",       # swapped: B = candidate A->X
    ])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm)
    assert len(llm.prompts) == 4
    assert result.chains[0].edges[0].source_gene == "A"
    assert result.chains[0].elo_rating > 1200.0 > result.chains[1].elo_rating
    d = result.debate_log[0]
    assert d.draw is False and (d.winner_index, d.loser_index) == (0, 1) and d.rationale_swapped
    assert result.bfs_stats["debates_run"] == 1 and result.bfs_stats["debates_position_inconsistent"] == 0


# Position biased debate becomes a draw.
def test_position_biased_debate_becomes_a_draw():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[
        FORWARD_BOTH, REVERSED_BOTH,
        "WINNER: A\nRATIONALE: picks the first listed.",
        "WINNER: A\nRATIONALE: picks the first listed again.",   # swapped -> the OTHER candidate wins: inconsistent
    ])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm)
    assert result.debate_log[0].draw is True
    assert [c.elo_rating for c in result.chains] == [1200.0, 1200.0]
    assert result.bfs_stats["debates_position_inconsistent"] == 1
    assert result.to_dict()["debate_log"][0]["draw"] is True


# Swapped debates can be switched off.
def test_swapped_debates_can_be_switched_off():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[FORWARD_BOTH, REVERSED_BOTH, "WINNER: B\nRATIONALE: B->Y is better."])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm, swap_debate_order=False)
    assert len(llm.prompts) == 3 and result.debate_log[0].rationale_swapped is None
    assert result.bfs_stats["debates_position_inconsistent"] is None


# Lone candidate is not judged and has no votes.
def test_lone_candidate_is_not_judged_and_has_no_votes():
    a1 = Stage1DiscoveryResult(
        disease1=DISEASE1, disease2=DISEASE2, disease1_efo_id="E1", disease2_efo_id="E2",
        disease1_targets=[GeneAssociation("A", "e1", "P1", 0.9)],
        disease2_targets=[GeneAssociation("X", "e3", "P3", 0.7)],
        citations={},
    )
    omnipath = StubOmniPathClient([Interaction("A", "P1", "X", "P3", "activates", "P1->P3")])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=StubLLMClient(responses=[]))
    assert result.chains[0].judge_votes is None


# ---------------------------------------------------------------- evidence block + commentary label
# Evidence block is built from database facts and the citations shown.
def test_evidence_block_is_built_from_database_facts_and_the_citations_shown():
    cites = {"A": [Citation("111", "Paper one"), Citation("222", "Paper two"), Citation("333", "Paper three")]}
    a1, omnipath = _two_candidate_setup(citations=cites)
    llm = StubLLMClient(responses=["1: PLAUSIBLE - A drives X.\n2: NOT PLAUSIBLE - weak.",
                                   "1: NOT PLAUSIBLE - weak.\n2: PLAUSIBLE - A drives X."])
    chain = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm).chains[0]
    ev = chain.evidence
    assert ev["n_edges"] == 1 and ev["n_edges_with_unknown_direction"] == 0
    assert ev["edges"][0] == {"step": "A -> X", "interaction_type": "activates", "direction_curated": True,
                              "curation_effort": 31, "n_sources": 12}
    assert ev["weakest_edge_curation_effort"] == 31
    gene_a = next(g for g in ev["genes"] if g["gene"] == "A")
    assert gene_a["disease1_assoc_score"] == 0.9
    assert [c["pmid"] for c in gene_a["literature_shown_to_model"]] == ["111", "222"]  # only what the prompt showed
    assert chain.edges[0].curation_effort == 31 and chain.edges[0].n_sources == 12
    assert "unverified" in chain.to_dict()["rationale_source"] or "not verified" in chain.to_dict()["rationale_source"]


# Unsigned edge is flagged in the evidence block.
def test_unsigned_edge_is_flagged_in_the_evidence_block():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=["1: NOT PLAUSIBLE - weak.\n2: PLAUSIBLE - B reaches Y.",
                                   "1: PLAUSIBLE - B reaches Y.\n2: NOT PLAUSIBLE - weak."])
    ev = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm).chains[0].evidence
    assert ev["n_edges_with_unknown_direction"] == 1 and ev["edges"][0]["direction_curated"] is False


# Off path gene in rationale is listed but does not reject the chain.
def test_off_path_gene_in_rationale_is_listed_but_does_not_reject_the_chain():
    a1 = Stage1DiscoveryResult(
        disease1=DISEASE1, disease2=DISEASE2, disease1_efo_id="E1", disease2_efo_id="E2",
        disease1_targets=[GeneAssociation("AKT1", "e1", "P1", 0.9), GeneAssociation("INSR", "e2", "P2", 0.8)],
        disease2_targets=[GeneAssociation("GSK3B", "e3", "P3", 0.7), GeneAssociation("PPARG", "e4", "P4", 0.6)],
        citations={},
    )
    omnipath = StubOmniPathClient([
        Interaction("AKT1", "P1", "GSK3B", "P3", "inhibits", "P1->P3"),
        Interaction("INSR", "P2", "PPARG", "P4", "activates", "P2->P4"),
    ])
    llm = StubLLMClient(responses=[
        "1: PLAUSIBLE - AKT1 inhibits GSK3B, as INSR signalling also does, and TP53 is unrelated.\n2: NOT PLAUSIBLE - weak.",
        "1: NOT PLAUSIBLE - weak.\n2: PLAUSIBLE - AKT1 inhibits GSK3B.",
    ])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm)
    assert len(result.chains) == 1  # flagged, not rejected
    # INSR/PPARG are genes the search saw but are off this chain -> listed; TP53 was never seen -> cannot be recognised
    assert result.chains[0].rationale_off_path_genes == ["INSR"]
    assert result.bfs_stats["n_chains_with_off_path_gene_in_rationale"] == 1
    assert result.to_dict()["chains"][0]["rationale_off_path_genes"] == ["INSR"]


# Off path genes helper rules.
def test_off_path_genes_helper_rules():
    path = _path("INSR", "PTPN1")
    universe = {"INSR", "PTPN1", "MAPK8", "CAD", "AD", "APP"}
    text = "INSR activates PTPN1 and, in CAD and AD, MAPK8 and APP matter; tp53 and Insulin are ignored."
    assert _off_path_genes(text, path, universe) == ["APP", "MAPK8"]  # on-path genes, disease abbreviations, non-gene words excluded
    assert _off_path_genes(None, path, universe) == []
    assert _off_path_genes("", path, universe) == []


# Old style results without new fields still serialise.
def test_old_style_results_without_new_fields_still_serialise():
    a1, omnipath = _two_candidate_setup()
    d = run_stage2_judging(
        a1, omnipath_client=omnipath, judge_passes=1, swap_debate_order=False,
        llm_client=StubLLMClient(responses=[FORWARD_BOTH, "WINNER: A\nRATIONALE: x."]),
    ).to_dict()
    assert "evidence" in d["chains"][0] and "judge_votes" in d["chains"][0]


# --- debate verdict parsing tolerance ---

# Debate verdict parses a misspelled rationale label.
def test_debate_verdict_parses_a_misspelled_rationale_label():
    from PipelineB.stage2_judging import _parse_debate_verdict
    assert _parse_debate_verdict("WINNER: A\nRATIONE: Candidate A is better.") == (0, "Candidate A is better.")


# Debate verdict parses winner with no rationale label.
def test_debate_verdict_parses_winner_with_no_rationale_label():
    from PipelineB.stage2_judging import _parse_debate_verdict
    assert _parse_debate_verdict("WINNER: B\nBecause B is direct.") == (1, "Because B is direct.")
    assert _parse_debate_verdict("winner: b") == (1, "(no rationale given)")


# Debate verdict still requires a clear winner line.
def test_debate_verdict_still_requires_a_clear_winner_line():
    from PipelineB.stage2_judging import _parse_debate_verdict
    assert _parse_debate_verdict("I cannot decide. RATIONALE: both are fine.") is None
    assert _parse_debate_verdict("WINNER: neither\nRATIONALE: x") is None


# --- noise-control pass, unjudged count, judge evidence and readable debate log ---

# Control pass measures noise without changing which chains are kept.
def test_control_pass_measures_noise_without_changing_which_chains_are_kept():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[
        "1: PLAUSIBLE - A drives X.\n2: NOT PLAUSIBLE - weak.",   # forward
        "1: NOT PLAUSIBLE - weak.\n2: PLAUSIBLE - A drives X.",   # reversed list [B->Y, A->X]: A->X again
        "1: PLAUSIBLE - A drives X.\n2: PLAUSIBLE - B reaches Y.",  # control: forward again, now ALSO endorses B->Y
    ])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm, judge_control_pass=True)
    assert len(llm.prompts) == 3
    assert [c.edges[0].source_gene for c in result.chains] == ["A"]   # the control never adds B->Y
    assert result.chains[0].judge_votes == 2                         # nor changes the votes
    st = result.bfs_stats
    assert st["judge_control_pass"] is True
    assert st["judge_endorsed_forward_repeat"] == 2
    assert st["judge_noise_disagreements"] == 1                      # forward vs forward-again differ on B->Y
    assert st["judge_order_disagreements"] == 0                      # forward vs reversed agree


# Control pass is off by default.
def test_control_pass_is_off_by_default():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[FORWARD_BOTH, REVERSED_BOTH, "WINNER: A\nRATIONALE: x.", "WINNER: B\nRATIONALE: y."])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm)
    assert "judge_control_pass" not in result.bfs_stats


# Candidates the model gave no verdict for are counted.
def test_candidates_the_model_gave_no_verdict_for_are_counted():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[
        "1: PLAUSIBLE - A drives X.",                              # forward: nothing at all said about candidate 2
        "Sorry, I cannot say anything about that one.",            # forward retry for candidate 2: still no verdict
        "2: PLAUSIBLE - A drives X.\n1: NOT PLAUSIBLE - weak.",    # reversed: both addressed
    ])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm)
    assert result.bfs_stats["judge_unjudged_forward"] == 1
    assert result.bfs_stats["judge_unjudged_reversed"] == 0
    diag = result.bfs_stats["judge_diagnostics"]["forward"]
    assert diag["retried_unjudged"] is True and diag["unjudged_first_reply"] == 1
    assert len(diag["raw_replies"]) == 2 and diag["raw_replies"][1]["retry"] is True


# Unaddressed candidates get a second chance and keep their numbers.
def test_unaddressed_candidates_get_a_second_chance_and_keep_their_numbers():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[
        "1: NOT PLAUSIBLE - weak.",                  # forward: candidate 2 never mentioned
        "2: PLAUSIBLE - B drives Y well.",           # forward retry asked ONLY about candidate 2, original number kept
        "1: NOT PLAUSIBLE - weak.\n2: NOT PLAUSIBLE - weak.",  # reversed
    ])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm)
    assert result.bfs_stats["judge_unjudged_forward"] == 0
    assert result.bfs_stats["judge_endorsed_forward"] == 1
    retry_prompt = llm.prompts[1]
    assert "\n2. " in retry_prompt and "\n1. " not in retry_prompt  # only candidate 2 listed, numbered 2


# Implausible is read as a rejection.
def test_implausible_is_read_as_a_rejection():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[
        "1: PLAUSIBLE - A drives X.\n2: IMPLAUSIBLE - no mechanism.",
        "1: IMPLAUSIBLE - no mechanism.\n2: PLAUSIBLE - A drives X.",
    ])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm)
    assert [c.edges[0].source_gene for c in result.chains] == ["A"]
    assert result.bfs_stats["judge_unjudged_forward"] == 0


# Judge sees OmniPath edge strength only when asked.
def test_judge_sees_omnipath_edge_strength_only_when_asked():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[FORWARD_BOTH, REVERSED_BOTH, "WINNER: A\nRATIONALE: x.", "WINNER: B\nRATIONALE: y."])
    run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm, judge_evidence=True)
    assert "[OmniPath: 12 resources, curation effort 31]" in llm.prompts[0]
    assert "[OmniPath: 1 resource, curation effort 2]" in llm.prompts[0]       # singular
    assert "better established" in llm.prompts[0]                             # the explanatory note
    assert "[OmniPath:" in llm.prompts[2]                                     # also in the debate prompts

    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[FORWARD_BOTH, REVERSED_BOTH, "WINNER: A\nRATIONALE: x.", "WINNER: B\nRATIONALE: y."])
    run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm)
    assert all("[OmniPath:" not in p for p in llm.prompts)


# PubMed stand-in that records searches and can fail for chosen genes.
class _FakePubMed:
    # Store the canned data.
    def __init__(self, fail_for=()):
        self.calls, self.fail_for = [], set(fail_for)

    # Record the search; raise for genes in fail_for, else return one citation.
    def search_citations(self, gene, disease, retmax=3):
        self.calls.append((gene, disease, retmax))
        if gene in self.fail_for:
            raise ConnectionError("PubMed down")
        return [Citation(pmid=f"{gene}-{disease[:3]}", title=f"{gene} in {disease}", gene_symbol=gene, disease_name=disease)]


# Bridge genes get one PubMed title per disease and stage1 citations are untouched.
def test_bridge_genes_get_one_pubmed_title_per_disease_and_stage1_citations_are_untouched():
    cites = {"A": [Citation(pmid="1", title="A already has a paper")]}
    a1, omnipath = _two_candidate_setup(citations=cites)
    pm = _FakePubMed()
    llm = StubLLMClient(responses=[FORWARD_BOTH, REVERSED_BOTH, "WINNER: A\nRATIONALE: x.", "WINNER: B\nRATIONALE: y."])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm, judge_evidence=True, pubmed_client=pm)
    searched = {c[0] for c in pm.calls}
    assert searched == {"B", "X", "Y"} and all(c[2] == 1 for c in pm.calls)   # A had literature: not searched
    assert '"X in Type 2 Diabetes Mellitus"' in llm.prompts[0]
    assert set(a1.citations) == {"A"}                                          # Stage 1's dict not mutated
    st = result.bfs_stats
    assert st["judge_evidence"] == "omnipath_edge_strength+bridge_gene_citations"
    assert (st["bridge_genes_searched"], st["bridge_genes_with_citations"], st["bridge_citation_search_errors"]) == (3, 3, 0)
    shown = {g["gene"]: g["literature_shown_to_model"] for c in result.chains for g in c.evidence["genes"]}
    assert shown["X"] and shown["X"][0]["title"].startswith("X in")           # the evidence block records what was shown


# A failed bridge search is counted not fatal.
def test_a_failed_bridge_search_is_counted_not_fatal():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[FORWARD_BOTH, REVERSED_BOTH, "WINNER: A\nRATIONALE: x.", "WINNER: B\nRATIONALE: y."])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm, judge_evidence=True,
                              pubmed_client=_FakePubMed(fail_for={"X"}))
    assert result.bfs_stats["bridge_citation_search_errors"] == 2             # X failed for both diseases
    assert len(result.chains) == 2


# Judge evidence without a PubMed client only adds edge strength.
def test_judge_evidence_without_a_pubmed_client_only_adds_edge_strength():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[FORWARD_BOTH, REVERSED_BOTH, "WINNER: A\nRATIONALE: x.", "WINNER: B\nRATIONALE: y."])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm, judge_evidence=True)
    assert result.bfs_stats["judge_evidence"] == "omnipath_edge_strength"
    assert "bridge_genes_searched" not in result.bfs_stats


# Debate log records which chain was candidate A first.
def test_debate_log_records_which_chain_was_candidate_a_first():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[FORWARD_BOTH, REVERSED_BOTH, "WINNER: B\nRATIONALE: B->Y is better."])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm, swap_debate_order=False)
    # B->Y won and is now chain 0; A->X (shown as Candidate A) is chain 1
    assert result.chains[0].edges[0].source_gene == "B"
    assert result.debate_log[0].first_a_index == 1
    assert result.to_dict()["debate_log"][0]["first_a_index"] == 1


# Swapped run unparsed is also a draw.
def test_swapped_run_unparsed_is_also_a_draw():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(responses=[FORWARD_BOTH, REVERSED_BOTH, "WINNER: A\nRATIONALE: x.", "no idea", "still no idea"])
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm)
    d = result.debate_log[0]
    assert d.draw and d.unparsed and d.rationale == "x." and "no parseable" in d.rationale_swapped
    assert result.bfs_stats["debates_position_inconsistent"] == 0 and result.bfs_stats["debates_unparsed"] == 1


# Debate log text explains which pathway was candidate A and B.
def test_debate_log_text_explains_which_pathway_was_candidate_a_and_b():
    from common.report_format import _format_debate_log
    log = [{"winner_index": 0, "loser_index": 1, "rationale": "Candidate B is better.", "draw": False,
            "rationale_swapped": "Candidate A is better.", "first_a_index": 1, "unparsed": False}]
    text = _format_debate_log(log, 2)
    assert "first run: A = Pathway 2, B = Pathway 1; second run: A = Pathway 1, B = Pathway 2" in text
    old = [{"winner_index": 0, "loser_index": 1, "rationale": "r", "draw": False}]
    assert "first run" not in _format_debate_log(old, 2)
    unparsed = [{"winner_index": 0, "loser_index": 1, "rationale": "(no parseable)", "draw": True, "unparsed": True}]
    assert "NO VERDICT" in _format_debate_log(unparsed, 2)


# BFS stats expose the pool of possible bridge genes.
def test_bfs_stats_expose_the_pool_of_possible_bridge_genes():
    a1, omnipath = _two_candidate_setup()
    llm = StubLLMClient(fixed_response="1: NOT PLAUSIBLE - weak.\n2: NOT PLAUSIBLE - weak.")
    result = run_stage2_judging(a1, omnipath_client=omnipath, llm_client=llm)
    # two single-edge candidates -> no intermediate genes; the key must exist and be a list
    assert result.bfs_stats["candidate_bridge_genes"] == []
