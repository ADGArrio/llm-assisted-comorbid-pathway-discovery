"""The readable report must label the model's free text as unverified commentary and show the code-generated
evidence line; draws in the debate log must not be printed as wins."""
from common.report_format import _format_chain_evidence, _format_debate_log

EVIDENCE = {"n_edges": 3, "n_edges_with_unknown_direction": 1, "weakest_edge_curation_effort": 2}


def _chain(**extra):
    edge = {"source_gene": "A", "target_gene": "B", "interaction_type": "activates", "source_kind": "gene", "target_kind": "gene"}
    chain = {"edges": [edge], "rationale": "A drives B.", "has_unconfirmed_direction": False, "elo_rating": None}
    chain.update(extra)
    return chain


def test_evidence_line_states_database_facts_and_flags_off_path_genes():
    line = _format_chain_evidence(_chain(evidence=EVIDENCE, judge_votes=2, rationale_off_path_genes=["INSR"]))
    assert "direction curated for 2 of 3 edge(s)" in line
    assert "weakest edge curation effort 2" in line and "endorsed in 2 judge pass(es)" in line
    assert "INSR" in line


def test_results_saved_before_the_evidence_block_print_no_evidence_line():
    assert _format_chain_evidence(_chain()) == ""


def test_draw_is_not_reported_as_a_win():
    log = [{"winner_index": 0, "loser_index": 1, "rationale": "x", "draw": True},
           {"winner_index": 1, "loser_index": 0, "rationale": "y", "draw": False}]
    text = _format_debate_log(log, 2)
    assert "DRAW" in text and "Pathway 2 beat Pathway 1: y" in text
    assert "Pathway 1 beat Pathway 2" not in text
