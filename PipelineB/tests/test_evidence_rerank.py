import pytest

from PipelineB.evidence_rerank import assess_chain, chain_label, gene_tissue_summary, rerank, rerank_saved


def _tc(gene, flags):
    return [{"gene_symbol": gene, "tissue": f"T{i}", "expressed": f} for i, f in enumerate(flags)]


def _chain(*gene_flags):
    return {"tissue_checks": [t for g, f in gene_flags for t in _tc(g, f)]}


def test_gene_summary_ignores_checks_without_data():
    s = gene_tissue_summary(_tc("A", [True, None, False]))
    assert s == {"A": {"checked": 2, "expressed": 1}}


def test_chain_with_a_gene_absent_everywhere_fails_even_if_most_checks_pass():
    a = assess_chain(_chain(("A", [True] * 6), ("B", [True] * 6), ("C", [False] * 3)))
    assert not a["passes"] and a["absent_genes"] == ["C"]
    assert a["tissue_fraction"] == pytest.approx(12 / 15)


def test_low_overall_fraction_fails_without_an_absent_gene():
    a = assess_chain(_chain(("A", [True, False, False, False]), ("B", [True, False, False, False])))
    assert not a["passes"] and a["absent_genes"] == []


def test_chain_with_no_tissue_data_is_not_penalised_but_flagged():
    a = assess_chain(_chain(("A", [None, None])))
    assert a["passes"] and not a["assessable"]


def test_failing_chain_moves_below_passing_chains_keeping_debate_order():
    chains = [{"elo_rating": 1216}, {"elo_rating": 1200}, {"elo_rating": 1184}]
    a3 = [_chain(("A", [False] * 3)), _chain(("B", [True] * 3)), _chain(("C", [True] * 3))]
    out = rerank(chains, a3)
    assert out["new_order"] == [1, 2, 0] and out["top_changed"] and not out["top_original_passes"]


def test_all_failing_chains_abstain():
    out = rerank([{}, {}], [_chain(("A", [False])), _chain(("B", [False]))])
    assert out["abstains"] and out["n_pass"] == 0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        rerank([{}], [])


def test_rerank_saved_adds_labels_and_handles_missing_stage3():
    edge = lambda s, t: {"source_gene": s, "target_gene": t}
    result = {"stage2": {"chains": [{"edges": [edge("X", "Y"), edge("Y", "Z")], "elo_rating": 1}]}, "stage3": [_chain(("X", [True]))]}
    assert rerank_saved(result)["chains"][0]["label"] == "X>Y>Z" == chain_label(result["stage2"]["chains"][0])
    assert rerank_saved({"stage2": result["stage2"], "stage3": []}) is None
