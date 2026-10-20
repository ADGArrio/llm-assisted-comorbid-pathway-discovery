"""Tests for scripts/verify_edge_provenance.py (pure functions; no network)."""
from scripts.verify_edge_provenance import (
    audit_rationale,
    audit_result,
    check_edge,
    chain_genes,
    extract_gene_mentions,
    genes_in_cited_titles,
)

UNIVERSE = {"INSR", "PTPN1", "CAPN1", "OGG1", "AKT1", "APP", "MAPT"}


def _edge(s, t, typ="activates"):
    return {"source_gene": s, "target_gene": t, "interaction_type": typ, "source_kind": "gene", "target_kind": "gene"}


def test_check_edge_exact_mismatch_and_missing():
    live = {("INSR", "PTPN1"): "activates", ("PTPN1", "CAPN1"): "unknown"}
    assert check_edge(_edge("INSR", "PTPN1"), live)["status"] == "exact"
    assert check_edge(_edge("PTPN1", "CAPN1", "inhibits"), live)["status"] == "type_mismatch"
    assert check_edge(_edge("CAPN1", "OGG1", "inhibits"), live)["status"] == "not_found"


def test_extract_gene_mentions_ignores_acronyms_and_non_universe_tokens():
    text = "INSR activates PTPN1; DNA damage and ATP levels matter in AD, as does the AKT1 kinase (PI3K)."
    assert extract_gene_mentions(text, UNIVERSE) == {"INSR", "PTPN1", "AKT1"}
    assert extract_gene_mentions(None, UNIVERSE) == set() and extract_gene_mentions("", UNIVERSE) == set()


def test_extract_gene_mentions_is_not_fooled_by_substrings():
    assert extract_gene_mentions("INSRX and xINSR and INSR2", UNIVERSE) == set()
    assert extract_gene_mentions("(INSR)", UNIVERSE) == {"INSR"}


def test_audit_rationale_flags_genes_outside_the_chain():
    allowed = {"INSR", "PTPN1"}
    out = audit_rationale("INSR activates PTPN1 and AKT1 then MAPT.", allowed, UNIVERSE)
    assert out["off_path_genes"] == ["AKT1", "MAPT"] and out["genes_mentioned"] == ["AKT1", "INSR", "MAPT", "PTPN1"]


def test_genes_named_in_shown_citation_titles_are_not_counted_as_outside_knowledge():
    cites = {"INSR": [{"title": "AKT1 signalling downstream of INSR"}, {"title": "x"}, {"title": "APP third title is not shown"}]}
    assert genes_in_cited_titles(["INSR"], cites, UNIVERSE) == {"AKT1", "INSR"}  # only the first 2 titles are shown to the LLM


def _saved(rationale, debate_rationale=None, edges=None):
    edges = edges or [_edge("INSR", "PTPN1"), _edge("PTPN1", "CAPN1", "inhibits")]
    agent2 = {"chains": [{"edges": edges, "rationale": rationale}, {"edges": [_edge("INSR", "APP")], "rationale": None}],
              "debate_log": [{"winner_index": 0, "loser_index": 1, "rationale": debate_rationale}] if debate_rationale else []}
    return {"citations": {}}, agent2


def test_audit_result_summary_all_grounded():
    a1, a2 = _saved("INSR activates PTPN1 which inhibits CAPN1.")
    live = {("INSR", "PTPN1"): "activates", ("PTPN1", "CAPN1"): "inhibits", ("INSR", "APP"): "activates"}
    s = audit_result(a1, a2, live, UNIVERSE)["summary"]
    assert s["n_gene_edges_checked"] == 3 and s["n_exact"] == 3 and s["all_edges_grounded"] is True
    assert s["n_rationales_naming_off_path_genes"] == 0


def test_audit_result_detects_a_missing_edge_and_off_path_rationale_and_debate():
    a1, a2 = _saved("INSR activates PTPN1, then AKT1 phosphorylates MAPT.", debate_rationale="Chain A is better because of OGG1.")
    live = {("INSR", "PTPN1"): "activates"}  # the other edges are not in the database
    out = audit_result(a1, a2, live, UNIVERSE)
    s = out["summary"]
    assert s["all_edges_grounded"] is False and s["n_not_found"] == 2 and s["n_exact"] == 1
    assert s["n_rationales_audited"] == 3 and s["n_rationales_naming_off_path_genes"] == 2  # chain 0 and the debate
    assert out["debates"][0]["off_path_genes"] == ["OGG1"]


def test_metabolite_edges_are_reported_as_not_checked_not_as_failures():
    edges = [dict(_edge("INSR", "Glucose"), target_kind="metabolite"), _edge("INSR", "PTPN1")]
    a1, a2 = _saved(None, edges=edges)
    out = audit_result(a1, a2, {("INSR", "PTPN1"): "activates", ("INSR", "APP"): "activates"}, UNIVERSE)
    assert out["summary"]["n_metabolite_edges_not_checked"] == 1 and out["summary"]["all_edges_grounded"] is True


def test_chain_genes_excludes_metabolites():
    edges = [dict(_edge("INSR", "Glucose"), target_kind="metabolite"), dict(_edge("Glucose", "PTPN1"), source_kind="metabolite")]
    assert chain_genes({"edges": edges}) == {"INSR", "PTPN1"}


def test_disease_abbreviations_that_are_also_gene_symbols_are_not_counted():
    universe = UNIVERSE | {"CAD", "AD"}
    assert extract_gene_mentions("Relevant to CAD and AD, via INSR.", universe) == {"INSR"}
