from PipelineB.omnipath_client import Interaction, OmniPathClient, StubOmniPathClient, _interaction_type, _synthesize_edge_id


def test_interaction_type_stimulation():
    assert _interaction_type("1", "0") == "activates"


def test_interaction_type_inhibition():
    assert _interaction_type("0", "1") == "inhibits"


def test_interaction_type_unknown_when_neither_flag_set():
    assert _interaction_type("0", "0") == "unknown"


def test_interaction_type_stimulation_wins_on_conflicting_flags():
    assert _interaction_type("1", "1") == "activates"


def test_synthesize_edge_id_is_deterministic():
    assert _synthesize_edge_id("P12345", "Q99999") == "P12345->Q99999"


def test_parse_response_skips_undirected_edges():
    tsv = (
        "source\ttarget\tsource_genesymbol\ttarget_genesymbol\tis_directed\tis_stimulation\tis_inhibition\n"
        "P1\tP2\tGENEA\tGENEB\t1\t1\t0\n"
        "P3\tP4\tGENEC\tGENED\t0\t1\t0\n"
    )
    client = OmniPathClient()
    interactions = client._parse_response(tsv)
    assert len(interactions) == 1
    assert interactions[0].source_gene == "GENEA"
    assert interactions[0].target_gene == "GENEB"
    assert interactions[0].interaction_type == "activates"


def test_parse_response_skips_complex_nodes_on_either_end():
    """Regression test for a real issue, fixed 2026-09-14 (see the module
    docstring's "Complex nodes are dropped" note): a real T2DM->AD run's
    BFS routed a chain through three OmniPath multi-protein complexes in a
    row (mTORC1, the ULK1 complex, the AMPK complex), rendered as long
    underscore-joined pseudo gene symbols. A complex can never carry an
    Open Targets score and can't be confirmed by Agent 3, so these rows
    must be dropped entirely, on either end, not passed through as if
    they were ordinary single-gene edges."""
    tsv = (
        "source\ttarget\tsource_genesymbol\ttarget_genesymbol\tis_directed\tis_stimulation\tis_inhibition\n"
        "P1\tP2\tGENEA\tGENEB\t1\t1\t0\n"
        "COMPLEX:P3_P4\tP5\tCOMPLEXNAME\tGENEC\t1\t1\t0\n"
        "P6\tCOMPLEX:P7_P8\tGENED\tCOMPLEXNAME2\t1\t1\t0\n"
    )
    client = OmniPathClient()
    interactions = client._parse_response(tsv)
    assert len(interactions) == 1
    assert interactions[0].source_gene == "GENEA"
    assert interactions[0].target_gene == "GENEB"


def test_parse_response_handles_empty_text():
    client = OmniPathClient()
    assert client._parse_response("") == []
    assert client._parse_response("   ") == []


def test_stub_returns_outgoing_edges_including_targets_outside_the_source_set():
    interactions = [
        Interaction("A", "P1", "B", "P2", "activates", "P1->P2"),
        Interaction("B", "P2", "C", "P3", "inhibits", "P2->P3"),
    ]
    client = StubOmniPathClient(interactions)
    # Querying source gene "A" only should still return A->B even though
    # B (the target) isn't itself in the queried source set -- outgoing
    # edges go wherever they go, not just to other queried genes.
    result = client.get_outgoing_interactions(["A"])
    assert len(result) == 1
    assert result[0].source_gene == "A"
    assert result[0].target_gene == "B"
    assert client.calls == [["A"]]
