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


def test_parse_response_keeps_complex_nodes_on_either_end():
    """Regression test locking in a deliberate decision, 2026-09-14 (see the
    module docstring's "Complex nodes are kept" note): a filter dropping any
    interaction touching an OmniPath multi-protein COMPLEX (rendered as a
    long underscore-joined pseudo gene symbol, e.g. the mTORC1 complex) was
    tried and then explicitly reverted -- complexes are real, important
    biology, and skipping them isn't a call to make unilaterally; the
    unresolved tradeoff (real mechanism vs. unscorable/unconfirmable) is
    tracked in claude/open_issues_for_committee.md instead. This test exists
    so a future edit can't silently reintroduce that filter without it being
    a deliberate, visible change."""
    tsv = (
        "source\ttarget\tsource_genesymbol\ttarget_genesymbol\tis_directed\tis_stimulation\tis_inhibition\n"
        "P1\tP2\tGENEA\tGENEB\t1\t1\t0\n"
        "COMPLEX:P3_P4\tP5\tCOMPLEXNAME\tGENEC\t1\t1\t0\n"
        "P6\tCOMPLEX:P7_P8\tGENED\tCOMPLEXNAME2\t1\t1\t0\n"
    )
    client = OmniPathClient()
    interactions = client._parse_response(tsv)
    assert len(interactions) == 3
    assert {i.source_gene for i in interactions} == {"GENEA", "COMPLEXNAME", "GENED"}
    assert {i.target_gene for i in interactions} == {"GENEB", "GENEC", "COMPLEXNAME2"}


def test_parse_response_handles_empty_text():
    client = OmniPathClient()
    assert client._parse_response("") == []
    assert client._parse_response("   ") == []


def test_get_outgoing_interactions_batches_large_gene_lists(monkeypatch):
    """Regression test, added 2026-09-14 alongside raising agent2_chain.py's
    _MAX_FRONTIER_SIZE from 300 to 3000: a single request with thousands of
    comma-joined gene symbols risks a URL-length failure. get_outgoing_interactions
    must split a large gene list into multiple _QUERY_BATCH_SIZE-sized requests
    and concatenate their results, transparently to the caller."""
    from PipelineB.omnipath_client import _QUERY_BATCH_SIZE

    calls = []

    class FakeResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    def fake_get(url, params, timeout):
        batch = params["sources"].split(",")
        calls.append(batch)
        rows = "\n".join(f"P_{g}\tPX\t{g}\tTARGET\t1\t1\t0" for g in batch)
        tsv = "source\ttarget\tsource_genesymbol\ttarget_genesymbol\tis_directed\tis_stimulation\tis_inhibition\n" + rows
        return FakeResponse(tsv)

    monkeypatch.setattr("PipelineB.omnipath_client.requests.get", fake_get)
    client = OmniPathClient()
    total_genes = _QUERY_BATCH_SIZE * 2 + 50  # forces 3 batches
    genes = [f"G{i:05d}" for i in range(total_genes)]

    interactions = client.get_outgoing_interactions(genes)

    assert len(calls) == 3
    assert len(calls[0]) == _QUERY_BATCH_SIZE
    assert len(calls[1]) == _QUERY_BATCH_SIZE
    assert len(calls[2]) == 50
    assert len(interactions) == total_genes


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
