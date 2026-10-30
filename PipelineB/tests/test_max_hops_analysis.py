"""Tests for scripts/max_hops_analysis.py: pure functions on a tiny in-memory network, with no network and no LLM."""
import random
from collections import Counter

from PipelineB.omnipath_client import Interaction
from scripts import max_hops_analysis as mh


# Build an Interaction from gene s to gene t.
def _i(s, t, typ="activates"):
    return Interaction(source_gene=s, source_uniprot="", target_gene=t, target_uniprot="", interaction_type=typ,
                       omnipath_edge_id=f"{s}->{t}")


# A -> B -> C -> D (chain, last edge unsigned); A -> X (dead end); E -> D
NET = [_i("A", "B"), _i("B", "C", "inhibits"), _i("C", "D", "unknown"), _i("A", "X"), _i("E", "D")]


# In memory client serves outgoing edges only.
def test_in_memory_client_serves_outgoing_edges_only():
    c = mh.InMemoryOmniPath(NET)
    got = c.get_outgoing_interactions(["A"])
    assert {(e.source_gene, e.target_gene) for e in got} == {("A", "B"), ("A", "X")}
    assert c.get_outgoing_interactions(["D"]) == []
    # duplicates in the request do not duplicate edges
    assert len(c.get_outgoing_interactions(["A", "A"])) == 2


# Run BFS finds shortest paths per goal and stats.
def test_run_bfs_finds_shortest_paths_per_goal_and_stats():
    c = mh.InMemoryOmniPath(NET)
    paths, stats = mh.run_bfs(c, ["A", "E"], {"B", "D"}, max_depth=5)
    by_goal = {p[-1].target_id: len(p) for p in paths}
    assert by_goal == {"B": 1, "D": 1}  # E->D is 1 hop and shorter than A->B->C->D
    assert stats["n_start_genes"] == 2 and stats["n_goal_genes"] == 2


# Run BFS respects depth and does not trim.
def test_run_bfs_respects_depth_and_does_not_trim():
    c = mh.InMemoryOmniPath(NET)
    paths, _ = mh.run_bfs(c, ["A"], {"D"}, max_depth=2)
    assert paths == []
    paths, _ = mh.run_bfs(c, ["A"], {"D"}, max_depth=3)
    assert [len(p) for p in paths] == [3]


# Per depth table counts and cumulative.
def test_per_depth_table_counts_and_cumulative():
    c = mh.InMemoryOmniPath(NET)
    paths, stats = mh.run_bfs(c, ["A"], {"B", "D"}, max_depth=5)
    t = mh.per_depth_table(paths, stats, 5)
    rows = {r["hops"]: r for r in t["rows"]}
    assert rows[1]["new_paths"] == 1 and rows[3]["new_paths"] == 1
    assert rows[3]["cumulative_paths"] == 2 and rows[5]["cumulative_paths"] == 2
    # the 3-hop path ends in an unsigned edge
    assert rows[3]["fully_signed_new_paths"] == 0 and rows[3]["fraction_fully_signed"] == 0.0
    assert rows[1]["fraction_fully_signed"] == 1.0
    assert rows[2]["fraction_fully_signed"] is None  # nothing new at hop 2
    # the longest-first trim: with budget 3 both paths are kept and mean length is 2
    assert rows[3]["kept_n"] == 2 and rows[3]["kept_mean_hops"] == 2.0
    assert rows[1]["kept_n"] == 1 and rows[1]["kept_mean_hops"] == 1.0


# Per depth table trim keeps longest.
def test_per_depth_table_trim_keeps_longest():
    c = mh.InMemoryOmniPath(NET)
    paths, stats = mh.run_bfs(c, ["A"], {"B", "D"}, max_depth=5)
    t = mh.per_depth_table(paths, stats, 5, max_paths=1)
    rows = {r["hops"]: r for r in t["rows"]}
    assert rows[3]["kept_n"] == 1 and rows[3]["kept_mean_hops"] == 3.0


# Sample matched same size same bucket and distinct.
def test_sample_matched_same_size_same_bucket_and_distinct():
    interactions = [_i(f"G{k}", f"T{k}") for k in range(20)] + [_i("G0", f"Z{k}") for k in range(8)]
    out_pools, _, out_deg, _ = mh.degree_pools(interactions)
    observed = ["G0", "G1", "G2"]
    rng = random.Random(1)
    picked = mh.sample_matched(observed, out_pools, out_deg, rng)
    assert len(picked) == 3
    for obs, pick in zip(observed, picked):
        assert mh._bin(out_deg[pick]) == mh._bin(out_deg[obs])
    assert len(set(picked)) == 3  # bucket of G1/G2 (degree 1) has plenty of members, so picks are distinct


# Sample matched is reproducible for a seed.
def test_sample_matched_is_reproducible_for_a_seed():
    interactions = [_i(f"G{k}", f"T{k}") for k in range(30)]
    out_pools, _, out_deg, _ = mh.degree_pools(interactions)
    a = mh.sample_matched(["G1", "G2"], out_pools, out_deg, random.Random(7))
    b = mh.sample_matched(["G1", "G2"], out_pools, out_deg, random.Random(7))
    assert a == b


# Empirical p bounds and direction.
def test_empirical_p_bounds_and_direction():
    nulls = [0, 1, 1, 2, 3]
    assert mh.empirical_p(0, nulls) == 1.0           # everything is >= 0
    assert mh.empirical_p(10, nulls) == round(1 / 6, 3)  # nothing reaches 10; +1 smoothing
    assert mh.empirical_p(2, nulls) == round((2 + 1) / 6, 3)


# Null new paths summary shape and determinism.
def test_null_new_paths_summary_shape_and_determinism():
    interactions = [_i(f"G{k}", f"G{(k + 1) % 12}") for k in range(12)]  # a ring
    starts, goals = ["G0", "G1"], {"G5", "G6"}
    a = mh.null_new_paths(interactions, starts, goals, replicates=5, max_depth=4, seed=3)
    b = mh.null_new_paths(interactions, starts, goals, replicates=5, max_depth=4, seed=3)
    assert a == b
    assert set(a) == {1, 2, 3, 4}
    assert all(v["n"] == 5 and v["p2_5"] <= v["mean"] <= v["p97_5"] + 1e-9 for v in a.values())
