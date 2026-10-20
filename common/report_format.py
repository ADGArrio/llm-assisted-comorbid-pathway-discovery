"""Plain-text formatting helpers for Agent 2 results, kept out of main.py (which runs the whole pipeline when
imported) so they can be unit-tested. Added 2026-10-18 with the evidence block and the draw-aware debate log."""


def _format_chain_evidence(chain: dict) -> str:
    """One code-generated line of grounded facts about a chain (see agent2_chain._build_evidence): how many
    edges have a curated direction, OmniPath's weakest per-edge evidence count, how many stage-1 passes
    endorsed it, and any other genes the model's commentary names (unverified claims). Empty for results
    saved before 2026-10-18, which have no evidence block."""
    ev = chain.get("evidence") or {}
    if not ev:
        return ""
    n_edges = ev.get("n_edges", 0)
    curated = n_edges - ev.get("n_edges_with_unknown_direction", 0)
    parts = [f"direction curated for {curated} of {n_edges} edge(s)"]
    if ev.get("weakest_edge_curation_effort") is not None:
        parts.append(f"weakest edge curation effort {ev['weakest_edge_curation_effort']}")
    if chain.get("stage1_votes") is not None:
        parts.append(f"endorsed in {chain['stage1_votes']} stage-1 pass(es)")
    line = "Evidence (from databases): " + "; ".join(parts) + "."
    off_path = chain.get("rationale_off_path_genes") or []
    if off_path:
        line += f" The commentary also names gene(s) not on this chain: {', '.join(off_path)}."
    return line


def _format_debate_log(debate_log: list, chain_count: int) -> str:
    """Every stage-2 pairwise debate Agent 2 ran between its endorsed candidates, in the order
    they were run -- each debate's winner and the model's own rationale for preferring it, kept
    in full rather than collapsed into just the final ranking. See agent2_chain.py's module
    docstring, "Pairwise debate ranking" note. `winner_index`/`loser_index` refer to the final,
    Elo-sorted Pathway numbers above (0-indexed here, so add 1 for the "Pathway N" label).
    The model's text calls the two chains "Candidate A" and "Candidate B"; when the saved debate records
    which chain was shown as A first (`first_a_index`, runs from 2026-10-18 on), a legend says which is which
    in each of the two runs (A and B swap in the second run)."""
    lines = [f"Pairwise debate log ({len(debate_log)} debate(s) among {chain_count} endorsed pathway(s)):"]
    for d in debate_log:
        one, two = d["winner_index"] + 1, d["loser_index"] + 1
        legend = ""
        a = d.get("first_a_index")
        if a is not None:
            first_a = a + 1
            first_b = (two if first_a == one else one)
            legend = f" [first run: A = Pathway {first_a}, B = Pathway {first_b}"
            if d.get("rationale_swapped"):
                legend += f"; second run: A = Pathway {first_b}, B = Pathway {first_a}"
            legend += "]"
        if d.get("unparsed"):
            lines.append(f"  Pathway {one} vs Pathway {two}: NO VERDICT (model reply could not be parsed; counted as a draw): {d['rationale']}")
        elif d.get("draw"):
            lines.append(
                f"  Pathway {one} vs Pathway {two}: DRAW (the verdict changed "
                f"when the two were presented in the opposite order){legend}: {d['rationale']}"
            )
        else:
            lines.append(f"  Pathway {one} beat Pathway {two}{legend}: {d['rationale']}")
    return "\n".join(lines)
