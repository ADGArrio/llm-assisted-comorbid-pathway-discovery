"""Plain-text formatting helpers for Stage 2 results (evidence line and pairwise-debate log), kept out of main.py so they can be unit-tested."""


# Build one line of database-derived facts about a chain. Input: a Stage 2 chain dict. Output: the evidence line, or an
# empty string if the chain has no evidence block.
def _format_chain_evidence(chain: dict) -> str:
    ev = chain.get("evidence") or {}
    if not ev:
        return ""
    n_edges = ev.get("n_edges", 0)
    curated = n_edges - ev.get("n_edges_with_unknown_direction", 0)
    parts = [f"direction curated for {curated} of {n_edges} edge(s)"]
    if ev.get("weakest_edge_curation_effort") is not None:
        parts.append(f"weakest edge curation effort {ev['weakest_edge_curation_effort']}")
    if chain.get("judge_votes") is not None:
        parts.append(f"endorsed in {chain['judge_votes']} judge pass(es)")
    line = "Evidence (from databases): " + "; ".join(parts) + "."
    off_path = chain.get("rationale_off_path_genes") or []
    if off_path:
        line += f" The commentary also names gene(s) not on this chain: {', '.join(off_path)}."
    return line


# Format every pairwise debate Stage 2 ran. Input: the debate log and the number of endorsed chains. Output: multi-line
# text with each debate's winner (or draw) and the model's rationale.
def _format_debate_log(debate_log: list, chain_count: int) -> str:
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
