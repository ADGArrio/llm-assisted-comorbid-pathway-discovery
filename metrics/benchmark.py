"""
BioPathBench scoring -- Protein Recall and Hallucination Rate for Pipeline A.

Service: none -- pure set comparison over two different reference sets.
Data: both functions take PipelineAResult.cited_symbols (the raw,
de-duplicated candidate list metrics/extract.py already produces --
Pipeline A calls extract_gene_protein_mentions() with no known_symbols
filter, so nothing has been screened out yet). Protein Recall compares
against Agent 1's own per-run discovered gene set (Open Targets output
for the same disease pair) -- "did the ungrounded LLM also name genes a
grounded source flagged as disease-relevant." Hallucination Rate
compares against the full HGNC approved-symbol list (gene_reference.py)
-- "did the LLM name something that isn't a real human gene at all,"
independent of disease relevance.
Used by: the Pipeline A vs Pipeline B comparison script (not yet built).

Note: `known_symbols` in metrics/extract.py filters/narrows the
extracted list at extraction time; it was never wired up in
PipelineA.pipeline_a (nothing calls extract_gene_protein_mentions with
it in production), so cited_symbols is always the full unfiltered
candidate list today, which is exactly what these two functions need.
"""
from typing import List, Set

from PipelineB.agent1_discovery import Agent1DiscoveryResult


def protein_recall(cited_symbols: List[str], agent1_result: Agent1DiscoveryResult) -> float:
    # Input: Pipeline A's cited_symbols, Agent 1's result for the SAME disease pair.
    # Output: fraction of Agent 1's discovered genes (both diseases, union) that also
    # appear in cited_symbols. 0.0 if Agent 1 found no genes at all (not undefined).
    reference = {t.gene_symbol.upper() for t in agent1_result.disease1_targets + agent1_result.disease2_targets}
    if not reference:
        return 0.0
    cited = {s.upper() for s in cited_symbols}
    return len(cited & reference) / len(reference)


def hallucination_rate(cited_symbols: List[str], known_gene_symbols: Set[str]) -> float:
    # Input: Pipeline A's cited_symbols, the full HGNC approved-symbol set (gene_reference.py).
    # Output: fraction of cited_symbols that are NOT a real approved gene symbol.
    # 0.0 if cited_symbols is empty (nothing was said, so nothing was hallucinated).
    if not cited_symbols:
        return 0.0
    cited = {s.upper() for s in cited_symbols}
    known = {s.upper() for s in known_gene_symbols}
    hallucinated = cited - known
    return len(hallucinated) / len(cited)
