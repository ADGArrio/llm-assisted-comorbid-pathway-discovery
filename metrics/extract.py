"""
Post-hoc gene/protein symbol extraction from free-text LLM output.

Service: none -- pure regex, no external API.
Data: plain text in, list of symbol strings out.
Used by: PipelineA.pipeline_a (extracts symbols from the ungrounded narrative); feeds BioPathBench's Protein Recall / Hallucination Rate metrics.
"""
import re
from typing import Iterable, Optional, List

# HGNC-style symbols: starts with a letter, uppercase alphanumeric, 2-10 chars,
# optional trailing "-<digits>" so "IRS-1" extracts whole instead of truncating to "IRS".
_SYMBOL_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,9}(?:-\d{1,2})?\b")

# Capitalized tokens that match the pattern but aren't gene symbols (disease abbreviations, acronyms, etc.).
_COMMON_FALSE_POSITIVES = {
    "DNA", "RNA", "MRNA", "LLM", "PPI", "API", "AI", "ICU", "EHR",
    "GWAS", "CAD", "AD", "T2DM", "COPD", "CVD", "MRI", "CT", "USA", "FDA",
    "NIH", "ATP", "ADP", "ICD", "EFO", "UI", "ID",
}


def extract_gene_protein_mentions(
    text: str,
    known_symbols: Optional[Iterable[str]] = None,
) -> List[str]:
    # Input: free text, optional known_symbols allowlist (e.g. Agent 1's gene set).
    # Output: candidate symbols in first-seen order, de-duplicated, false positives removed,
    # filtered to known_symbols if provided.
    candidates = _SYMBOL_PATTERN.findall(text or "")

    deduped = []
    seen = set()
    for symbol in candidates:
        if symbol in _COMMON_FALSE_POSITIVES or symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)

    if known_symbols is not None:
        allowed = {s.upper() for s in known_symbols}
        deduped = [s for s in deduped if s in allowed]

    return deduped
