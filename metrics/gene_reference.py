"""
HGNC approved gene symbol reference, for hallucination detection.

Service: none -- reads a local copy of HGNC's complete gene set (public,
not credentialed, just not reachable from this environment's network --
same class of restriction as Open Targets/OmniPath/GTEx/GWAS Catalog).
Download once from https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt
and set HGNC_GENE_LIST_PATH in .env to its local path.
Data: tab-separated; the `symbol` column is the current HGNC-approved
human gene symbol, one per row (~43k rows).
Used by: metrics/benchmark.py's hallucination_rate(), as the ground
truth for "is this string naming a real human gene at all."
"""
import os

import pandas as pd

SYMBOL_COLUMN = "symbol"


def load_known_gene_symbols(path: str = None) -> set:
    # Input: path to HGNC's hgnc_complete_set.txt, or None to read HGNC_GENE_LIST_PATH from .env.
    # Output: set of uppercased approved gene symbols. Raises FileNotFoundError with a download
    # hint if no path is configured or the file doesn't exist.
    path = path or os.environ.get("HGNC_GENE_LIST_PATH")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "No HGNC gene list found. Download the complete set (public, no credentials needed) "
            "from https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt "
            "and set HGNC_GENE_LIST_PATH in .env to its local path -- this environment's network "
            "can't reach genenames.org/ebi.ac.uk directly."
        )
    df = pd.read_csv(path, sep="\t", usecols=[SYMBOL_COLUMN], dtype=str)
    return set(df[SYMBOL_COLUMN].dropna().str.upper())
