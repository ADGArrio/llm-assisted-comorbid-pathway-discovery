"""
Agent 3, tissue-expression check -- GTEx.

Service: GTEx Portal API v2 (https://gtexportal.org/api/v2/expression/medianGeneExpression),
no key needed, accepts a plain gene symbol (auto-resolved).
Data: GTEx response is {"data": [...]}, each record's median value tried
under a few plausible field names (median/value/data) since the exact
schema wasn't confirmed live from this sandbox.
Used by: PipelineB.agent3_confirmation, checking whether a chain's genes
are expressed in tissues relevant to the disease they sit near
(common/config.py's DISEASE_RELEVANT_TISSUES) -- evidence Agent 1 never
looked at.

Simplified 2026-09-14 to GTEx only -- the Human Protein Atlas fallback
was dropped (see claude/agent3_simplification.md): GTEx alone already
covers every tissue this project's three diseases need
(DISEASE_RELEVANT_TISSUES in common/config.py). The removed HPAClient/
StubHPAClient code is still in git history if a real run later shows a
genuine GTEx coverage gap worth filling.

Not tested against its live endpoint from this sandbox (network
allowlist) -- verify field names on a real run.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

GTEX_DEFAULT_BASE_URL = "https://gtexportal.org/api/v2"
GTEX_DEFAULT_DATASET_ID = "gtex_v10"

# A gene counts as "expressed" in a tissue at >= this many nTPM. 1.0 is a
# commonly used low-expression cutoff -- a documented simplification, not
# an exact biological threshold.
EXPRESSION_THRESHOLD = 1.0

# Field names this code will try, in order, when parsing a GTEx
# medianGeneExpression record -- see module docstring for why this isn't
# a single assumed name.
_GTEX_VALUE_FIELD_CANDIDATES = ("median", "value", "data")
_GTEX_GENE_FIELD_CANDIDATES = ("geneSymbol", "gene_symbol", "symbol")


@dataclass
class TissueExpressionResult:
    gene_symbol: str
    tissue: str
    source: str  # "GTEx" | "none"
    expressed: Optional[bool]  # None if GTEx had no record
    value: Optional[float]
    note: str

    def to_dict(self) -> dict:
        return {
            "gene_symbol": self.gene_symbol,
            "tissue": self.tissue,
            "source": self.source,
            "expressed": self.expressed,
            "value": self.value,
            "note": self.note,
        }


def _parse_median_expression(record: dict) -> Optional[float]:
    # Input: one GTEx expression record. Output: its median value, or None if no known field matched.
    for field in _GTEX_VALUE_FIELD_CANDIDATES:
        if field in record and isinstance(record[field], (int, float)):
            return float(record[field])
    return None


class GTExClient:
    def __init__(self, base_url: str = None, dataset_id: str = None, timeout: int = 30):
        self.base_url = base_url or os.environ.get("GTEX_BASE_URL", GTEX_DEFAULT_BASE_URL)
        self.dataset_id = dataset_id or os.environ.get("GTEX_DATASET_ID", GTEX_DEFAULT_DATASET_ID)
        self.timeout = timeout

    def get_median_expression(self, gene_symbol: str, tissue_site_detail_id: str) -> Optional[float]:
        # Input: gene symbol, GTEx tissueSiteDetailId (e.g. "Pancreas", "Brain_Cortex").
        # Output: median nTPM, or None if GTEx has no record (distinct from a confirmed 0.0).
        resp = requests.get(
            f"{self.base_url}/expression/medianGeneExpression",
            params={
                "gencodeId": gene_symbol,  # GTEx v2 accepts a plain gene symbol and auto-resolves it
                "tissueSiteDetailId": tissue_site_detail_id,
                "datasetId": self.dataset_id,
                "itemsPerPage": 5,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get("data") or []
        if not records:
            return None
        return _parse_median_expression(records[0])


class StubGTExClient:
    """gene_symbol|tissue -> median expression value (omit key for "no data"). No network call."""

    def __init__(self, expression: Dict[str, float]):
        self.expression = expression
        self.calls: List[tuple] = []

    def get_median_expression(self, gene_symbol: str, tissue_site_detail_id: str) -> Optional[float]:
        self.calls.append((gene_symbol, tissue_site_detail_id))
        return self.expression.get(f"{gene_symbol}|{tissue_site_detail_id}")


def check_tissue_expression(gene_symbol: str, tissue: str, gtex_client) -> TissueExpressionResult:
    # Input: gene, tissue, GTEx client. Output: TissueExpressionResult.
    value = gtex_client.get_median_expression(gene_symbol, tissue)
    if value is not None:
        return TissueExpressionResult(
            gene_symbol=gene_symbol,
            tissue=tissue,
            source="GTEx",
            expressed=value >= EXPRESSION_THRESHOLD,
            value=value,
            note=f"GTEx median expression {value:.3g} nTPM (threshold {EXPRESSION_THRESHOLD}).",
        )
    return TissueExpressionResult(
        gene_symbol=gene_symbol,
        tissue=tissue,
        source="none",
        expressed=None,
        value=None,
        note="No GTEx record for this gene/tissue.",
    )
