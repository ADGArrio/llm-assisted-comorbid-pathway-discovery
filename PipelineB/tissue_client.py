"""
Agent 3, tissue-expression check -- GTEx.

Service: GTEx Portal API v2 (https://gtexportal.org/api/v2/expression/medianGeneExpression),
no key needed.
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

BUG FIX, 2026-09-14: a real T2DM->AD run came back with 0/54 tissue
checks confirmed across every gene in every endorsed chain -- including
extremely well-studied genes (INSR, IRS1, GLP1R) that certainly do have
GTEx records. Root cause: this module was passing the plain gene symbol
(e.g. "INSR") as `gencodeId` to medianGeneExpression. Per GTEx's own
published API schema, `gencodeId` requires a *versioned GENCODE/Ensembl
gene id* (e.g. "ENSG00000010404.19"), not a gene symbol -- so every
lookup silently matched zero records and this module reported "no GTEx
data" for genes that plainly have data. This was the "auto-resolved"
assumption in this docstring's prior wording, which was never verified
live (this sandbox's network allowlist blocks gtexportal.org, and
WebFetch is blocked by gtexportal.org's own robots.txt) -- confirmed
instead from GTEx's own OpenAPI spec (mirrored at
github.com/broadinstitute/gtex-public/blob/master/gtex_api/gtex_api.yml).
Fixed by resolving gene symbol -> gencodeId first, via GTEx's own
`/reference/gene` endpoint (which does accept a plain gene symbol) --
see `_resolve_gencode_id` below. Still not verified against the live
endpoint from this sandbox; if a real run still comes back with
unexpectedly many "no GTEx record" results, check that resolution step
first (e.g. print `GTExClient()._resolve_gencode_id("INSR")`) before
assuming the genes really do lack data.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

GTEX_DEFAULT_BASE_URL = "https://gtexportal.org/api/v2"
GTEX_DEFAULT_DATASET_ID = "gtex_v10"
# gtex_v10 (like v8) is built on the GRCh38/hg38 genome assembly -- passed explicitly when
# resolving a gene symbol to its gencodeId so the resolved id's genome build actually matches
# the dataset being queried (GTEx also serves a legacy GRCh37/hg19 gencodeVersion by default,
# which would silently mismatch gtex_v10).
GTEX_DEFAULT_GENOME_BUILD = "GRCh38/hg38"

# A gene counts as "expressed" in a tissue at >= this many nTPM. 1.0 is a
# commonly used low-expression cutoff -- a documented simplification, not
# an exact biological threshold.
EXPRESSION_THRESHOLD = 1.0

# Field names this code will try, in order, when parsing a GTEx
# medianGeneExpression record -- see module docstring for why this isn't
# a single assumed name.
_GTEX_VALUE_FIELD_CANDIDATES = ("median", "value", "data")
_GTEX_GENE_FIELD_CANDIDATES = ("geneSymbol", "gene_symbol", "symbol")
_GTEX_GENCODE_ID_FIELD_CANDIDATES = ("gencodeId", "gencode_id")


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
    def __init__(
        self,
        base_url: str = None,
        dataset_id: str = None,
        genome_build: str = None,
        timeout: int = 30,
    ):
        self.base_url = base_url or os.environ.get("GTEX_BASE_URL", GTEX_DEFAULT_BASE_URL)
        self.dataset_id = dataset_id or os.environ.get("GTEX_DATASET_ID", GTEX_DEFAULT_DATASET_ID)
        self.genome_build = genome_build or os.environ.get("GTEX_GENOME_BUILD", GTEX_DEFAULT_GENOME_BUILD)
        self.timeout = timeout
        # gene symbol -> resolved gencodeId (or None if unresolvable), cached per client instance
        # since the same gene is looked up once per relevant tissue (see agent3_confirmation.py).
        self._gencode_id_cache: Dict[str, Optional[str]] = {}

    def _resolve_gencode_id(self, gene_symbol: str) -> Optional[str]:
        # Input: gene symbol. Output: its versioned gencodeId for self.genome_build, or None if
        # GTEx's own gene reference has no matching record -- see module docstring's bug-fix note:
        # medianGeneExpression requires this versioned id, a plain gene symbol always returns
        # zero records.
        if gene_symbol in self._gencode_id_cache:
            return self._gencode_id_cache[gene_symbol]

        resp = requests.get(
            f"{self.base_url}/reference/gene",
            params={"geneId": gene_symbol, "genomeBuild": self.genome_build},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get("data") or []

        gencode_id = None
        for record in records:
            symbol = None
            for field in _GTEX_GENE_FIELD_CANDIDATES:
                if field in record:
                    symbol = record[field]
                    break
            if symbol and str(symbol).upper() == gene_symbol.upper():
                for field in _GTEX_GENCODE_ID_FIELD_CANDIDATES:
                    if field in record:
                        gencode_id = record[field]
                        break
                break

        self._gencode_id_cache[gene_symbol] = gencode_id
        return gencode_id

    def get_median_expression(self, gene_symbol: str, tissue_site_detail_id: str) -> Optional[float]:
        # Input: gene symbol, GTEx tissueSiteDetailId (e.g. "Pancreas", "Brain_Cortex").
        # Output: median nTPM, or None if GTEx has no record (distinct from a confirmed 0.0) --
        # including the case where gene_symbol itself doesn't resolve to a gencodeId at all.
        gencode_id = self._resolve_gencode_id(gene_symbol)
        if gencode_id is None:
            return None

        resp = requests.get(
            f"{self.base_url}/expression/medianGeneExpression",
            params={
                "gencodeId": gencode_id,
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
