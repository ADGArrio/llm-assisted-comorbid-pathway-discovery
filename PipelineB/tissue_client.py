"""
Agent 3, tissue-expression check -- GTEx (primary) + Human Protein Atlas (fallback).

Service: GTEx Portal API v2 (https://gtexportal.org/api/v2/expression/medianGeneExpression),
no key needed, accepts a plain gene symbol (auto-resolved). Fallback: HPA
per-gene JSON at https://www.proteinatlas.org/<ensembl_id>.json, used only
when GTEx has no record AND an Ensembl id is known for the gene.
Data: GTEx response is {"data": [...]}, each record's median value tried
under a few plausible field names (median/value/data) since the exact
schema wasn't confirmed live from this sandbox. HPA's per-gene JSON tissue
keys are similarly unconfirmed and parsed defensively; returns None (not
an exception) if nothing matches, since HPA is only ever a fallback.
Used by: PipelineB.agent3_confirmation, checking whether a chain's genes
are expressed in tissues relevant to the disease they sit near
(common/config.py's DISEASE_RELEVANT_TISSUES) -- evidence Agent 1 never
looked at.

Neither client tested against its live endpoint from this sandbox
(network allowlist) -- verify field names on a real run.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

GTEX_DEFAULT_BASE_URL = "https://gtexportal.org/api/v2"
GTEX_DEFAULT_DATASET_ID = "gtex_v10"
HPA_DEFAULT_BASE_URL = "https://www.proteinatlas.org"

# A gene counts as "expressed" in a tissue at >= this many nTPM (GTEx or HPA,
# both normalized-TPM-family units). 1.0 is a common low-expression cutoff.
EXPRESSION_THRESHOLD = 1.0

_GTEX_VALUE_FIELD_CANDIDATES = ("median", "value", "data")
_GTEX_GENE_FIELD_CANDIDATES = ("geneSymbol", "gene_symbol", "symbol")


@dataclass
class TissueExpressionResult:
    gene_symbol: str
    tissue: str
    source: str  # "GTEx" | "HPA" | "none"
    expressed: Optional[bool]  # None if no data was found in either source
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


def _parse_hpa_tissue_expression(payload: dict, tissue: str) -> Optional[float]:
    # Input: HPA per-gene JSON payload, tissue name. Output: nTPM value if a known key
    # variant matches, else None (HPA is a fallback -- missing data here isn't an error).
    tissue_key_variants = (
        f"t_RNA_{tissue}",
        f"RNA {tissue.replace('_', ' ')} nTPM",
        tissue,
    )
    for key in tissue_key_variants:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    for item in payload.get("Tissue RNA", []) if isinstance(payload.get("Tissue RNA"), list) else []:
        if isinstance(item, dict) and str(item.get("tissue", "")).replace(" ", "_") == tissue:
            value = item.get("value") or item.get("nTPM")
            if isinstance(value, (int, float)):
                return float(value)
    return None


class HPAClient:
    def __init__(self, base_url: str = None, timeout: int = 30):
        self.base_url = base_url or os.environ.get("HPA_BASE_URL", HPA_DEFAULT_BASE_URL)
        self.timeout = timeout

    def get_tissue_expression(self, ensembl_id: str, tissue: str) -> Optional[float]:
        # Input: Ensembl gene id, tissue name. Output: nTPM value, or None.
        resp = requests.get(f"{self.base_url}/{ensembl_id}.json", timeout=self.timeout)
        resp.raise_for_status()
        return _parse_hpa_tissue_expression(resp.json(), tissue)


class StubHPAClient:
    """ensembl_id|tissue -> nTPM value. No network call."""

    def __init__(self, expression: Dict[str, float]):
        self.expression = expression
        self.calls: List[tuple] = []

    def get_tissue_expression(self, ensembl_id: str, tissue: str) -> Optional[float]:
        self.calls.append((ensembl_id, tissue))
        return self.expression.get(f"{ensembl_id}|{tissue}")


def check_tissue_expression(
    gene_symbol: str,
    tissue: str,
    gtex_client,
    hpa_client,
    ensembl_id: Optional[str] = None,
) -> TissueExpressionResult:
    # Input: gene, tissue, both clients, optional Ensembl id (needed for the HPA fallback).
    # Output: TissueExpressionResult. Tries GTEx first; falls back to HPA only if GTEx has no
    # record and an Ensembl id is available (not every chain gene has one).
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

    if ensembl_id:
        value = hpa_client.get_tissue_expression(ensembl_id, tissue)
        if value is not None:
            return TissueExpressionResult(
                gene_symbol=gene_symbol,
                tissue=tissue,
                source="HPA",
                expressed=value >= EXPRESSION_THRESHOLD,
                value=value,
                note=f"GTEx had no record; HPA nTPM {value:.3g} (threshold {EXPRESSION_THRESHOLD}).",
            )
        return TissueExpressionResult(
            gene_symbol=gene_symbol,
            tissue=tissue,
            source="none",
            expressed=None,
            value=None,
            note="No expression data in GTEx or HPA for this gene/tissue.",
        )

    return TissueExpressionResult(
        gene_symbol=gene_symbol,
        tissue=tissue,
        source="none",
        expressed=None,
        value=None,
        note="No GTEx record, and no known Ensembl id for this gene to try HPA (likely a bridge "
        "gene that never appeared in Agent 1's Open Targets output).",
    )
