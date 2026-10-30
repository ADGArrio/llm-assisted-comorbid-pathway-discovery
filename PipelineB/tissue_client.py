"""Stage 3 tissue-expression check against the GTEx Portal API v2 (no key needed). A gene symbol is first resolved to its
versioned gencodeId, then its median expression (nTPM) is read per tissue; a gene counts as expressed in a tissue at or above
EXPRESSION_THRESHOLD. Failed or rate-limited requests are reported as "no data", never as a confirmed absence.
"""
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

GTEX_DEFAULT_BASE_URL = "https://gtexportal.org/api/v2"
GTEX_DEFAULT_DATASET_ID = "gtex_v8"
# gtex_v8 is used because gtex_v10 returns an empty result for every gene on this endpoint. gtex_v8 uses genome build
# GRCh38/hg38, passed explicitly when resolving a gene symbol so the gencodeId matches the dataset.
GTEX_DEFAULT_GENOME_BUILD = "GRCh38/hg38"

# A gene counts as expressed in a tissue at or above this many nTPM (a common low-expression cutoff; a simplification,
# not an exact biological threshold).
EXPRESSION_THRESHOLD = 1.0

# Field names tried, in order, when parsing a GTEx record.
_GTEX_VALUE_FIELD_CANDIDATES = ("median", "value", "data")
_GTEX_GENE_FIELD_CANDIDATES = ("geneSymbol", "gene_symbol", "symbol")
_GTEX_GENCODE_ID_FIELD_CANDIDATES = ("gencodeId", "gencode_id")

# Client-side pacing between requests and 429 retry/backoff settings (conservative guesses; GTEx publishes no rate
# limit).
_REQUEST_PACING_SECONDS = 0.34  # ~3 requests/sec
_MAX_RATE_LIMIT_RETRIES = 4
_RATE_LIMIT_INITIAL_BACKOFF_SECONDS = 2.0


# One gene's expression in one tissue, with whether it counts as expressed.
@dataclass
class TissueExpressionResult:
    gene_symbol: str
    tissue: str
    source: str  # "GTEx" | "none"
    expressed: Optional[bool]  # None if GTEx had no record
    value: Optional[float]
    note: str

    # Output: the result as a plain dict.
    def to_dict(self) -> dict:
        return {
            "gene_symbol": self.gene_symbol,
            "tissue": self.tissue,
            "source": self.source,
            "expressed": self.expressed,
            "value": self.value,
            "note": self.note,
        }


# Input: one GTEx expression record. Output: its median value, or None if no known field matched.
def _parse_median_expression(record: dict) -> Optional[float]:
    for field in _GTEX_VALUE_FIELD_CANDIDATES:
        if field in record and isinstance(record[field], (int, float)):
            return float(record[field])
    return None


# Like raise_for_status but includes the response body in the error message. Input: a requests.Response. Output: none;
# raises requests.HTTPError on an error status.
def _raise_for_status_with_body(resp) -> None:
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f"{e} -- response body: {resp.text[:2000]!r}", response=resp) from e


# Queries the GTEx Portal API.
class GTExClient:
    # Input: optional base URL, dataset id, genome build and timeout (defaults from .env or module constants). Output:
    # none.
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
        # Gene symbol -> resolved gencodeId (None if unresolvable), cached per client.
        self._gencode_id_cache: Dict[str, Optional[str]] = {}

    # GET with client-side pacing and retries of 429 responses (honouring Retry-After, else exponential backoff). Input:
    # URL and params. Output: the response, which may still be a 429 after the last attempt; callers treat that as no
    # data.
    def _get(self, url: str, params: dict):
        backoff = _RATE_LIMIT_INITIAL_BACKOFF_SECONDS
        resp = None
        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            time.sleep(_REQUEST_PACING_SECONDS)
            resp = requests.get(url, params=params, timeout=self.timeout)
            if resp.status_code != 429:
                return resp
            if attempt == _MAX_RATE_LIMIT_RETRIES:
                return resp
            retry_after = resp.headers.get("Retry-After")
            wait_seconds = float(retry_after) if retry_after and retry_after.strip().isdigit() else backoff
            print(
                f"GTEx: 429 rate-limited on {url} -- retrying in {wait_seconds:.1f}s "
                f"(attempt {attempt + 1}/{_MAX_RATE_LIMIT_RETRIES})...",
                flush=True,
            )
            time.sleep(wait_seconds)
            backoff *= 2
        return resp

    # Look up a gene's versioned gencodeId for the configured genome build, which medianGeneExpression requires. Input:
    # gene symbol. Output: the id, or None if GTEx has no matching record or the request failed.
    def _resolve_gencode_id(self, gene_symbol: str) -> Optional[str]:
        if gene_symbol in self._gencode_id_cache:
            return self._gencode_id_cache[gene_symbol]

        resp = self._get(
            f"{self.base_url}/reference/gene",
            params={"geneId": gene_symbol, "genomeBuild": self.genome_build, "itemsPerPage": 10},
        )
        if resp.status_code == 429:
            print(
                f"GTEx: still rate-limited after {_MAX_RATE_LIMIT_RETRIES} retries resolving "
                f"{gene_symbol!r} -- treating as unresolved this run (not a real negative).",
                flush=True,
            )
            self._gencode_id_cache[gene_symbol] = None
            return None
        try:
            _raise_for_status_with_body(resp)
        except requests.HTTPError as e:
            # A failed request is reported and treated as unresolved rather than crashing the whole pipeline.
            print(f"GTEx: request failed resolving {gene_symbol!r} -- treating as unresolved this run. {e}", flush=True)
            self._gencode_id_cache[gene_symbol] = None
            return None
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

    # Input: gene symbol and GTEx tissueSiteDetailId (e.g. "Pancreas"). Output: median nTPM, or None if GTEx has no
    # record, the gene did not resolve, or the request was still rate-limited after retries (distinct from a measured
    # 0.0).
    def get_median_expression(self, gene_symbol: str, tissue_site_detail_id: str) -> Optional[float]:
        gencode_id = self._resolve_gencode_id(gene_symbol)
        if gencode_id is None:
            return None

        resp = self._get(
            f"{self.base_url}/expression/medianGeneExpression",
            params={
                "gencodeId": gencode_id,
                "tissueSiteDetailId": tissue_site_detail_id,
                "datasetId": self.dataset_id,
                "itemsPerPage": 5,
            },
        )
        if resp.status_code == 429:
            print(
                f"GTEx: still rate-limited after {_MAX_RATE_LIMIT_RETRIES} retries checking "
                f"{gene_symbol!r} in {tissue_site_detail_id!r} -- treating as 'no data' this run.",
                flush=True,
            )
            return None
        try:
            _raise_for_status_with_body(resp)
        except requests.HTTPError as e:
            print(
                f"GTEx: request failed checking {gene_symbol!r} in {tissue_site_detail_id!r} -- "
                f"treating as 'no data' this run. {e}",
                flush=True,
            )
            return None
        payload = resp.json()
        records = payload.get("data") or []
        if not records:
            return None
        return _parse_median_expression(records[0])


# Test double mapping "gene|tissue" to a median value; omitted keys mean no data. No network call.
class StubGTExClient:

    # Input: the expression map. Output: none.
    def __init__(self, expression: Dict[str, float]):
        self.expression = expression
        self.calls: List[tuple] = []

    # Input: gene symbol and tissue id. Output: the mapped value, or None.
    def get_median_expression(self, gene_symbol: str, tissue_site_detail_id: str) -> Optional[float]:
        self.calls.append((gene_symbol, tissue_site_detail_id))
        return self.expression.get(f"{gene_symbol}|{tissue_site_detail_id}")


# Check one gene in one tissue. Input: gene symbol, tissue id and a GTEx client. Output: a TissueExpressionResult.
def check_tissue_expression(gene_symbol: str, tissue: str, gtex_client) -> TissueExpressionResult:
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
