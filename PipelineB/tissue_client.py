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
see `_resolve_gencode_id` below.

BUG FIX #2, 2026-09-14 (same day, next run): the fix above still came
back 0/18 on the next real run -- for INSR, PTPN1, CAPN1, OGG1, all of
which have real GTEx data. Root cause: the query parameter names above
were still wrong, just wrong in a new way -- `geneId` and `gencodeId`/
`tissueSiteDetailId` are singular, but GTEx's v2 API actually expects
the *plural* forms `geneIds`, `gencodeIds`, `tissueSiteDetailIds` (a
single value under the plural key still works fine -- these are
"character vector" params, and one gene is a length-1 vector). The
mirrored OpenAPI spec used for the first fix (from an older
broadinstitute repo, itself only documenting the legacy v6p/v7 datasets)
was simply out of date. This was re-confirmed via GTEx's own actively
maintained R client, ropensci/gtexr (docs.ropensci.org/gtexr and its
source on GitHub), which sends `geneIds=`/`gencodeIds=`/
`tissueSiteDetailIds=` and documents the exact response field names
this module already parses for (`gencodeId`, `geneSymbol`, `median` --
response fields stay singular even though request params are plural).
A silently-ignored/unrecognized query param is why the first fix's
`/reference/gene` call kept returning either zero or unrelated records
instead of erroring -- there was nothing to catch. Still not verified
against the live endpoint from this sandbox (network allowlist blocks
gtexportal.org, WebFetch blocked by its robots.txt); if a real run still
comes back with unexpectedly many "no GTEx record" results after this,
run `GTExClient()._resolve_gencode_id("INSR")` and print the raw
response (`requests.get(f"{client.base_url}/reference/gene",
params={"geneIds": "INSR", "genomeBuild": "GRCh38/hg38"}).json()`)
before assuming the genes really do lack data.

RATE-LIMIT HARDENING, 2026-09-14 (same day): the equivalent fix in this
module's sibling, crossdb_client.py (GWAS Catalog), made that client's
own success its next problem -- more real HTTP requests per gene tripped
GWAS Catalog's rate limit and crashed the whole pipeline with an
uncaught 429. This module makes two requests per gene per tissue
(gencodeId resolution + expression lookup, though resolution is cached
per gene), which carries the same risk on a bigger run even though it
hasn't actually failed this way yet. Applied proactively, before it did:
`_get` paces every request and transparently retries a 429 (honoring a
Retry-After header, else exponential backoff) up to
_MAX_RATE_LIMIT_RETRIES times; if still rate-limited after retries, that
gene/tissue check degrades to `None` ("no data this time") instead of
raising and taking down the whole run -- same philosophy as GWAS
Catalog's fix, see crossdb_client.py's bug-fix #2 note for the full
reasoning.
"""
import os
import time
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

# Client-side pacing between requests, and 429 retry/backoff settings -- see module docstring's
# rate-limit-hardening note. GTEx doesn't publish a documented rate limit, so these are
# conservative guesses (matching crossdb_client.py's), not a verified real limit.
_REQUEST_PACING_SECONDS = 0.34  # ~3 requests/sec
_MAX_RATE_LIMIT_RETRIES = 4
_RATE_LIMIT_INITIAL_BACKOFF_SECONDS = 2.0


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

    def _get(self, url: str, params: dict):
        # Input: request url/params. Output: the requests.Response, after this client's own
        # pacing delay (_REQUEST_PACING_SECONDS) and after transparently retrying a 429 up to
        # _MAX_RATE_LIMIT_RETRIES times (honoring a Retry-After header when the server sends
        # one, else exponential backoff) -- see module docstring's rate-limit-hardening note.
        # Callers get the response back even if it's still a 429 on the final attempt; both
        # callers below treat that the same as "no data" rather than raising.
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

    def _resolve_gencode_id(self, gene_symbol: str) -> Optional[str]:
        # Input: gene symbol. Output: its versioned gencodeId for self.genome_build, or None if
        # GTEx's own gene reference has no matching record (including "still rate-limited after
        # retries") -- see module docstring's bug-fix note: medianGeneExpression requires this
        # versioned id, a plain gene symbol always returns zero records.
        if gene_symbol in self._gencode_id_cache:
            return self._gencode_id_cache[gene_symbol]

        resp = self._get(
            f"{self.base_url}/reference/gene",
            params={"geneIds": gene_symbol, "genomeBuild": self.genome_build, "itemsPerPage": 10},
        )
        if resp.status_code == 429:
            print(
                f"GTEx: still rate-limited after {_MAX_RATE_LIMIT_RETRIES} retries resolving "
                f"{gene_symbol!r} -- treating as unresolved this run (not a real negative).",
                flush=True,
            )
            self._gencode_id_cache[gene_symbol] = None
            return None
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
        # including the case where gene_symbol itself doesn't resolve to a gencodeId at all, or
        # is still rate-limited after retries.
        gencode_id = self._resolve_gencode_id(gene_symbol)
        if gencode_id is None:
            return None

        resp = self._get(
            f"{self.base_url}/expression/medianGeneExpression",
            params={
                "gencodeIds": gencode_id,
                "tissueSiteDetailIds": tissue_site_detail_id,
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
