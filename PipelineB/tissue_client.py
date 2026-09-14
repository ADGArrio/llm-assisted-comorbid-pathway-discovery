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

BUG FIX #3, IN PROGRESS, 2026-09-14 (same day, next run again): the
plural-param fix above got further -- a real `422 Unprocessable Entity`
from `/reference/gene?geneIds=IRS1&genomeBuild=GRCh38%2Fhg38&itemsPerPage=10`,
a real validation rejection from the live server instead of a silent
empty result. Progress, but the exact cause isn't nailed down yet:
further research turned up a *third*, independently-implemented client
(ToolUniverse's Python `gtex_v2_tool`) that queries this same endpoint
with `geneId` (singular, matching bug-fix #1) and `gencodeVersion`, with
no `genomeBuild` parameter at all -- directly contradicting the
`geneIds`/`genomeBuild` combination bug-fix #2 took from the R client.
Two independently-maintained clients disagreeing, plus this sandbox
still unable to reach the live endpoint itself (network allowlist +
robots.txt), means the next attempt needs the actual 422 response body
(FastAPI/Pydantic-style validation errors normally name the exact
offending field) rather than a fourth guess from secondary sources.
`resp.raise_for_status()` was replaced with `_raise_for_status_with_body`
below specifically so that response body is captured automatically the
next time this fails, instead of needing a separate diagnostic run. Both
callers also now catch that error and degrade to "no data this run"
(printing the full message, body included) rather than letting it
propagate -- consistent with agent3_confirmation.py's existing
philosophy that Agent 3 findings are reported per-check, never a hard
pipeline failure; a broken request for one gene's tissue check
shouldn't take down Agents 1/2/4's results along with it.

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

BUG FIX #4, DEFINITIVE ROOT CAUSE FOUND, 2026-09-14 (same day, after
getting the user to run three standalone diagnostics directly against
the live API from their own machine, since this sandbox cannot reach
gtexportal.org at all -- see below): the persistent 0/18 result across
every prior round was caused by TWO separate bugs, now both confirmed
against the real, live API rather than inferred from secondary/
conflicting documentation:

(1) Bug-fix #2's "plural param names" theory was simply wrong. A live
call to `/reference/gene?geneIds=IRS1&genomeBuild=...` returned a real
`422 {"detail":[{"type":"missing","loc":["query","geneId"],"msg":"Field
required","input":null}]}` -- the server itself says the plural
`geneIds` is not a recognized param and the singular `geneId` is
required. A follow-up call with `geneId=IRS1` (singular) returned a
real `200` with IRS1's correct record. So bug-fix #1's original
singular param names (`geneId`, `gencodeId`, `tissueSiteDetailId`) were
right all along; bug-fix #2 regressed them based on a misreading of the
ropensci/gtexr R package's docs -- that package's R-level argument name
`geneIds` (which accepts an R vector, since R has no scalar/vector
distinction) does not mean the wire-level query parameter key is
plural. Its HTTP layer (httr2 with `.multi = "explode"`) actually sends
the *same singular* key once per value. This is reverted below: both
`_resolve_gencode_id` and `get_median_expression` go back to singular
`geneId`/`gencodeId`/`tissueSiteDetailId`.

(2) The real, previously-undiagnosed root cause of the 0/18 symptom
itself was `GTEX_DEFAULT_DATASET_ID = "gtex_v10"`. A live diagnostic
querying `/expression/medianGeneExpression` for IRS1's resolved
gencodeId in Pancreas with `datasetId=gtex_v10` returned a real `200`
but with `"data": []` -- a *silent* empty result, no error at all. The
exact same query with `datasetId=gtex_v8` instead returned `200` with a
real, correctly-shaped record (`median: 2.26513, unit: "TPM",
geneSymbol: "IRS1"`). So `gtex_v10` is not a valid/populated dataset id
on this endpoint (possibly not yet released on the live API, or named
differently) -- every single request this module ever made, across all
three prior "fixes," was silently filtered down to zero rows by this
one bad default, regardless of whether the gene resolved correctly.
This explains why the 0/18 symptom survived bug-fix #1 (which correctly
fixed gene resolution) and only changed shape under bug-fix #2 (which
additionally broke resolution itself, producing the 422). Fixed by
changing `GTEX_DEFAULT_DATASET_ID` to `"gtex_v8"`.

Both of these are now grounded in direct live-API responses obtained by
the user running short standalone diagnostics (this sandbox still
cannot reach gtexportal.org -- network allowlist blocks it, and
WebFetch is blocked by its robots.txt), not inferred from third-party
docs that turned out to disagree with each other and with the real
server. If a future run still shows unexpectedly many "no GTEx record"
results, the response bodies are now surfaced in the printed warning
(see `_raise_for_status_with_body`) -- check those first before
suspecting the param names or dataset id again.
"""
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

GTEX_DEFAULT_BASE_URL = "https://gtexportal.org/api/v2"
GTEX_DEFAULT_DATASET_ID = "gtex_v8"
# gtex_v10 looked like the natural choice (it's GTEx's newest release) but is NOT valid on this
# live endpoint -- see bug-fix #4 in the module docstring: datasetId=gtex_v10 returns a silent
# 200 {"data": []} for every gene/tissue combination, which is what caused the persistent 0/18
# result across every prior "fix" in this file's history. gtex_v8 is confirmed (via a live
# diagnostic) to return real data. gtex_v8 is built on the GRCh38/hg38 genome assembly -- passed
# explicitly when resolving a gene symbol to its gencodeId so the resolved id's genome build
# actually matches the dataset being queried (GTEx also serves a legacy GRCh37/hg19
# gencodeVersion by default, which would silently mismatch gtex_v8).
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


def _raise_for_status_with_body(resp) -> None:
    # Input: a requests.Response. Output: none, or raises requests.HTTPError with the response
    # body text folded into the message. Plain resp.raise_for_status() only shows the status
    # line ("422 Client Error: Unprocessable Entity for url: ...") and swallows the actual
    # validation detail a FastAPI-style API returns in the body (which query param was rejected,
    # and why) -- exactly the detail bug-fix #3 (see module docstring) needs and didn't have.
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f"{e} -- response body: {resp.text[:2000]!r}", response=resp) from e


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
            # Never let a broken request (wrong param name, a real GTEx-side error, etc.) crash
            # the whole pipeline over one gene's tissue check -- see agent3_confirmation.py's
            # module docstring: Agent 3 findings are reported per-check, never a hard failure.
            # The full error (now including the response body, see _raise_for_status_with_body)
            # is printed so this is diagnosable from a normal run's console output alone.
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
