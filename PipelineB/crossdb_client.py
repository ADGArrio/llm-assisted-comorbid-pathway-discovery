"""
Agent 3, cross-database gene-disease association check -- GWAS Catalog.

Service: GWAS Catalog REST API (https://www.ebi.ac.uk/gwas/rest/api,
finder path /singleNucleotidePolymorphisms/search/findByGene -- the v1
HAL pattern; the exact v1/v2 path couldn't be confirmed live from this
sandbox, so this may 404 on a real run -- check
https://www.ebi.ac.uk/gwas/rest/docs/api if so).
Data: HAL JSON (_embedded.singleNucleotidePolymorphisms -> associations
-> efoTraits). This is real, numerical GWAS-study evidence (p-values,
study accessions) rather than another literature-aggregation source --
Agent 1's PubMed citations already cover literature evidence.
Used by: PipelineB.agent3_confirmation, to independently confirm a
chain's two disease-anchored endpoint genes against a source Open
Targets doesn't itself aggregate.

Simplified 2026-09-14 to GWAS Catalog only -- DisGeNET was dropped (see
claude/agent3_simplification.md); the removed DisGeNETClient/
StubDisGeNETClient/DisgenetAssociation code is still in git history if
needed later.

BUG FIX, 2026-09-14: a real T2DM->AD run came back with 0/6 endpoint
checks confirmed -- including well-studied genes (INSR, IRS1, GLP1R)
that certainly do have real GWAS Catalog hits. Root cause: findByGene's
response nests SNPs under `_embedded.singleNucleotidePolymorphisms`, but
(per this API's HAL conventions -- corroborated by multiple independent
third-party usage reports, e.g. https://www.biostars.org/p/9466494/ and
https://www.biostars.org/p/438387/, since this sandbox can't reach the
live endpoint itself: network allowlist blocks it, and WebFetch is
blocked by ebi.ac.uk's robots.txt) each SNP entry does NOT embed its
`associations`/`efoTraits` data inline -- it only carries a `_links`
object pointing to where that data actually lives. The old code read
`snp.get("_embedded", {}).get("associations", snp.get("associations", []))`,
which is always `[]` for a response shaped this way, so every gene came
back "not confirmed" regardless of real underlying evidence. Fixed by
following each SNP's own `_links.associations.href` (standard HAL
"embed on request only" pattern) and parsing *that* response's
`_embedded.associations` instead -- see `_associations_for_snp` below.
The original `efoTraits`/`trait`/`shortForm` parsing is kept as-is for
an individual association resource, since that part of the schema is
well documented and wasn't implicated by this bug.

BUG FIX #2, 2026-09-14 (same day, next run): the fix above made this
client's own success its next problem -- following one HTTP request per
SNP (up to 20 per gene) made a real T2DM->AD run's Agent 3 step hit
`429 Too Many Requests` from GWAS Catalog and crash the whole pipeline
(uncaught `requests.HTTPError`, right after the fix above was pushed).
Three changes address this:
  1. `_get` now paces every request (`_REQUEST_PACING_SECONDS` between
     calls) instead of bursting them, and transparently retries a 429 up
     to `_MAX_RATE_LIMIT_RETRIES` times, honoring the server's own
     `Retry-After` header when present (falling back to exponential
     backoff otherwise).
  2. `_associations_for_snp` now stops following further SNPs for a gene
     as soon as it finds one matching association -- `confirmed` only
     needs one hit to be True, so paying for up to 20 requests to a gene
     that already matched on the first SNP was pure waste under a real
     rate limit. This does mean `gwas_associations` may not list every
     matching study once a gene is already confirmed -- a real
     completeness/robustness tradeoff, made deliberately after the 429
     crash, not an oversight.
  3. If a request is still rate-limited after all retries, that
     gene/SNP is skipped (logged, not crashed) and treated the same as
     "no data" -- consistent with this module's existing philosophy that
     an absent GWAS Catalog hit is inconclusive, not a real negative
     (see module docstring above). A transient rate limit is an even
     weaker signal than that, so it must never be allowed to crash the
     whole run or to be mistaken for `confirmed=False` meaning "checked,
     not found" when it really means "not checked this time."
Still not verified against the live endpoint from this sandbox (network
allowlist blocks it) -- if a real run logs many "still rate-limited
after retries" warnings, GWAS Catalog's real limit is stricter than
`_REQUEST_PACING_SECONDS` assumes; widen the pacing/backoff constants
below rather than removing the retry logic.
"""
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

GWAS_DEFAULT_BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api"
_ASSOCIATIONS_BY_GENE_PATH = "/singleNucleotidePolymorphisms/search/findByGene"
# Cap on how many of a gene's SNPs get a follow-up associations request -- a highly studied gene
# can turn up hundreds of SNPs, and each now costs its own HTTP round trip (see bug-fix note
# above), so this bounds worst-case request count per gene/disease check. findByGene's own
# default page size is 20, so this doesn't cut into what a single search page already returns.
# In practice _associations_for_snp stops early once a match is found (see bug-fix #2 note),
# so this cap is really only exercised in full by genes with no matching evidence at all.
_MAX_SNPS_PER_GENE_FOR_ASSOCIATIONS = 20
# Client-side pacing between requests, and 429 retry/backoff settings -- see bug-fix #2 note.
# GWAS Catalog doesn't publish a documented rate limit, so these are conservative guesses, not
# a verified real limit.
_REQUEST_PACING_SECONDS = 0.34  # ~3 requests/sec
_MAX_RATE_LIMIT_RETRIES = 4
_RATE_LIMIT_INITIAL_BACKOFF_SECONDS = 2.0


@dataclass
class GwasAssociation:
    gene_symbol: str
    trait_name: str
    efo_id: Optional[str]
    pvalue: Optional[float]
    study_accession: Optional[str]

    def to_dict(self) -> dict:
        return {
            "gene_symbol": self.gene_symbol,
            "trait_name": self.trait_name,
            "efo_id": self.efo_id,
            "pvalue": self.pvalue,
            "study_accession": self.study_accession,
        }


def _trait_matches_disease(trait_name: str, disease_name: str) -> bool:
    # Input: a GWAS Catalog trait string, this project's disease name.
    # Output: loose substring match (GWAS Catalog trait text isn't a controlled vocabulary) --
    # a first-pass filter; review matches by hand, don't trust this alone.
    disease_key_terms = {
        "Type 2 Diabetes Mellitus": ["type 2 diabetes", "type ii diabetes", "t2dm", "t2d"],
        "Alzheimer's Disease": ["alzheimer"],
        "Coronary Artery Disease": ["coronary artery disease", "coronary heart disease", "cad"],
    }
    terms = disease_key_terms.get(disease_name, [disease_name.lower()])
    trait_lower = trait_name.lower()
    return any(term in trait_lower for term in terms)


class GWASCatalogClient:
    def __init__(self, base_url: str = None, timeout: int = 30):
        self.base_url = base_url or os.environ.get("GWAS_CATALOG_BASE_URL", GWAS_DEFAULT_BASE_URL)
        self.timeout = timeout

    def _get(self, url: str, params: dict = None):
        # Input: request url/params. Output: the requests.Response, after this client's own
        # pacing delay (_REQUEST_PACING_SECONDS) and after transparently retrying a 429 up to
        # _MAX_RATE_LIMIT_RETRIES times (honoring a Retry-After header when the server sends
        # one, else exponential backoff) -- see module docstring's bug-fix #2 note. Callers
        # still get the response back even if it's a 429 on the final attempt -- they decide
        # whether to raise or degrade gracefully (this module always degrades; see callers).
        backoff = _RATE_LIMIT_INITIAL_BACKOFF_SECONDS
        resp = None
        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            time.sleep(_REQUEST_PACING_SECONDS)
            resp = requests.get(url, params=params, timeout=self.timeout, headers={"Accept": "application/json"})
            if resp.status_code != 429:
                return resp
            if attempt == _MAX_RATE_LIMIT_RETRIES:
                return resp
            retry_after = resp.headers.get("Retry-After")
            wait_seconds = float(retry_after) if retry_after and retry_after.strip().isdigit() else backoff
            print(
                f"GWAS Catalog: 429 rate-limited on {url} -- retrying in {wait_seconds:.1f}s "
                f"(attempt {attempt + 1}/{_MAX_RATE_LIMIT_RETRIES})...",
                flush=True,
            )
            time.sleep(wait_seconds)
            backoff *= 2
        return resp

    def get_gene_disease_associations(self, gene_symbol: str, disease_name: str) -> List[GwasAssociation]:
        # Input: gene symbol, disease name. Output: associations whose trait text matches the
        # disease (see _trait_matches_disease). [] means no hits, no match, or still rate-limited
        # after retries -- none of these are treated as an error (see module docstring).
        resp = self._get(f"{self.base_url}{_ASSOCIATIONS_BY_GENE_PATH}", params={"geneName": gene_symbol})
        if resp.status_code == 404:
            raise RuntimeError(
                f"GWAS Catalog: {_ASSOCIATIONS_BY_GENE_PATH} 404'd for gene {gene_symbol!r} -- the "
                f"v1/v2 endpoint shape may have changed. Check https://www.ebi.ac.uk/gwas/rest/docs/api "
                f"and update crossdb_client.py's _ASSOCIATIONS_BY_GENE_PATH."
            )
        if resp.status_code == 429:
            print(
                f"GWAS Catalog: still rate-limited after {_MAX_RATE_LIMIT_RETRIES} retries for gene "
                f"{gene_symbol!r} -- skipping this gene's check this run (treated as 'not checked', "
                f"not a real negative).",
                flush=True,
            )
            return []
        resp.raise_for_status()
        payload = resp.json()

        results: List[GwasAssociation] = []
        snps = payload.get("_embedded", {}).get("singleNucleotidePolymorphisms", [])
        for snp in snps[:_MAX_SNPS_PER_GENE_FOR_ASSOCIATIONS]:
            results.extend(self._associations_for_snp(snp, gene_symbol, disease_name))
            if results:
                # One matching association is enough to prove `confirmed=True` -- stop paying
                # for further per-SNP follow-up requests to this gene. See bug-fix #2 note.
                break

        # Defensive fallback for the other plausible HAL shape (associations embedded directly
        # at the top level rather than nested per-SNP) -- see bug-fix note in the module
        # docstring on why the exact live shape couldn't be confirmed from this sandbox.
        if not results:
            for assoc in payload.get("_embedded", {}).get("associations", []):
                results.extend(self._parse_association(assoc, gene_symbol, disease_name))

        return results

    def _associations_for_snp(self, snp: dict, gene_symbol: str, disease_name: str) -> List[GwasAssociation]:
        # Input: one SNP entry from the findByGene response. Output: its matching associations,
        # or [] if this SNP has no HAL link to follow, or if the follow-up request is still
        # rate-limited after retries (see module docstring's bug-fix #2 note -- never raises for
        # a rate limit; the caller treats [] as "no evidence found here" either way).
        embedded = snp.get("_embedded", {})
        raw_associations = embedded.get("associations")
        if raw_associations is None and "associations" in snp:
            raw_associations = snp["associations"]  # older/alternate flat shape, if it ever appears

        if raw_associations is None:
            href = snp.get("_links", {}).get("associations", {}).get("href")
            if not href:
                return []
            assoc_resp = self._get(href)
            if assoc_resp.status_code == 429:
                print(
                    f"GWAS Catalog: still rate-limited after {_MAX_RATE_LIMIT_RETRIES} retries following "
                    f"{href} -- skipping this SNP.",
                    flush=True,
                )
                return []
            assoc_resp.raise_for_status()
            raw_associations = assoc_resp.json().get("_embedded", {}).get("associations", [])

        results: List[GwasAssociation] = []
        for assoc in raw_associations:
            results.extend(self._parse_association(assoc, gene_symbol, disease_name))
        return results

    def _parse_association(self, assoc: dict, gene_symbol: str, disease_name: str) -> List[GwasAssociation]:
        # Input: one association resource (embedded or fetched separately). Output: the
        # GwasAssociation entries among its efoTraits that match disease_name.
        results: List[GwasAssociation] = []
        for efo_trait in assoc.get("efoTraits", []):
            trait_name = efo_trait.get("trait", "")
            if not _trait_matches_disease(trait_name, disease_name):
                continue
            results.append(
                GwasAssociation(
                    gene_symbol=gene_symbol,
                    trait_name=trait_name,
                    efo_id=efo_trait.get("shortForm"),
                    pvalue=assoc.get("pvalue"),
                    study_accession=assoc.get("studyAccession"),
                )
            )
        return results


class StubGWASCatalogClient:
    """(gene_symbol, disease_name) -> list of GwasAssociation. No network call."""

    def __init__(self, associations: Dict[tuple, List[GwasAssociation]]):
        self.associations = associations
        self.calls: List[tuple] = []

    def get_gene_disease_associations(self, gene_symbol: str, disease_name: str) -> List[GwasAssociation]:
        self.calls.append((gene_symbol, disease_name))
        return self.associations.get((gene_symbol, disease_name), [])
