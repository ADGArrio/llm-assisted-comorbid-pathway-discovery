"""
Stage 3, cross-database gene-disease association check -- GWAS Catalog.

Service: GWAS Catalog REST API v2 (https://www.ebi.ac.uk/gwas/rest/api/v2), one endpoint:
  GET /associations?mapped_gene=<SYMBOL>&efo_trait=<exact EFO trait name>&size=100&page=N
Data: each association carries p_value, efo_traits [{efo_id, efo_trait}], mapped_genes,
accession_id (the GCST study accession), pubmed_id and the SNP alleles. This is real, numerical
GWAS-study evidence rather than another literature-aggregation source -- Stage 1's PubMed
citations already cover literature evidence.
Used by: PipelineB.stage3_confirmation, to independently confirm a chain's two disease-anchored
endpoint genes against a source Open Targets does not itself aggregate.

MIGRATION, 2026-10-18 (v1 -> v2). The v1 endpoint this module was built on
(/singleNucleotidePolymorphisms/search/findByGene) now answers HTTP 410 Gone -- EBI retired it. A
live run of the whole pipeline crashed Stage 3 on its first gene (FGFR4) with exactly that error.
It failed LOUDLY (HTTPError), which is the right failure mode: it did not silently report "no
GWAS evidence". The v2 API is much simpler: one request returns the associations themselves, with
traits inline, instead of the v1 chain gene -> SNPs -> associations -> efoTraits (up to ~30
requests per gene). Consequences:
  * The v1 workarounds (follow each SNP's _links, de-duplicate rsIds, follow each association's
    efoTraits link, walk up to 10 SNP pages) are gone. The pacing / 429-retry / connection-failure
    handling in `_get` is kept unchanged.
  * `study_accession` is now actually filled in (it was always None under v1).
  * Trait matching is done SERVER-SIDE by exact EFO trait name (DISEASE_GWAS_TRAITS below) and
    re-checked client-side. Verified live 2026-10-18 (counts of associations in the whole
    Catalog): "type 2 diabetes mellitus" 8,864; "Alzheimer disease" 6,852; "coronary artery
    disorder" 3,913. The old label "coronary artery disease" now returns ZERO -- the Catalog's
    ontology renamed that trait to "coronary artery disorder", so the same name-matching would
    silently have found nothing for every CAD gene. Narrower traits ("coronary atherosclerosis",
    "myocardial infarction") are deliberately NOT included: they are different phenotypes.
  * The old client-side term list contained the substring "cad" (would match e.g. "cadmium") and
    "t2d"; terms are now matched on word boundaries.
  * Only the 10 strongest (lowest p-value) associations per gene/disease are returned, to keep
    the saved JSON small (APOE x Alzheimer has 138). Whether ANY exists is what `confirmed` uses.
  * Behaviour change to remember when comparing runs: results from before this date came from the
    v1 API and a different EFO vocabulary; do not mix them with new runs without saying so.

HISTORY of the v1 client (2026-09-14 .. 2026-09-17), kept because it explains the pacing and
rate-limit design that survives: BUG FIX: a real T2DM->AD run came back with 0/6 endpoint
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
SNP (up to 20 per gene) made a real T2DM->AD run's Stage 3 step hit
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

BUG FIX #3, 2026-09-14 (same day, after the GTEx fix in tissue_client.py
was confirmed working live -- see that module's docstring's bug-fix #4):
a full T2DM->AD run's console output showed exactly the warning sign
flagged above -- a long run of "429 rate-limited... retrying... (attempt
4/4)" messages across a dozen-plus distinct SNPs, not an isolated blip.
GWAS Catalog's real sustained limit is clearly stricter than the ~3
req/sec this module was pacing at. This matters beyond just wasted
retries: `get_gene_disease_associations` returned `[]` for every one of
6 endpoint checks that run, including IRS1 x Type 2 Diabetes Mellitus --
IRS1 is one of the most heavily replicated common-variant T2DM GWAS loci
in the literature (rs2943641 and related SNPs, first reported ~2009, a
DIAGRAM-consortium-era finding since replicated many times over), so a
clean "GWAS Catalog has zero matching evidence for IRS1 and T2DM" is
biologically implausible. The far more likely explanation, given the
log: some of IRS1's SNP follow-up requests (or the gene's own
associations, if the true-hit SNP simply wasn't among the ones that
happened to clear rate-limiting before `_MAX_RATE_LIMIT_RETRIES` ran
out) were silently dropped by the existing graceful-degradation path --
which is by design not an error, but was, until this fix, also
indistinguishable downstream from "checked cleanly, found nothing."

Two changes:
  1. Pacing/backoff widened -- `_REQUEST_PACING_SECONDS` 0.34s -> 1.0s
     (roughly matching the "be polite" defaults other public bio-data
     APIs like NCBI E-utilities recommend), `_RATE_LIMIT_INITIAL_BACKOFF_SECONDS`
     2.0s -> 5.0s, `_MAX_RATE_LIMIT_RETRIES` 4 -> 5 -- slower and more
     patient, so fewer genuine associations get lost to a still-live
     rate-limit window rather than a real absence of evidence. This is a
     one-directional, safe change (a client is always allowed to go
     slower than a server's real limit); it isn't a guess about the
     exact number GWAS Catalog enforces, unlike the earlier param-name
     rounds in tissue_client.py's history.
  2. `GWASCatalogClient.last_check_rate_limited` -- a plain bool
     attribute, reset to `False` at the start of every
     `get_gene_disease_associations` call and set `True` if *any* part
     of that check (the top-level search, or any per-SNP follow-up) gave
     up after exhausting retries. Callers that care whether a `[]`/
     `confirmed=False` result might actually be "not fully checked" now
     have a way to tell -- see stage3_confirmation.py's
     `EndpointCrossDbCheck.possibly_incomplete` and main.py's readable
     report, which both surface this rather than silently reporting a
     rate-limit artifact as if it were a real negative finding. This
     matters specifically because the pipeline exits the per-SNP loop
     early on a match (bug-fix #2's design) -- a gene with real evidence
     usually finds it fast and never risks exhausting retries, but a
     gene that's rate-limited into looking like it has *no* evidence
     pays the full cost of checking every SNP, so this failure mode is
     concentrated exactly on the negative results that most need to be
     trustworthy.
`StubGWASCatalogClient` also gets a `last_check_rate_limited` attribute
(always `False`, since it never makes real requests) purely so code that
reads it works the same against either client without an `isinstance`
check or a `getattr` default at every call site outside
stage3_confirmation.py (which still uses `getattr(..., False)` defensively
for any third client that doesn't define the attribute at all).

BUG FIX #4, 2026-09-14 (same day, after bug-fix #3's pacing widening):
the next full run showed *zero* "rate-limited before finishing this
check" notes anywhere -- bug-fix #3 worked, GWAS Catalog stopped cutting
checks short -- but the 0/6 result, including IRS1 x Type 2 Diabetes
Mellitus, didn't change at all. With rate-limiting ruled out, the user
ran a direct diagnostic against the live endpoint (this sandbox still
can't reach it) that finally explains why, with real data:

  GET /singleNucleotidePolymorphisms/search/findByGene?geneName=IRS1
  -> 200, page info {"size": 20, "totalElements": 115, "totalPages": 6,
     "number": 0} -- IRS1 has 115 matching SNP entries across 6 pages,
     and this module only ever requested page 0 (no `page` param sent at
     all). The 20 entries on that one page were also only 10 *distinct*
     rsIds -- every rsId appeared twice in a row (e.g. "rs7558386",
     "rs7558386", "rs10933153", "rs10933153", ...) -- a real, consistent
     quirk of this endpoint's response, not a diagnostic artifact (both
     entries for a given rsId look identical on inspection; the cause
     wasn't tracked down further, but deduping by rsId is a safe
     mitigation regardless of why it happens). So the old
     `snps[:_MAX_SNPS_PER_GENE_FOR_ASSOCIATIONS]` slice was checking only
     ~10 unique SNPs out of 115 real candidates, and paying for 20 HTTP
     requests to do it (half of them wasted on exact duplicates).
  GET /singleNucleotidePolymorphisms/rs2943641/associations (queried
  directly, bypassing findByGene) -> 200, a real association: pvalue
  9e-12, authorReportedGenes including IRS1 (ensemblGeneId
  ENSG00000169047, matching the gencodeId Round 5 resolved for this same
  gene in tissue_client.py). rs2943641 is IRS1's well-known T2DM GWAS
  SNP -- and it was nowhere in the 10 unique rsIds page 0 returned, so
  this module could never have found it no matter how the per-SNP
  parsing or trait-matching logic looked. (The diagnostic's printed
  output was truncated before showing that association's efoTraits text
  directly, so this fix doesn't depend on assuming what specific trait
  string it carries -- only on the now-confirmed fact that the SNP
  itself was unreachable before this fix, which is enough on its own to
  explain a false negative.)

Fixed by making `get_gene_disease_associations` walk pages (`page=0`,
`page=1`, ... -- the standard Spring Data REST HAL pagination param,
and exactly the field name the response's own `page.number` already
uses) instead of only ever reading page 0, and dedup SNP entries by
`rsId` (within and across pages) so a repeated entry doesn't cost a
second follow-up request. `_MAX_SNPS_PER_GENE_FOR_ASSOCIATIONS` now
means *distinct* SNPs checked, not raw entries sliced from one page.
Paging stops as soon as: a match is found (early-exit, unchanged from
bug-fix #2), the distinct-SNP cap is reached, the live response's own
`page.totalPages` says there's nothing more, or `_MAX_SEARCH_PAGES_PER_GENE`
search-page requests have been made (a safety cap independent of
`totalPages`, in case a gene has far more pages than are worth the
request budget to walk). A `page` key absent from a test's fake
response (every existing test predates this fix) defaults `totalPages`
to `1`, so old single-page test fixtures keep behaving exactly as
before without modification.

3 new tests (dedup within a page, pagination to a later page, and a safety cap on total pages
walked), 117 passing (was 114).

BUG FIX #5, 2026-09-14 (same day, right after bug-fix #4's pagination
fix landed): the next full run -- confirmed via a fresh-process
diagnostic to rule out a stale cached import, not a guess -- still came
back 0/6, IRS1 x Type 2 Diabetes Mellitus included, with
`last_check_rate_limited=False` (ruling out rate limiting too). A direct
fetch of rs2943641's (IRS1's well-known T2DM SNP, see bug-fix #4)
associations found 32 real associations -- and every single one showed
`efoTraits: []` when read via this module's own `_trait_matches_disease`
check. `efoTraits` is simply never embedded inline on a live association
resource. This is the exact same HAL "embed on request only" pattern bug
fix #1 already found and fixed one level up (SNP -> associations) -- it
just turns out the pattern goes one level deeper than assumed. The
original module docstring even said so explicitly: "the efoTraits/
trait/shortForm parsing is kept as-is for an individual association
resource, since that part of the schema is well documented and wasn't
implicated by this bug." That assumption was wrong, and this is now
directly proven: a user-run diagnostic followed one association's own
`_links.efoTraits.href` (`.../associations/{id}/efoTraits`) and got back
`{"_embedded": {"efoTraits": [{"trait": "type 2 diabetes mellitus",
"shortForm": "MONDO_0005148", ...}]}}` -- the real trait data, one hop
away from where this module was looking. (Also notable: this trait's
`shortForm` is a MONDO accession, not an EFO one, despite the field name
-- harmless here since the code only stores it as an opaque id string,
never parses it as EFO-specific.)

Fixed by making `_parse_association` follow `_links.efoTraits.href` when
`efoTraits` isn't already present on the association dict (mirroring
`_associations_for_snp`'s existing embed-or-follow-link pattern exactly)
-- a test or a future response shape that already embeds `efoTraits`
directly still works unchanged, since the follow-the-link branch only
runs when the key is genuinely absent. This adds one more HTTP request
per association actually examined, which matters: a single busy SNP can
have dozens of associations (32 for rs2943641 alone), so
`_associations_for_snp` now also stops iterating a SNP's own association
list as soon as one match is found -- the same early-exit philosophy
bug-fix #2 already applies across SNPs, now applied within one SNP's
associations too, to avoid needlessly re-triggering the rate-limiting
bug-fix #3 fixed.

One known, deliberately deferred gap: a live association resource has no
top-level `studyAccession` field (only an unfollowed `_links.study.href`
pointer), so `GwasAssociation.study_accession` is always `None` in
practice. Left alone since nothing downstream reads it yet (main.py's
readable report only uses `trait_name`); follow that link too if a real
future need for it surfaces.

3 new tests (following the efoTraits link, stopping early once matched,
and degrading gracefully if that follow-up is rate-limited), 120
passing (was 117).

BUG FIX #6, 2026-09-17: a real T2DM->CAD run crashed with an uncaught
`requests.exceptions.ConnectTimeout` reaching www.ebi.ac.uk (30s timeout,
no response at all -- not a 429). `_get`'s retry/backoff loop only
handles a real HTTP 429 *response*; a connection-level failure (timeout,
DNS hiccup, reset) raises straight out of `requests.get` before a
response object even exists, so none of bug-fixes #2/#3's retry logic
ever saw it, and it took the whole run down instead of degrading
gracefully like every other failure mode this module already handles.

Fixed by wrapping the `requests.get` call itself in a
`try/except requests.exceptions.RequestException`, using the exact same
`_MAX_RATE_LIMIT_RETRIES`/backoff loop as the 429 path. If it still
hasn't succeeded after retries, `_get` returns a synthetic
`requests.Response` with `status_code = 429` rather than raising --
every caller already has "if resp.status_code == 429: log, set
last_check_rate_limited, treat as not-checked" handling from bug-fix #2,
so this reuses that existing degrade-gracefully path instead of adding a
second one. A connection failure is conceptually the same as a
rate-limit failure for this module's purposes: neither means "checked,
found nothing," both mean "couldn't check this time."
"""

import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

GWAS_DEFAULT_BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api/v2"
_ASSOCIATIONS_PATH = "/associations"
_PAGE_SIZE = 100
_MAX_PAGES_PER_TRAIT = 5  # safety cap: 500 associations per gene/trait is far more than needed
_MAX_ASSOCIATIONS_RETURNED = 10  # strongest (lowest p-value) kept in the result

# disease name -> exact GWAS Catalog EFO trait names to query. See the 2026-10-18 migration note.
DISEASE_GWAS_TRAITS = {
    "Type 2 Diabetes Mellitus": ["type 2 diabetes mellitus"],
    "Alzheimer's Disease": ["Alzheimer disease"],
    "Coronary Artery Disease": ["coronary artery disorder"],
}

# Client-side pacing between requests, and 429 retry/backoff settings -- see the v1 history above.
_REQUEST_PACING_SECONDS = 1.0
_MAX_RATE_LIMIT_RETRIES = 5
_RATE_LIMIT_INITIAL_BACKOFF_SECONDS = 5.0


@dataclass
class GwasAssociation:
    gene_symbol: str
    trait_name: str
    efo_id: Optional[str]
    pvalue: Optional[float]
    study_accession: Optional[str]
    pubmed_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "gene_symbol": self.gene_symbol,
            "trait_name": self.trait_name,
            "efo_id": self.efo_id,
            "pvalue": self.pvalue,
            "study_accession": self.study_accession,
            "pubmed_id": self.pubmed_id,
        }


_DISEASE_KEY_TERMS = {
    "Type 2 Diabetes Mellitus": ["type 2 diabetes", "type ii diabetes", "t2dm", "t2d"],
    "Alzheimer's Disease": ["alzheimer"],
    "Coronary Artery Disease": ["coronary artery disease", "coronary artery disorder", "coronary heart disease"],
}


def _trait_matches_disease(trait_name: str, disease_name: str) -> bool:
    # Input: a GWAS Catalog trait string, this project's disease name.
    # Output: True if the trait text contains one of the disease's key terms as whole words. A
    # first-pass filter on top of the server-side exact-trait query; review matches by hand.
    terms = _DISEASE_KEY_TERMS.get(disease_name, [disease_name.lower()])
    trait_lower = trait_name.lower()
    return any(re.search(r"\b" + re.escape(term) + r"\b", trait_lower) for term in terms)


class GWASCatalogClient:
    def __init__(self, base_url: str = None, timeout: int = 30):
        self.base_url = base_url or os.environ.get("GWAS_CATALOG_BASE_URL", GWAS_DEFAULT_BASE_URL)
        self.timeout = timeout
        # True if the most recent get_gene_disease_associations call gave up after exhausting
        # retries (rate limit or connection failure) -- "not fully checked", NOT a clean negative.
        self.last_check_rate_limited = False

    def _get(self, url: str, params: dict = None):
        # Input: request url/params. Output: the requests.Response, after this client's own
        # pacing delay (_REQUEST_PACING_SECONDS) and after transparently retrying a 429 up to
        # _MAX_RATE_LIMIT_RETRIES times (honoring a Retry-After header when the server sends
        # one, else exponential backoff) -- see module docstring's bug-fix #2 note. Callers
        # still get the response back even if it's a 429 on the final attempt -- they decide
        # whether to raise or degrade gracefully (this module always degrades; see callers).
        #
        # A connection-level failure (timeout, DNS hiccup, reset -- no HTTP response at all,
        # distinct from a real 429 response) gets the same retry/backoff treatment and, if it
        # never recovers, comes back as a synthetic 429 response so every caller's existing
        # "still rate-limited after retries -- skip this part" handling covers it too, instead
        # of crashing the run. See module docstring's bug-fix #6 note.
        backoff = _RATE_LIMIT_INITIAL_BACKOFF_SECONDS
        resp = None
        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            time.sleep(_REQUEST_PACING_SECONDS)
            try:
                resp = requests.get(url, params=params, timeout=self.timeout, headers={"Accept": "application/json"})
            except requests.exceptions.RequestException as exc:
                if attempt == _MAX_RATE_LIMIT_RETRIES:
                    print(
                        f"GWAS Catalog: connection failure on {url} after {_MAX_RATE_LIMIT_RETRIES} retries "
                        f"({exc.__class__.__name__}: {exc}) -- treating as rate-limited so callers skip "
                        f"this part gracefully instead of crashing the run.",
                        flush=True,
                    )
                    fake_resp = requests.Response()
                    fake_resp.status_code = 429
                    return fake_resp
                print(
                    f"GWAS Catalog: connection failure on {url} ({exc.__class__.__name__}) -- retrying in "
                    f"{backoff:.1f}s (attempt {attempt + 1}/{_MAX_RATE_LIMIT_RETRIES})...",
                    flush=True,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
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
        # Input: gene symbol, disease name. Output: up to _MAX_ASSOCIATIONS_RETURNED associations
        # (lowest p-value first) between SNPs mapped to that gene and the disease's GWAS traits.
        # [] means none found OR still rate-limited after retries -- check last_check_rate_limited
        # to tell them apart. A 404/410 raises (the endpoint moved/was retired): an API change must
        # fail loudly, never look like "no evidence".
        self.last_check_rate_limited = False
        found: Dict[str, GwasAssociation] = {}
        for trait in DISEASE_GWAS_TRAITS.get(disease_name, [disease_name]):
            for page_number in range(_MAX_PAGES_PER_TRAIT):
                resp = self._get(
                    f"{self.base_url}{_ASSOCIATIONS_PATH}",
                    params={"mapped_gene": gene_symbol, "efo_trait": trait, "size": _PAGE_SIZE, "page": page_number},
                )
                if resp.status_code in (404, 410):
                    raise RuntimeError(
                        f"GWAS Catalog: {_ASSOCIATIONS_PATH} returned HTTP {resp.status_code} for gene "
                        f"{gene_symbol!r} -- the API path has changed or been retired. Check "
                        f"https://www.ebi.ac.uk/gwas/rest/docs and update crossdb_client.py."
                    )
                if resp.status_code == 429:
                    print(
                        f"GWAS Catalog: still rate-limited after {_MAX_RATE_LIMIT_RETRIES} retries for gene "
                        f"{gene_symbol!r} -- treated as 'not checked', not a real negative.",
                        flush=True,
                    )
                    self.last_check_rate_limited = True
                    return self._best(found)
                resp.raise_for_status()
                payload = resp.json()
                for assoc in payload.get("_embedded", {}).get("associations", []):
                    self._collect(assoc, gene_symbol, disease_name, found)
                if page_number + 1 >= payload.get("page", {}).get("totalPages", 1):
                    break
        return self._best(found)

    @staticmethod
    def _best(found: Dict[str, GwasAssociation]) -> List[GwasAssociation]:
        ranked = sorted(found.values(), key=lambda a: (a.pvalue is None, a.pvalue if a.pvalue is not None else 0.0))
        return ranked[:_MAX_ASSOCIATIONS_RETURNED]

    @staticmethod
    def _collect(assoc: dict, gene_symbol: str, disease_name: str, found: Dict[str, GwasAssociation]) -> None:
        # Input: one v2 association record. Adds a GwasAssociation to `found` (keyed by association
        # id, so a repeat across pages/traits is counted once) if the gene is among its mapped genes
        # and one of its EFO traits matches the disease.
        if gene_symbol not in assoc.get("mapped_genes", []):
            return
        key = str(assoc.get("association_id") or id(assoc))
        if key in found:
            return
        for efo_trait in assoc.get("efo_traits", []):
            trait_name = efo_trait.get("efo_trait", "")
            if _trait_matches_disease(trait_name, disease_name):
                pubmed = assoc.get("pubmed_id")
                found[key] = GwasAssociation(
                    gene_symbol=gene_symbol,
                    trait_name=trait_name,
                    efo_id=efo_trait.get("efo_id"),
                    pvalue=assoc.get("p_value"),
                    study_accession=assoc.get("accession_id"),
                    pubmed_id=str(pubmed) if pubmed is not None else None,
                )
                return


class StubGWASCatalogClient:
    """(gene_symbol, disease_name) -> list of GwasAssociation. No network call."""

    def __init__(self, associations: Dict[tuple, List[GwasAssociation]]):
        self.associations = associations
        self.calls: List[tuple] = []
        # Always False -- the stub never makes real requests, so it can never be rate-limited.
        # Present purely so code reads the same attribute off either client without a getattr
        # default at every call site -- see GWASCatalogClient's own attribute for what it means.
        # Tests that want to exercise the "possibly incomplete" path can just set this directly.
        self.last_check_rate_limited = False

    def get_gene_disease_associations(self, gene_symbol: str, disease_name: str) -> List[GwasAssociation]:
        self.calls.append((gene_symbol, disease_name))
        return self.associations.get((gene_symbol, disease_name), [])
