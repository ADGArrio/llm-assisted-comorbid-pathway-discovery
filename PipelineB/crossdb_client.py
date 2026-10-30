"""Stage 3 cross-database check against the GWAS Catalog REST API v2 (no key needed). Queries associations by mapped gene and
exact EFO trait name, keeps those whose trait matches the disease, and returns the strongest (lowest p-value) ones. A lookup
that gives up after exhausting retries sets last_check_rate_limited, so it is reported as unchecked rather than as no evidence;
a 404/410 raises because it means the endpoint changed.
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

# Disease name -> exact GWAS Catalog EFO trait names to query.
DISEASE_GWAS_TRAITS = {
    "Type 2 Diabetes Mellitus": ["type 2 diabetes mellitus"],
    "Alzheimer's Disease": ["Alzheimer disease"],
    "Coronary Artery Disease": ["coronary artery disorder"],
}

# Client-side pacing between requests and 429 retry/backoff settings.
_REQUEST_PACING_SECONDS = 1.0
_MAX_RATE_LIMIT_RETRIES = 5
_RATE_LIMIT_INITIAL_BACKOFF_SECONDS = 5.0


# One SNP-trait association from the GWAS Catalog.
@dataclass
class GwasAssociation:
    gene_symbol: str
    trait_name: str
    efo_id: Optional[str]
    pvalue: Optional[float]
    study_accession: Optional[str]
    pubmed_id: Optional[str] = None

    # Output: the association as a plain dict.
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


# First-pass filter for GWAS trait text. Input: a GWAS Catalog trait string and a disease name. Output: True if the
# trait contains one of the disease's key terms as whole words.
def _trait_matches_disease(trait_name: str, disease_name: str) -> bool:
    terms = _DISEASE_KEY_TERMS.get(disease_name, [disease_name.lower()])
    trait_lower = trait_name.lower()
    return any(re.search(r"\b" + re.escape(term) + r"\b", trait_lower) for term in terms)


# Queries the GWAS Catalog associations endpoint.
class GWASCatalogClient:
    # Input: optional base URL and timeout. Output: none.
    def __init__(self, base_url: str = None, timeout: int = 30):
        self.base_url = base_url or os.environ.get("GWAS_CATALOG_BASE_URL", GWAS_DEFAULT_BASE_URL)
        self.timeout = timeout
        # True if the latest lookup gave up after exhausting retries (rate limit or connection failure): not fully
        # checked, not a clean negative.
        self.last_check_rate_limited = False

    # GET with client-side pacing and retries of 429 responses (honouring Retry-After, else exponential backoff).
    # Connection failures get the same retries and, if they never recover, come back as a synthetic 429. Input: URL and
    # params. Output: the response.
    def _get(self, url: str, params: dict = None):
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

    # Input: gene symbol and disease name. Output: up to _MAX_ASSOCIATIONS_RETURNED associations (lowest p-value first)
    # between SNPs mapped to the gene and the disease's traits. An empty list means none found or still rate-limited
    # (check last_check_rate_limited); a 404/410 raises.
    def get_gene_disease_associations(self, gene_symbol: str, disease_name: str) -> List[GwasAssociation]:
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

    # Input: associations keyed by id. Output: the strongest ones, lowest p-value first.
    @staticmethod
    def _best(found: Dict[str, GwasAssociation]) -> List[GwasAssociation]:
        ranked = sorted(found.values(), key=lambda a: (a.pvalue is None, a.pvalue if a.pvalue is not None else 0.0))
        return ranked[:_MAX_ASSOCIATIONS_RETURNED]

    # Add a record to `found` if the gene is among its mapped genes and one of its traits matches the disease. Input:
    # one v2 association record, gene symbol, disease name and the dict keyed by association id (so repeats count once).
    # Output: none.
    @staticmethod
    def _collect(assoc: dict, gene_symbol: str, disease_name: str, found: Dict[str, GwasAssociation]) -> None:
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


# Test double mapping (gene, disease) to associations. No network call.
class StubGWASCatalogClient:

    # Input: the association map. Output: none.
    def __init__(self, associations: Dict[tuple, List[GwasAssociation]]):
        self.associations = associations
        self.calls: List[tuple] = []
        # Always False: the stub never makes requests. Present so code can read the attribute off either client.
        self.last_check_rate_limited = False

    # Input: gene symbol and disease name. Output: the mapped associations, or an empty list.
    def get_gene_disease_associations(self, gene_symbol: str, disease_name: str) -> List[GwasAssociation]:
        self.calls.append((gene_symbol, disease_name))
        return self.associations.get((gene_symbol, disease_name), [])
