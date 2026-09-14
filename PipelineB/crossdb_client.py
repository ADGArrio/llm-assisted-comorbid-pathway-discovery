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
well documented and wasn't implicated by this bug. Still not verified
against the live endpoint from this sandbox -- if a real run still comes
back with unexpectedly few confirmations, print one raw
`requests.get(...).json()` response for a well-studied gene/SNP and
check the actual field names before assuming the genes lack evidence.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

GWAS_DEFAULT_BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api"
_ASSOCIATIONS_BY_GENE_PATH = "/singleNucleotidePolymorphisms/search/findByGene"
# Cap on how many of a gene's SNPs get a follow-up associations request -- a highly studied gene
# can turn up hundreds of SNPs, and each now costs its own HTTP round trip (see bug-fix note
# above), so this bounds worst-case request count per gene/disease check. findByGene's own
# default page size is 20, so this doesn't cut into what a single search page already returns.
_MAX_SNPS_PER_GENE_FOR_ASSOCIATIONS = 20


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

    def get_gene_disease_associations(self, gene_symbol: str, disease_name: str) -> List[GwasAssociation]:
        # Input: gene symbol, disease name. Output: associations whose trait text matches the
        # disease (see _trait_matches_disease). [] means no hits or no match -- not an error.
        resp = requests.get(
            f"{self.base_url}{_ASSOCIATIONS_BY_GENE_PATH}",
            params={"geneName": gene_symbol},
            timeout=self.timeout,
            headers={"Accept": "application/json"},
        )
        if resp.status_code == 404:
            raise RuntimeError(
                f"GWAS Catalog: {_ASSOCIATIONS_BY_GENE_PATH} 404'd for gene {gene_symbol!r} -- the "
                f"v1/v2 endpoint shape may have changed. Check https://www.ebi.ac.uk/gwas/rest/docs/api "
                f"and update crossdb_client.py's _ASSOCIATIONS_BY_GENE_PATH."
            )
        resp.raise_for_status()
        payload = resp.json()

        results: List[GwasAssociation] = []
        snps = payload.get("_embedded", {}).get("singleNucleotidePolymorphisms", [])
        for snp in snps[:_MAX_SNPS_PER_GENE_FOR_ASSOCIATIONS]:
            results.extend(self._associations_for_snp(snp, gene_symbol, disease_name))

        # Defensive fallback for the other plausible HAL shape (associations embedded directly
        # at the top level rather than nested per-SNP) -- see bug-fix note in the module
        # docstring on why the exact live shape couldn't be confirmed from this sandbox.
        for assoc in payload.get("_embedded", {}).get("associations", []):
            results.extend(self._parse_association(assoc, gene_symbol, disease_name))

        return results

    def _associations_for_snp(self, snp: dict, gene_symbol: str, disease_name: str) -> List[GwasAssociation]:
        # Input: one SNP entry from the findByGene response. Output: its matching associations.
        # See module docstring's bug-fix note: a SNP entry's associations are usually NOT
        # embedded inline, so this follows the SNP's own HAL "associations" link when needed.
        embedded = snp.get("_embedded", {})
        raw_associations = embedded.get("associations")
        if raw_associations is None and "associations" in snp:
            raw_associations = snp["associations"]  # older/alternate flat shape, if it ever appears

        if raw_associations is None:
            href = snp.get("_links", {}).get("associations", {}).get("href")
            if not href:
                return []
            assoc_resp = requests.get(href, timeout=self.timeout, headers={"Accept": "application/json"})
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
