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
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

GWAS_DEFAULT_BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api"
_ASSOCIATIONS_BY_GENE_PATH = "/singleNucleotidePolymorphisms/search/findByGene"


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
        for snp in snps:
            for assoc in snp.get("_embedded", {}).get("associations", snp.get("associations", [])):
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
