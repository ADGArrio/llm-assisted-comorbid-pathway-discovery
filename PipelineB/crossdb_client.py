"""
Agent 3, independent cross-database check -- GWAS Catalog + DisGeNET.

Service: GWAS Catalog REST API (https://www.ebi.ac.uk/gwas/rest/api,
finder path /singleNucleotidePolymorphisms/search/findByGene -- the v1
HAL pattern; the exact v1/v2 path couldn't be confirmed live from this
sandbox, so this may 404 on a real run -- check
https://www.ebi.ac.uk/gwas/rest/docs/api if so). DisGeNET REST API
(https://api.disgenet.com/api/v1/gda/summary, auth header
"Authorization: <raw_key>", param disease=UMLS_<CUI>) -- confirmed via
real client source in the ToolUniverse project. NOTE: this is NOT the
mcp.disgenet.com URL initially suggested; that host doesn't match
DisGeNET's documented real API (api.disgenet.com) and isn't in any
connector registry. Requires DISGENET_API_KEY in .env.
Data: GWAS Catalog returns HAL JSON (_embedded.singleNucleotidePolymorphisms
-> associations -> efoTraits); DisGeNET returns {"payload": [...]} with
symbolOfGene/diseaseName/score/numPMIDs fields.
Used by: PipelineB.agent3_confirmation, to independently confirm a
chain's two disease-anchored endpoint genes against sources Open Targets
doesn't itself aggregate.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

GWAS_DEFAULT_BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api"
_ASSOCIATIONS_BY_GENE_PATH = "/singleNucleotidePolymorphisms/search/findByGene"

DISGENET_DEFAULT_BASE_URL = "https://api.disgenet.com/api/v1"

# DisGeNET requires UMLS-CUI-prefixed disease ids -- widely cited CUIs, worth
# double-checking against DisGeNET's own disease search on a real run.
_DISEASE_UMLS_CUI = {
    "Type 2 Diabetes Mellitus": "C0011860",
    "Alzheimer's Disease": "C0002395",
    "Coronary Artery Disease": "C0010054",
}


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


@dataclass
class DisgenetAssociation:
    gene_symbol: str
    disease_name: str
    score: float
    num_pmids: int

    def to_dict(self) -> dict:
        return {
            "gene_symbol": self.gene_symbol,
            "disease_name": self.disease_name,
            "score": self.score,
            "num_pmids": self.num_pmids,
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


class DisGeNETClient:
    def __init__(self, api_key: str = None, base_url: str = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("DISGENET_API_KEY")
        self.base_url = base_url or os.environ.get("DISGENET_BASE_URL", DISGENET_DEFAULT_BASE_URL)
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "DisGeNETClient requires an API key. Set DISGENET_API_KEY in .env -- get one via your "
                "academic license at https://www.disgenet.com/."
            )

    def get_gene_disease_association(self, gene_symbol: str, disease_name: str) -> Optional[DisgenetAssociation]:
        # Input: gene symbol, disease name (must be in _DISEASE_UMLS_CUI). Output: the top
        # association record, or None if DisGeNET has none on record -- a legitimate outcome.
        cui = _DISEASE_UMLS_CUI.get(disease_name)
        if cui is None:
            raise ValueError(f"No UMLS CUI mapped for {disease_name!r}; add one to _DISEASE_UMLS_CUI.")

        resp = requests.get(
            f"{self.base_url}/gda/summary",
            params={"gene_symbol": gene_symbol, "disease": f"UMLS_{cui}", "page_number": 0},
            headers={"Authorization": self.api_key},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get("payload") or []
        if not records:
            return None
        record = records[0]
        return DisgenetAssociation(
            gene_symbol=record.get("symbolOfGene", gene_symbol),
            disease_name=record.get("diseaseName", disease_name),
            score=record.get("score", 0.0),
            num_pmids=record.get("numPMIDs", 0),
        )


class StubDisGeNETClient:
    """(gene_symbol, disease_name) -> DisgenetAssociation or None. No network call."""

    def __init__(self, associations: Dict[tuple, Optional[DisgenetAssociation]]):
        self.associations = associations
        self.calls: List[tuple] = []

    def get_gene_disease_association(self, gene_symbol: str, disease_name: str) -> Optional[DisgenetAssociation]:
        self.calls.append((gene_symbol, disease_name))
        return self.associations.get((gene_symbol, disease_name))
