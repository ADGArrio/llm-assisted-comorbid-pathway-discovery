"""Stage 1 Open Targets Platform client (GraphQL API v4, no key needed). Resolves a disease name to an EFO/MONDO id and
returns the genes associated with it (symbol, Ensembl id, UniProt id, association score).
"""
import os
from dataclasses import dataclass
from typing import List, Optional

import requests

DEFAULT_BASE_URL = "https://api.platform.opentargets.org/api/v4/graphql"

_SEARCH_QUERY = """
query searchDisease($q: String!) {
  search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 5}) {
    hits {
      id
      entity
      name
      score
    }
  }
}
"""

_ASSOCIATED_TARGETS_QUERY = """
query diseaseTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    name
    associatedTargets(page: {index: 0, size: $size}) {
      rows {
        score
        target {
          id
          approvedSymbol
          proteinIds {
            id
            source
          }
        }
      }
    }
  }
}
"""


# One gene associated with a disease, with its Open Targets association score.
@dataclass
class GeneAssociation:
    gene_symbol: str
    ensembl_id: str
    uniprot_id: Optional[str]
    disease_assoc_score: float

    # Output: the record as a plain dict.
    def to_dict(self) -> dict:
        return {
            "gene_symbol": self.gene_symbol,
            "ensembl_id": self.ensembl_id,
            "uniprot_id": self.uniprot_id,
            "disease_assoc_score": self.disease_assoc_score,
        }


# Choose the preferred UniProt id for a target. Input: its proteinIds list ({id, source} dicts). Output: the UniProt id,
# or None.
def _pick_uniprot_id(protein_ids: List[dict]) -> Optional[str]:
    if not protein_ids:
        return None
    for p in protein_ids:
        if p.get("source") == "uniprot_swissprot":
            return p.get("id")
    for p in protein_ids:
        if "uniprot" in (p.get("source") or "").lower():
            return p.get("id")
    return None


_META_QUERY = """
query { meta { apiVersion { x y z } dataVersion { year month iteration } } }
"""


# Queries the Open Targets GraphQL endpoint.
class OpenTargetsClient:
    # Input: optional endpoint URL and request timeout. Output: none.
    def __init__(self, base_url: str = None, timeout: int = 60):
        self.base_url = base_url or os.environ.get("OPEN_TARGETS_BASE_URL", DEFAULT_BASE_URL)
        self.timeout = timeout

    # Send one GraphQL request. Input: query string and variables dict. Output: the response's "data" object; raises on
    # GraphQL errors.
    def _post(self, query: str, variables: dict) -> dict:
        resp = requests.post(
            self.base_url,
            json={"query": query, "variables": variables},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("errors"):
            raise RuntimeError(f"Open Targets GraphQL error: {payload['errors']}")
        return payload["data"]

    # Ask which Open Targets release is answering. Output: {"api_version": ..., "data_version": ...}.
    def get_data_version(self) -> dict:
        # Which Open Targets release answered; scores change between releases, so a saved run records it.
        meta = self._post(_META_QUERY, {})["meta"]
        api = meta["apiVersion"]
        data = meta["dataVersion"]
        iteration = f".{data['iteration']}" if data.get("iteration") else ""
        return {
            "api_version": f"{api['x']}.{api['y']}.{api['z']}",
            "data_version": f"{data['year']}.{data['month']}{iteration}",
        }

    # Resolve a disease name to an id. Input: free-text disease name. Output: the best-matching EFO/MONDO id; raises if
    # there is no disease hit.
    def search_disease(self, disease_name: str) -> str:
        data = self._post(_SEARCH_QUERY, {"q": disease_name})
        hits = [h for h in data["search"]["hits"] if h["entity"] == "disease"]
        if not hits:
            raise ValueError(f"No disease found in Open Targets search for {disease_name!r}")
        return hits[0]["id"]  # hits are already ranked by relevance

    # Fetch a disease's associated genes. Input: EFO/MONDO id, minimum association score and maximum rows. Output:
    # GeneAssociation list for genes scoring at least the minimum.
    def get_associated_targets(
        self,
        efo_id: str,
        min_score: float = 0.1,
        size: int = 200,
    ) -> List[GeneAssociation]:
        data = self._post(_ASSOCIATED_TARGETS_QUERY, {"efoId": efo_id, "size": size})
        disease = data.get("disease")
        if disease is None:
            raise ValueError(f"Open Targets has no disease record for EFO id {efo_id!r}")

        rows = disease["associatedTargets"]["rows"]
        results = []
        for row in rows:
            score = row["score"]
            if score < min_score:
                continue
            target = row["target"]
            results.append(
                GeneAssociation(
                    gene_symbol=target["approvedSymbol"],
                    ensembl_id=target["id"],
                    uniprot_id=_pick_uniprot_id(target.get("proteinIds") or []),
                    disease_assoc_score=score,
                )
            )
        return results


# Test double returning fixed responses with no network call.
class StubOpenTargetsClient:

    # Input: a disease-name to EFO-id map and an EFO-id to gene-list map. Output: none.
    def __init__(self, disease_to_efo: dict, efo_to_targets: dict):
        self.disease_to_efo = disease_to_efo
        self.efo_to_targets = efo_to_targets
        self.search_calls = []
        self.target_calls = []

    # Output: a fixed version dict.
    def get_data_version(self) -> dict:
        return {"api_version": "stub", "data_version": "stub"}

    # Input: disease name. Output: its mapped EFO id; raises if unmapped.
    def search_disease(self, disease_name: str) -> str:
        self.search_calls.append(disease_name)
        if disease_name not in self.disease_to_efo:
            raise ValueError(f"No disease found in Open Targets search for {disease_name!r}")
        return self.disease_to_efo[disease_name]

    # Input: EFO id, minimum score and size. Output: the mapped genes scoring at least the minimum.
    def get_associated_targets(self, efo_id: str, min_score: float = 0.1, size: int = 200) -> List[GeneAssociation]:
        self.target_calls.append((efo_id, min_score, size))
        rows = self.efo_to_targets.get(efo_id, [])
        return [r for r in rows if r.disease_assoc_score >= min_score]
