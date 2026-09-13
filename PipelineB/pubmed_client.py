"""
Agent 1, sub-agent 2 -- PubMed E-utilities client.

Service: NCBI E-utilities (https://eutils.ncbi.nlm.nih.gov/entrez/eutils),
esearch (gene+disease terms -> PMIDs) then esummary (PMIDs -> titles).
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/. No key required;
set PUBMED_API_KEY/PUBMED_EMAIL in .env to raise the <=3 req/sec limit.
Data: JSON. esearchresult.idlist -> list of PMIDs; result.<pmid>.title.
Used by: PipelineB.agent1_discovery, for citation support on genes
Open Targets already surfaced (not a second discovery source).

Not tested against the live endpoint from this sandbox (network
allowlist blocks it) -- verify on your own machine.
"""
import os
import time
from dataclasses import dataclass
from typing import List

import requests

DEFAULT_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_NO_KEY_MIN_INTERVAL_SECONDS = 0.4  # throttle to stay under NCBI's no-key 3 req/sec limit


@dataclass
class Citation:
    pmid: str
    title: str

    def to_dict(self) -> dict:
        return {"pmid": self.pmid, "title": self.title}


class PubMedClient:
    def __init__(self, base_url: str = None, api_key: str = None, email: str = None, timeout: int = 30):
        self.base_url = (base_url or os.environ.get("PUBMED_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.environ.get("PUBMED_API_KEY")
        self.email = email or os.environ.get("PUBMED_EMAIL")
        self.timeout = timeout
        self._last_request_time = 0.0

    def _throttle(self):
        # Input/output: none -- sleeps if needed to respect the no-key rate limit.
        if self.api_key:
            return
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _NO_KEY_MIN_INTERVAL_SECONDS:
            time.sleep(_NO_KEY_MIN_INTERVAL_SECONDS - elapsed)

    def _get(self, path: str, params: dict) -> dict:
        # Input: E-utilities path + params. Output: parsed JSON response.
        params = {**params, "retmode": "json", "tool": "comorbidity-pathway-pipeline"}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email

        self._throttle()
        resp = requests.get(f"{self.base_url}/{path}", params=params, timeout=self.timeout)
        self._last_request_time = time.monotonic()
        resp.raise_for_status()
        return resp.json()

    def search_citations(self, gene_symbol: str, disease_name: str, retmax: int = 3) -> List[Citation]:
        # Input: gene symbol, disease name, max results. Output: up to `retmax` Citations.
        # Returns [] if nothing found -- a normal, expected outcome, not an error.
        term = f"{gene_symbol}[Title/Abstract] AND {disease_name}[Title/Abstract]"
        search_data = self._get("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": retmax})
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        summary_data = self._get("esummary.fcgi", {"db": "pubmed", "id": ",".join(pmids)})
        result = summary_data.get("result", {})
        citations = []
        for pmid in pmids:
            doc = result.get(pmid)
            if doc:
                citations.append(Citation(pmid=pmid, title=doc.get("title", "")))
        return citations


class StubPubMedClient:
    """Fixed, injectable responses for offline tests -- no network call."""

    def __init__(self, citations_by_gene: dict):
        self.citations_by_gene = citations_by_gene
        self.calls = []

    def search_citations(self, gene_symbol: str, disease_name: str, retmax: int = 3) -> List[Citation]:
        self.calls.append((gene_symbol, disease_name, retmax))
        return self.citations_by_gene.get(gene_symbol, [])[:retmax]
