"""Stage 2 OmniPath client (REST web service, no key needed). Fetches directed gene-gene interactions from a set of source
genes to wherever they point, including genes outside that set, so intermediate bridge proteins can be found.

Direction rule: one direction flag set gives that direction; neither gives "unknown"; both set (resources disagree) gives
OmniPath's consensus direction if exactly one consensus flag is set, else "unknown".
Dataset scope: DEFAULT_DATASETS is OmniPath's core curated set; EXTENDED_DATASETS adds the curated pathwayextra and
kinaseextra sets. Complex nodes (ids prefixed "COMPLEX:") are kept as ordinary nodes.
"""
import csv
import io
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

DEFAULT_BASE_URL = "https://omnipathdb.org/interactions"
DEFAULT_DATASETS = "omnipath"  # core curated set
EXTENDED_DATASETS = "omnipath,pathwayextra,kinaseextra"  # core plus broader curated signaling and kinase interactions
_QUERY_BATCH_SIZE = 300  # genes per HTTP request
_MAX_REQUEST_ATTEMPTS = 4  # 1 try + 3 retries
_RETRY_BACKOFF_SECONDS = 1.0  # doubles on each retry: 1, 2, 4
_RETRYABLE_STATUS = (429, 500, 502, 503, 504)


# One directed gene-gene interaction with its sign and evidence counts.
@dataclass
class Interaction:
    source_gene: str
    source_uniprot: str
    target_gene: str
    target_uniprot: str
    interaction_type: str  # "activates" | "inhibits" | "unknown"
    omnipath_edge_id: str
    curation_effort: Optional[int] = None  # OmniPath's own count of (resource, reference) records behind the edge; None if not returned
    n_sources: Optional[int] = None  # number of distinct source resources listing the edge; None if not returned

    # Output: the interaction as a plain dict.
    def to_dict(self) -> dict:
        return {
            "source_gene": self.source_gene,
            "source_protein": self.source_uniprot,
            "target_gene": self.target_gene,
            "target_protein": self.target_uniprot,
            "interaction_type": self.interaction_type,
            "omnipath_edge_id": self.omnipath_edge_id,
            "curation_effort": self.curation_effort,
            "n_sources": self.n_sources,
        }


# Input: a raw string field. Output: whether it represents true.
def _truthy(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


# Turn OmniPath's direction flags into a label. Input: the stimulation and inhibition flags plus the two consensus flags
# (used only to break a tie). Output: "activates", "inhibits" or "unknown".
def _interaction_type(
    is_stimulation: str, is_inhibition: str, consensus_stimulation: str = "", consensus_inhibition: str = ""
) -> str:
    # One flag set gives that direction; neither gives "unknown". Both set (resources disagree) uses OmniPath's
    # consensus columns if exactly one is set, else "unknown".
    stim, inhib = _truthy(is_stimulation), _truthy(is_inhibition)
    if stim and not inhib:
        return "activates"
    if inhib and not stim:
        return "inhibits"
    if not stim and not inhib:
        return "unknown"
    c_stim, c_inhib = _truthy(consensus_stimulation), _truthy(consensus_inhibition)
    if c_stim and not c_inhib:
        return "activates"
    if c_inhib and not c_stim:
        return "inhibits"
    return "unknown"


# Input: a raw string field. Output: its integer value, or None if empty or not a number.
def _optional_int(value) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


# Input: source and target UniProt ids. Output: a deterministic edge id (OmniPath has no single edge-id column).
def _synthesize_edge_id(source_uniprot: str, target_uniprot: str) -> str:
    return f"{source_uniprot}->{target_uniprot}"


# Queries the OmniPath interactions endpoint.
class OmniPathClient:
    # Input: optional base URL, dataset selection and request timeout. Output: none.
    def __init__(self, base_url: str = None, datasets: str = DEFAULT_DATASETS, timeout: int = 60):
        self.base_url = base_url or os.environ.get("OMNIPATH_BASE_URL", DEFAULT_BASE_URL)
        self.datasets = datasets
        self.timeout = timeout

    # Fetch directed interactions leaving a set of genes, in batches to keep request URLs short. Input: source gene
    # symbols. Output: Interaction list from any of them to whatever they point at.
    def get_outgoing_interactions(self, source_genes: List[str]) -> List[Interaction]:
        if not source_genes:
            return []

        genes = sorted(set(source_genes))
        interactions: List[Interaction] = []
        for i in range(0, len(genes), _QUERY_BATCH_SIZE):
            batch = genes[i : i + _QUERY_BATCH_SIZE]
            params = {
                "sources": ",".join(batch),
                "genesymbols": "1",
                "datasets": self.datasets,
                "fields": "sources,curation_effort",  # per-edge evidence strength
            }
            interactions.extend(self._parse_response(self._get_with_retry(params)))
        return interactions

    # GET one batch, retrying connection errors, timeouts, 429 and 5xx with doubling backoff; other HTTP errors raise at
    # once, and the last error is raised after the final attempt. Input: query params. Output: the response body.
    def _get_with_retry(self, params: dict) -> str:
        delay = _RETRY_BACKOFF_SECONDS
        for attempt in range(1, _MAX_REQUEST_ATTEMPTS + 1):
            try:
                resp = requests.get(self.base_url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except requests.exceptions.RequestException as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                retryable = isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)) \
                    or status in _RETRYABLE_STATUS
                if not retryable or attempt == _MAX_REQUEST_ATTEMPTS:
                    raise
                time.sleep(delay)
                delay *= 2

    # Parse the tab-separated response. Input: raw response body. Output: Interaction list (directed edges only).
    def _parse_response(self, text: str) -> List[Interaction]:
        text = text.strip()
        if not text:
            return []

        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        interactions = []
        for row in reader:
            if not _truthy(row.get("is_directed", "")):
                continue  # an undirected edge cannot support a directional chain step
            source_uniprot = row.get("source", "")
            target_uniprot = row.get("target", "")
            # Complex nodes (ids prefixed "COMPLEX:") are kept.
            source_gene = row.get("source_genesymbol") or row.get("source")
            target_gene = row.get("target_genesymbol") or row.get("target")
            interactions.append(
                Interaction(
                    source_gene=source_gene,
                    source_uniprot=source_uniprot,
                    target_gene=target_gene,
                    target_uniprot=target_uniprot,
                    interaction_type=_interaction_type(
                        row.get("is_stimulation", ""),
                        row.get("is_inhibition", ""),
                        row.get("consensus_stimulation", ""),
                        row.get("consensus_inhibition", ""),
                    ),
                    omnipath_edge_id=_synthesize_edge_id(source_uniprot, target_uniprot),
                    curation_effort=_optional_int(row.get("curation_effort")),
                    n_sources=len([x for x in (row.get("sources") or "").split(";") if x]) or None,
                )
            )
        return interactions


# Test double returning fixed interactions with no network call.
class StubOmniPathClient:

    # Input: the interactions to serve. Output: none.
    def __init__(self, interactions: List[Interaction]):
        self.interactions = interactions
        self.calls = []

    # Input: source gene symbols. Output: the fixed interactions whose source is one of them.
    def get_outgoing_interactions(self, source_genes: List[str]) -> List[Interaction]:
        genes = set(source_genes)
        self.calls.append(sorted(genes))
        return [i for i in self.interactions if i.source_gene in genes]
