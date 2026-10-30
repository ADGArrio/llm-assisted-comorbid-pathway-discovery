"""
Stage 2 -- OmniPath client.

Service: OmniPath REST web service (https://omnipathdb.org/interactions),
no key needed. Docs: pypath.omnipathdb.org/webservice.html.
Data: tab-separated text; columns include source/target (UniProt ids),
source_genesymbol/target_genesymbol, is_directed, is_stimulation,
is_inhibition, consensus_stimulation, consensus_inhibition. Wire format
confirmed live 2026-10-15 (flags are the strings "True"/"False"); the
query also asks for `fields=sources,curation_effort` (per-edge evidence
strength, added 2026-10-18).

Direction rule (changed 2026-10-18): one flag -> that direction; neither ->
"unknown"; BOTH (a tie, 1.6% of core edges) -> OmniPath's consensus columns
if exactly one is set, else "unknown". It used to be "activates", which was
wrong for 647 of the 1,418 core-network ties. See _interaction_type.
Used by: PipelineB.stage2_judging's BFS, to fetch each layer's outgoing
directed interactions FROM a set of source genes to wherever they
actually point -- including genes outside that set, so real intermediate
bridge proteins can be discovered (not restricted back to a fixed list).

Dataset scope (`datasets` param, added as an explicit comparison lever
2026-09-14): `DEFAULT_DATASETS = "omnipath"` is OmniPath's core,
highest-confidence combined resource -- the strictest curation bar.
`EXTENDED_DATASETS` adds two more of OmniPath's own curated categories
on top of it: `pathwayextra` (further curated signaling/pathway
interactions with a somewhat looser inclusion bar than the core set) and
`kinaseextra` (further kinase-substrate relationships specifically --
the exact edge type a step like AKT1->GSK3B is). Both are still curated,
sourced data, not inferred/predicted edges -- this is "how much of
OmniPath's own curated catalog do we query," not "how strict is the
evidence for one edge." Widening this changes what counts as a "real,
grounded" edge for the whole Pipeline B comparison, which is a
methods-relevant choice, not a pure bug fix -- see
the Stage 2 BFS and prompt recalibration note for the decision to expose
this as an explicit, comparable option (`run_stage2(...,
omnipath_datasets="extended")` in main.py, or `--omnipath-datasets` on
this module's sibling CLI in stage2_judging.py) rather than silently
changing the default.

Complex nodes are kept, tried-and-reverted 2026-09-14: OmniPath
annotates some interactions as involving a multi-protein COMPLEX rather
than a single gene product on one or both ends -- its source/target id
column reads e.g. "COMPLEX:P42345_Q8N122_Q8TB45_Q96B36_Q9BVC4" and
genesymbols=1 renders it as "AKT1S1_DEPTOR_MLST8_MTOR_RPTOR" (the mTORC1
complex, as one underscore-joined pseudo gene symbol). A real T2DM->AD
run's BFS routed a chain through three of these in a row (mTORC1, the
ULK1 autophagy-initiation complex, and the AMPK complex) before landing
on a real single gene again -- real, literature-accurate systems biology
(mTORC1 does inhibit the ULK1 complex, which does feed into AMPK, which
does inhibit HMGCR), but a pseudo "gene" like that has real downsides
for this pipeline specifically: it can never carry an Open Targets
disease-association score, so it can never be a `goal_genes` endpoint;
Stage 3's confirmation step (GTEx, GWAS Catalog) operates on single
genes, so a complex hop can't be independently confirmed; and it renders
as a long, confusing underscore-joined string in the readable pathway
output. A filter dropping any interaction touching a COMPLEX was tried
the same day, then reverted per explicit direction: complexes are real,
important biology, and skipping them isn't a call to make unilaterally.
`get_outgoing_interactions` includes them again. The unresolved tradeoff
(real mechanism vs. unscorable/unconfirmable/hard-to-render) is tracked
as an open issue for committee discussion in
claude/open_issues_for_committee.md, not resolved by code.
"""
import csv
import io
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

DEFAULT_BASE_URL = "https://omnipathdb.org/interactions"
DEFAULT_DATASETS = "omnipath"  # the core curated PPI set
EXTENDED_DATASETS = "omnipath,pathwayextra,kinaseextra"  # + broader curated signaling/kinase coverage; see module docstring
_QUERY_BATCH_SIZE = 300  # genes per HTTP request -- see get_outgoing_interactions
_MAX_REQUEST_ATTEMPTS = 4  # 1 try + 3 retries; added 2026-10-15 (before: any network blip aborted the whole BFS)
_RETRY_BACKOFF_SECONDS = 1.0  # doubles each retry: 1, 2, 4
_RETRYABLE_STATUS = (429, 500, 502, 503, 504)


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


def _truthy(value: str) -> bool:
    # Input: a raw string field. Output: whether it represents true.
    return str(value).strip().lower() in ("1", "true", "yes")


def _interaction_type(
    is_stimulation: str, is_inhibition: str, consensus_stimulation: str = "", consensus_inhibition: str = ""
) -> str:
    # Input: the two raw direction flags, plus OmniPath's own consensus flags (used ONLY to break a tie).
    # Output: "activates"/"inhibits"/"unknown".
    # Exactly one flag set -> that direction. Neither -> "unknown" (kept, not dropped).
    # BOTH set (resources disagree; 1,418 of the 86,189 directed core edges, 1.6%): resolved by OmniPath's
    # consensus columns when exactly one of them is set, otherwise "unknown". Until 2026-10-18 every tie was
    # labelled "activates", which was wrong for 647 of the 1,418 (OmniPath's consensus says inhibition) and
    # unsupported for the 215 whose consensus is neither/both -- see the Stage 2 hardening note.
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


def _optional_int(value) -> Optional[int]:
    # Input: a raw string field. Output: its integer value, or None if empty/not a number.
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _synthesize_edge_id(source_uniprot: str, target_uniprot: str) -> str:
    # Input: source/target UniProt ids. Output: a deterministic edge id string
    # (OmniPath's table has no single "edge id" column of its own).
    return f"{source_uniprot}->{target_uniprot}"


class OmniPathClient:
    def __init__(self, base_url: str = None, datasets: str = DEFAULT_DATASETS, timeout: int = 60):
        self.base_url = base_url or os.environ.get("OMNIPATH_BASE_URL", DEFAULT_BASE_URL)
        self.datasets = datasets
        self.timeout = timeout

    def get_outgoing_interactions(self, source_genes: List[str]) -> List[Interaction]:
        # Input: list of source gene symbols. Output: directed Interactions FROM
        # any of those genes, to whatever they point at in OmniPath (may be outside source_genes).
        #
        # Batched in groups of _QUERY_BATCH_SIZE, added 2026-09-14 alongside raising
        # stage2_judging.py's _MAX_FRONTIER_SIZE: a single GET with thousands of comma-joined gene
        # symbols in one query string risks hitting a URL-length limit (many servers/proxies cap
        # around 2-8k characters) or a slow/failed request, untested from this network-blocked
        # sandbox. Splitting into several smaller, known-safe-sized requests avoids that risk
        # regardless of how large the caller's gene list is, at the cost of more HTTP round trips.
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
                "fields": "sources,curation_effort",  # per-edge evidence strength (added 2026-10-18)
            }
            interactions.extend(self._parse_response(self._get_with_retry(params)))
        return interactions

    def _get_with_retry(self, params: dict) -> str:
        # Input: query params for one batch. Output: the response body. Retries a connection error,
        # timeout, 429 or 5xx with doubling backoff (_MAX_REQUEST_ATTEMPTS tries in all); any other
        # HTTP error (e.g. 400) is a real bug and raises immediately. After the last attempt the
        # original error is raised -- a failed layer is never silently treated as "no edges".
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

    def _parse_response(self, text: str) -> List[Interaction]:
        # Input: raw TSV response body. Output: parsed Interaction list (directed edges only).
        text = text.strip()
        if not text:
            return []

        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        interactions = []
        for row in reader:
            if not _truthy(row.get("is_directed", "")):
                continue  # an undirected edge can't support a directional chain step
            source_uniprot = row.get("source", "")
            target_uniprot = row.get("target", "")
            # Complex nodes (source/target id prefixed "COMPLEX:") are intentionally NOT
            # filtered here -- see module docstring's "Complex nodes are kept" note.
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


class StubOmniPathClient:
    """Fixed, injectable responses for offline tests -- no network call."""

    def __init__(self, interactions: List[Interaction]):
        self.interactions = interactions
        self.calls = []

    def get_outgoing_interactions(self, source_genes: List[str]) -> List[Interaction]:
        genes = set(source_genes)
        self.calls.append(sorted(genes))
        return [i for i in self.interactions if i.source_gene in genes]
