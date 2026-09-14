"""
Agent 2 -- OmniPath client.

Service: OmniPath REST web service (https://omnipathdb.org/interactions),
no key needed. Docs: pypath.omnipathdb.org/webservice.html.
Data: tab-separated text; columns include source/target (UniProt ids),
source_genesymbol/target_genesymbol, is_directed, is_stimulation,
is_inhibition. Parsed defensively (exact wire format not yet confirmed
live from this sandbox -- network allowlist blocks it).
Used by: PipelineB.agent2_chain's BFS, to fetch each layer's outgoing
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
claude/agent2_bfs_and_prompt_recalibration.md for the decision to expose
this as an explicit, comparable option (`run_agent2(...,
omnipath_datasets="extended")` in main.py, or `--omnipath-datasets` on
this module's sibling CLI in agent2_chain.py) rather than silently
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
Agent 3's confirmation step (GTEx, GWAS Catalog) operates on single
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
from dataclasses import dataclass
from typing import List

import requests

DEFAULT_BASE_URL = "https://omnipathdb.org/interactions"
DEFAULT_DATASETS = "omnipath"  # the core curated PPI set
EXTENDED_DATASETS = "omnipath,pathwayextra,kinaseextra"  # + broader curated signaling/kinase coverage; see module docstring
_QUERY_BATCH_SIZE = 300  # genes per HTTP request -- see get_outgoing_interactions


@dataclass
class Interaction:
    source_gene: str
    source_uniprot: str
    target_gene: str
    target_uniprot: str
    interaction_type: str  # "activates" | "inhibits" | "unknown"
    omnipath_edge_id: str

    def to_dict(self) -> dict:
        return {
            "source_gene": self.source_gene,
            "source_protein": self.source_uniprot,
            "target_gene": self.target_gene,
            "target_protein": self.target_uniprot,
            "interaction_type": self.interaction_type,
            "omnipath_edge_id": self.omnipath_edge_id,
        }


def _truthy(value: str) -> bool:
    # Input: a raw string field. Output: whether it represents true.
    return str(value).strip().lower() in ("1", "true", "yes")


def _interaction_type(is_stimulation: str, is_inhibition: str) -> str:
    # Input: the two raw direction flags. Output: "activates"/"inhibits"/"unknown".
    # Ties (both flags true) resolve to "activates"; neither flag set is "unknown", not dropped.
    if _truthy(is_stimulation):
        return "activates"
    if _truthy(is_inhibition):
        return "inhibits"
    return "unknown"


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
        # agent2_chain.py's _MAX_FRONTIER_SIZE: a single GET with thousands of comma-joined gene
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
            }
            resp = requests.get(self.base_url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            interactions.extend(self._parse_response(resp.text))
        return interactions

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
                    interaction_type=_interaction_type(row.get("is_stimulation", ""), row.get("is_inhibition", "")),
                    omnipath_edge_id=_synthesize_edge_id(source_uniprot, target_uniprot),
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
