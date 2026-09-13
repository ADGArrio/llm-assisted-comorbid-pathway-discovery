"""
Agent 2 -- Chain Construction.

Service: OmniPathClient (omnipath_client.py) for real PPI edges, LLMClient
(common/llm_client.py) to judge between multiple candidate paths.
Data: BFS over live OmniPath edges finds every real shortest directed
path from a disease1-associated gene to a disease2-associated gene
(<= max_hops). The LLM never invents edges -- it only selects which of
the code-found real candidates are biologically plausible (zero, one, or
several), and is skipped entirely when there's only one candidate.
Used by: PipelineB.agent3_confirmation, and the top-level pipeline.

Edge score fields (schema per project_architecture.md):
source_disease_assoc_score = source gene's Open Targets score vs disease1;
target_disease_assoc_score = target gene's score vs disease2. Either can
be None for a middle hop not itself disease-associated in that direction.

BFS note: when two branches reach the same gene at the same depth, only
the first-found path through it is kept (marked visited) -- affects which
specific path is reported through a shared node, not whether the
shortest depth is found. `_MAX_FRONTIER_SIZE` caps genes queried per BFS
layer (logs a warning + truncates deterministically if exceeded) to keep
request size/runtime bounded; a truncated gene could in theory miss a
legitimate longer path through it.

DEFAULT_MAX_HOPS = 5: deep enough to usually surface more than one
candidate for the LLM to actually adjudicate between, per a real
max_hops=10 run that returned 10 candidates but 100% LLM endorsement
(see claude/agent2_findings_maxhops10_run.md). `--max-hops` overrides.

Unknown-direction handling: OmniPath edges with no curated
activation/inhibition flag (interaction_type == "unknown") are marked as
such in the LLM's candidate text and prompted to be treated with
skepticism. Independent of what the LLM says, `_with_unknown_direction_caveat`
appends a fixed caveat sentence to any returned chain containing an
unknown-direction edge (including the single-candidate bypass, which
skips the LLM). `ChainCandidate.has_unconfirmed_direction` exposes this
as a boolean so it doesn't need to be re-parsed from the rationale text.
"""
import argparse
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from common.config import DISEASE_PAIRS, get_pair
from common.llm_client import LLMClient
from PipelineB.agent1_discovery import Agent1DiscoveryResult, run_agent1_discovery
from PipelineB.omnipath_client import Interaction, OmniPathClient
from PipelineB.pubmed_client import Citation

DEFAULT_MAX_HOPS = 5
DEFAULT_MAX_CANDIDATE_PATHS = 10
_MAX_FRONTIER_SIZE = 300  # safety cap per BFS layer; see module docstring


@dataclass
class ChainEdge:
    source_gene: str
    source_protein: str
    target_gene: str
    target_protein: str
    interaction_type: str
    omnipath_edge_id: str
    source_disease_assoc_score: Optional[float]
    target_disease_assoc_score: Optional[float]

    def to_dict(self) -> dict:
        return {
            "source_gene": self.source_gene,
            "source_protein": self.source_protein,
            "target_gene": self.target_gene,
            "target_protein": self.target_protein,
            "interaction_type": self.interaction_type,
            "omnipath_edge_id": self.omnipath_edge_id,
            "source_disease_assoc_score": self.source_disease_assoc_score,
            "target_disease_assoc_score": self.target_disease_assoc_score,
        }


@dataclass
class ChainCandidate:
    """One LLM-endorsed plausible chain, with its rationale. A single
    Agent2ChainResult can hold more than one of these."""

    edges: List[ChainEdge]
    rationale: Optional[str]

    @property
    def has_unconfirmed_direction(self) -> bool:
        # True if any edge has no curated OmniPath direction flag. See module docstring.
        return any(edge.interaction_type == "unknown" for edge in self.edges)

    def to_dict(self) -> dict:
        return {
            "edges": [e.to_dict() for e in self.edges],
            "rationale": self.rationale,
            "has_unconfirmed_direction": self.has_unconfirmed_direction,
        }


@dataclass
class Agent2ChainResult:
    disease1: str
    disease2: str
    chains: List[ChainCandidate]
    candidate_path_count: int
    shared_genes: List[str]

    def to_dict(self) -> dict:
        return {
            "disease1": self.disease1,
            "disease2": self.disease2,
            "chains": [c.to_dict() for c in self.chains],
            "candidate_path_count": self.candidate_path_count,
            "shared_genes": self.shared_genes,
        }


class NoChainFoundError(RuntimeError):
    """Raised when BFS finds no directed path at all within max_hops --
    a real, expected possible outcome, not a bug. Distinct from a
    successful result with an empty `chains` list, which means BFS found
    candidates but the LLM judged none of them plausible."""


def _score_maps(agent1_result: Agent1DiscoveryResult) -> Tuple[Dict[str, float], Dict[str, float]]:
    # Input: Agent1DiscoveryResult. Output: (disease1 gene->score dict, disease2 gene->score dict).
    d1 = {t.gene_symbol: t.disease_assoc_score for t in agent1_result.disease1_targets}
    d2 = {t.gene_symbol: t.disease_assoc_score for t in agent1_result.disease2_targets}
    return d1, d2


def _bfs_shortest_paths(
    omnipath_client,
    start_genes: List[str],
    goal_genes: set,
    max_hops: int,
    max_paths: int,
) -> List[List[Interaction]]:
    # Input: client, start gene pool, goal gene set, hop limit, max paths to collect.
    # Output: list of shortest paths (each a list of Interaction edges), each >= 1 edge long.
    visited = set(start_genes)
    frontier: List[Tuple[str, List[Interaction]]] = [(gene, []) for gene in start_genes]
    found: List[List[Interaction]] = []
    depth = 0

    while frontier and depth < max_hops and not found:
        depth += 1

        if len(frontier) > _MAX_FRONTIER_SIZE:
            print(
                f"WARNING: Agent 2 BFS depth {depth}: frontier of {len(frontier)} genes exceeds "
                f"the {_MAX_FRONTIER_SIZE}-gene safety cap; truncating (sorted, deterministic).",
                flush=True,
            )
            frontier = sorted(frontier, key=lambda item: item[0])[:_MAX_FRONTIER_SIZE]

        layer_genes = [gene for gene, _ in frontier]
        paths_by_gene = dict(frontier)

        interactions = omnipath_client.get_outgoing_interactions(layer_genes)
        by_source: Dict[str, List[Interaction]] = {}
        for edge in interactions:
            by_source.setdefault(edge.source_gene, []).append(edge)

        next_frontier: List[Tuple[str, List[Interaction]]] = []
        for gene in layer_genes:
            path = paths_by_gene[gene]
            for edge in by_source.get(gene, []):
                if edge.target_gene in goal_genes:
                    found.append(path + [edge])
                    if len(found) >= max_paths:
                        return found
                elif edge.target_gene not in visited:
                    visited.add(edge.target_gene)
                    next_frontier.append((edge.target_gene, path + [edge]))
        frontier = next_frontier

    return found


def _has_unknown_direction(path: List[Interaction]) -> bool:
    # Input: a path (list of edges). Output: whether any edge lacks a curated direction flag.
    return any(edge.interaction_type == "unknown" for edge in path)


_UNKNOWN_DIRECTION_CAVEAT = (
    "Caveat: at least one interaction in this pathway has no curated activation/inhibition "
    "flag in OmniPath, so its causal direction is not established by curated evidence -- "
    "treat that step's directionality as unconfirmed, not as a known activate/inhibit "
    "relationship, pending further literature review."
)


def _with_unknown_direction_caveat(path: List[Interaction], rationale: Optional[str]) -> Optional[str]:
    # Input: path + its rationale (may be None). Output: rationale with the fixed caveat
    # sentence appended if the path has any unknown-direction edge; unchanged otherwise.
    if not _has_unknown_direction(path):
        return rationale
    if rationale:
        return f"{rationale} {_UNKNOWN_DIRECTION_CAVEAT}"
    return _UNKNOWN_DIRECTION_CAVEAT


_SELECTION_PROMPT_TEMPLATE = """You are a skeptical biomedical researcher evaluating several \
candidate molecular pathways, all independently confirmed to exist in a protein-protein \
interaction database (OmniPath). Each pathway links a protein associated with {disease1} to a \
protein associated with {disease2}, through the same number of intermediate steps.

Some genes below have supporting literature citations; use them as evidence where available. \
Where none are listed for a gene, rely on your own domain knowledge of it -- but if you are not \
confident about a gene's identity, function, or any alternative name/alias for it, say so \
explicitly rather than asserting a specific claim you are not sure of.

Some edges below are marked "UNKNOWN DIRECTION": OmniPath has no curated activation/inhibition \
flag for that interaction, meaning it is not established by curated evidence whether the source \
activates or inhibits the target. Treat any candidate containing an UNKNOWN DIRECTION edge with \
default skepticism -- only endorse it if you can either (a) supply independent evidence (a \
citation above, or well-established domain knowledge) for that edge's direction, or (b) say \
plainly in your rationale that this step's direction is not confirmed and the mechanism should \
be read as tentative.

{candidates_text}

A candidate is biologically plausible only if you can state a SPECIFIC, named mechanism for \
EVERY step in it (e.g. "X phosphorylates Y, activating Y's kinase activity" -- not merely "X and \
Y are known to interact"). If you cannot do that for every step, the candidate is not plausible \
-- reject it rather than guessing generously. It is expected and fine for zero, one, or only a \
few of the candidates below to clear this bar; do not endorse a candidate just because it is \
structurally present in the list. Do not invent any new steps or genes; judge only the \
candidates given.

Respond with exactly one line per plausible candidate, in this format:
<candidate number>: <one-sentence rationale naming the specific mechanism at each step, citing \
the evidence above or your own reasoning, and noting any UNKNOWN DIRECTION caveat>

If none of the candidates are plausible, respond with the single word: NONE"""


def _format_candidate(index: int, path: List[Interaction], citations: Dict[str, List[Citation]]) -> str:
    # Input: candidate index, path edges, gene->citations dict. Output: prompt text block for this candidate.
    def _step(e: Interaction) -> str:
        label = "UNKNOWN DIRECTION" if e.interaction_type == "unknown" else e.interaction_type
        return f"{e.source_gene} --{label}--> {e.target_gene}"

    steps = " -> ".join(_step(e) for e in path)
    genes_in_path = sorted({e.source_gene for e in path} | {e.target_gene for e in path})

    lines = [f"{index}. {steps}"]
    for gene in genes_in_path:
        cites = citations.get(gene) or []
        if cites:
            titles = "; ".join(f'"{c.title}"' for c in cites[:2])
            lines.append(f"   {gene} literature: {titles}")
    return "\n".join(lines)


def _select_paths(
    paths: List[List[Interaction]],
    disease1: str,
    disease2: str,
    citations: Dict[str, List[Citation]],
    llm_client,
) -> List[Tuple[List[Interaction], Optional[str]]]:
    # Input: candidate paths, disease names, citations, LLM client.
    # Output: list of (path, rationale) tuples the LLM endorsed -- zero, one, or many.
    # Skips the LLM (returns the single path, rationale=None) when there's nothing to adjudicate.
    # Callers must still run every returned tuple through _with_unknown_direction_caveat.
    if len(paths) == 1:
        return [(paths[0], None)]

    candidates_text = "\n".join(_format_candidate(i + 1, p, citations) for i, p in enumerate(paths))
    prompt = _SELECTION_PROMPT_TEMPLATE.format(disease1=disease1, disease2=disease2, candidates_text=candidates_text)
    response = llm_client.generate(prompt, temperature=0.2)

    if response.strip().upper().startswith("NONE"):
        return []

    selections: List[Tuple[List[Interaction], Optional[str]]] = []
    for line in response.strip().splitlines():
        match = re.match(r"\s*(\d+)\s*:\s*(.+)", line)
        if not match:
            continue  # tolerate stray preamble/formatting rather than failing the whole response
        index = int(match.group(1)) - 1
        if 0 <= index < len(paths):
            selections.append((paths[index], match.group(2).strip()))

    if not selections:
        raise RuntimeError(
            f"Agent 2: LLM path-selection response had no parseable '<number>: <rationale>' lines "
            f"and wasn't 'NONE' either: {response!r}"
        )

    return selections


def _to_chain_edges(
    path: List[Interaction],
    disease1_scores: Dict[str, float],
    disease2_scores: Dict[str, float],
) -> List[ChainEdge]:
    # Input: path edges + both diseases' gene->score maps. Output: ChainEdge list with scores attached.
    return [
        ChainEdge(
            source_gene=edge.source_gene,
            source_protein=edge.source_uniprot,
            target_gene=edge.target_gene,
            target_protein=edge.target_uniprot,
            interaction_type=edge.interaction_type,
            omnipath_edge_id=edge.omnipath_edge_id,
            source_disease_assoc_score=disease1_scores.get(edge.source_gene),
            target_disease_assoc_score=disease2_scores.get(edge.target_gene),
        )
        for edge in path
    ]


def run_agent2_chain(
    agent1_result: Agent1DiscoveryResult,
    omnipath_client: Optional[object] = None,
    llm_client: Optional[object] = None,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_candidate_paths: int = DEFAULT_MAX_CANDIDATE_PATHS,
) -> Agent2ChainResult:
    # Input: Agent 1's result, optional clients (default to real OmniPathClient/LLMClient; pass
    # Stub* clients in tests), hop/candidate limits. Output: Agent2ChainResult.
    # Raises NoChainFoundError if BFS finds nothing within max_hops.
    omnipath_client = omnipath_client or OmniPathClient()

    disease1_scores, disease2_scores = _score_maps(agent1_result)
    disease1_genes = list(disease1_scores.keys())
    disease2_genes = set(disease2_scores.keys())
    shared_genes = sorted(disease1_scores.keys() & disease2_scores.keys())

    paths = _bfs_shortest_paths(omnipath_client, disease1_genes, disease2_genes, max_hops, max_candidate_paths)
    if not paths:
        raise NoChainFoundError(
            f"Agent 2: no directed OmniPath path found between any {agent1_result.disease1}-associated "
            f"protein and any {agent1_result.disease2}-associated protein, within {max_hops} hops, "
            f"starting from {len(disease1_genes)} disease1 genes and {len(disease2_genes)} disease2 genes "
            f"Agent 1 surfaced (search could pass through other genes OmniPath connects them to)."
        )

    llm_client = llm_client or LLMClient()
    selections = _select_paths(paths, agent1_result.disease1, agent1_result.disease2, agent1_result.citations, llm_client)

    chains = [
        ChainCandidate(
            edges=_to_chain_edges(path, disease1_scores, disease2_scores),
            rationale=_with_unknown_direction_caveat(path, rationale),
        )
        for path, rationale in selections
    ]

    return Agent2ChainResult(
        disease1=agent1_result.disease1,
        disease2=agent1_result.disease2,
        chains=chains,
        candidate_path_count=len(paths),
        shared_genes=shared_genes,
    )


def _cli():
    parser = argparse.ArgumentParser(description="Run Agent 1 then Agent 2 (Chain Construction) for one disease pair.")
    parser.add_argument("--pair", choices=[p["id"] for p in DISEASE_PAIRS], help="Disease pair id from common/config.py")
    parser.add_argument("--disease1", help="Override: disease 1 name")
    parser.add_argument("--disease2", help="Override: disease 2 name")
    parser.add_argument("--min-score", type=float, default=0.1, help="Minimum Open Targets association score, passed to Agent 1")
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS, help="Max PPI hops to search")
    args = parser.parse_args()

    if args.pair:
        pair = get_pair(args.pair)
        disease1, disease2 = pair["disease1"], pair["disease2"]
    elif args.disease1 and args.disease2:
        disease1, disease2 = args.disease1, args.disease2
    else:
        parser.error("Pass --pair, or both --disease1 and --disease2.")

    print(f"Running Agent 1 (Discovery) for {disease1} / {disease2}...", flush=True)
    agent1_result = run_agent1_discovery(disease1, disease2, min_score=args.min_score)

    print("Running Agent 2 (Chain Construction)...", flush=True)
    agent2_result = run_agent2_chain(agent1_result, max_hops=args.max_hops)

    if not agent2_result.chains:
        print(
            f"Agent 2 found {agent2_result.candidate_path_count} structural candidate path(s), "
            f"but the LLM did not judge any of them biologically plausible.",
            flush=True,
        )

    print(json.dumps(agent2_result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()
