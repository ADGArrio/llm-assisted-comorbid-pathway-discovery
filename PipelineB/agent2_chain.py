"""
Agent 2 -- Chain Construction.

Service: OmniPathClient (omnipath_client.py) for real PPI edges,
MetalinksClient (metalinks_client.py, optional) for real gene<->metabolite
edges, LLMClient (common/llm_client.py) to judge between multiple
candidate paths.
Data: BFS over live OmniPath (gene->gene) and MetalinksDB (gene<->metabolite)
edges finds every real shortest directed path from a disease1-associated
gene to a disease2-associated gene (<= max_hops). A path's first and last
node are always genes (BFS starts from genes and only stops at a goal
gene); a metabolite can only appear as a middle hop, e.g.
GeneA --produces--> Cholesterol --activates--> GeneB. Passing
metalinks_client=None (the default for library/test callers) restricts
the search to protein-only chains, matching the pre-metabolite behavior
exactly; `_cli()` passes a real MetalinksClient() so real runs include
metabolites. The LLM never invents edges -- it only selects which of the
code-found real candidates are biologically plausible (zero, one, or
several), and is skipped entirely when there's only one candidate.
Used by: PipelineB.agent3_confirmation, and the top-level pipeline.

Edge score fields (schema per project_architecture.md):
source_disease_assoc_score = source gene's Open Targets score vs disease1;
target_disease_assoc_score = target gene's score vs disease2. Both are
always None for a metabolite-kind end (Open Targets scores genes, not
metabolites), and can also be None for a gene-kind middle hop not itself
disease-associated in that direction.

BFS note: when two branches reach the same node (gene or metabolite) at
the same depth, only the first-found path through it is kept (marked
visited) -- affects which specific path is reported through a shared
node, not whether the shortest depth is found. `_MAX_FRONTIER_SIZE` caps
nodes queried per BFS layer (logs a warning + truncates deterministically
if exceeded) to keep request size/runtime bounded; a truncated node
could in theory miss a legitimate longer path through it.

DEFAULT_MAX_HOPS = 5: deep enough to usually surface more than one
candidate for the LLM to actually adjudicate between, per a real
max_hops=10 run that returned 10 candidates but 100% LLM endorsement
(see claude/agent2_findings_maxhops10_run.md). `--max-hops` overrides.

Unknown-direction handling: an edge with no curated activation/inhibition
mode (interaction_type == "unknown") -- from OmniPath OR MetalinksDB's
mor==0 -- is marked as such in the LLM's candidate text and prompted to
be treated with skepticism. Independent of what the LLM says,
`_with_unknown_direction_caveat` appends a fixed caveat sentence to any
returned chain containing an unknown-direction edge (including the
single-candidate bypass, which skips the LLM). `ChainCandidate.has_unconfirmed_direction`
exposes this as a boolean so it doesn't need to be re-parsed from the
rationale text.

Rejection transparency (added 2026-09-14): when the LLM rejects every
candidate (responds NONE), it's now asked to give one sentence saying why,
so an empty `chains` list isn't a dead end -- `Agent2ChainResult.rejection_reason`
carries that sentence (None if the LLM gave a bare "NONE" with no reason,
which is tolerated rather than treated as a formatting error).

Plausibility bar loosened (2026-09-14, same day): the original bar required
a specific, named mechanism for every single hop, which was rejecting all
10 candidates outright on real T2DM->AD runs (see
claude/agent2_findings_maxhops10_run.md and the repeated
candidate_path_count=10/chains=[] results that prompted this change). The
bar is now whether the chain as a whole tells a coherent biological story --
an individual step can be justified by general biological reasoning (shared
pathway membership, an established broad functional relationship, a
plausible indirect role) rather than a fully specified named mechanism, as
long as the LLM says explicitly when a step is on that weaker footing. A
chain is still rejected if it requires a genuine, unsupported logical leap,
and the LLM still never invents edges/genes/metabolites -- it only judges
the real candidates BFS found. `claude/possible_metrics.md` has the
(separate, not-yet-built) plan to eventually compare this lenient default
against a stricter variant as a metric; this change is not that experiment,
it's a direct change to the one production prompt.
"""
import argparse
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from common.config import DISEASE_PAIRS, get_pair
from common.llm_client import LLMClient
from PipelineB.agent1_discovery import Agent1DiscoveryResult, run_agent1_discovery
from PipelineB.metalinks_client import MetaboliteEdge, MetalinksClient
from PipelineB.omnipath_client import Interaction, OmniPathClient
from PipelineB.pubmed_client import Citation

DEFAULT_MAX_HOPS = 5
DEFAULT_MAX_CANDIDATE_PATHS = 10
_MAX_FRONTIER_SIZE = 300  # safety cap per BFS layer; see module docstring


@dataclass
class ChainEdge:
    source_gene: str  # gene symbol, or a metabolite name when source_kind == "metabolite"
    source_protein: Optional[str]
    target_gene: str  # gene symbol, or a metabolite name when target_kind == "metabolite"
    target_protein: Optional[str]
    interaction_type: str  # activates | inhibits | unknown (OmniPath); also produces | degrades (MetalinksDB "pd")
    omnipath_edge_id: str
    source_disease_assoc_score: Optional[float]
    target_disease_assoc_score: Optional[float]
    source_kind: str = "gene"  # "gene" | "metabolite"
    target_kind: str = "gene"
    source_hmdb_id: Optional[str] = None
    target_hmdb_id: Optional[str] = None

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
            "source_kind": self.source_kind,
            "target_kind": self.target_kind,
            "source_hmdb_id": self.source_hmdb_id,
            "target_hmdb_id": self.target_hmdb_id,
        }


@dataclass
class ChainCandidate:
    """One LLM-endorsed plausible chain, with its rationale. A single
    Agent2ChainResult can hold more than one of these."""

    edges: List[ChainEdge]
    rationale: Optional[str]

    @property
    def has_unconfirmed_direction(self) -> bool:
        # True if any edge has no curated direction flag. See module docstring.
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
    rejection_reason: Optional[str] = None  # set when chains is empty and the LLM gave a reason for rejecting everything

    def to_dict(self) -> dict:
        return {
            "disease1": self.disease1,
            "disease2": self.disease2,
            "chains": [c.to_dict() for c in self.chains],
            "candidate_path_count": self.candidate_path_count,
            "shared_genes": self.shared_genes,
            "rejection_reason": self.rejection_reason,
        }


class NoChainFoundError(RuntimeError):
    """Raised when BFS finds no directed path at all within max_hops --
    a real, expected possible outcome, not a bug. Distinct from a
    successful result with an empty `chains` list, which means BFS found
    candidates but the LLM judged none of them plausible."""


@dataclass
class _RawEdge:
    """Internal, source-agnostic edge shape BFS operates on -- OmniPath
    Interactions and MetalinksDB MetaboliteEdges are both converted into
    this before BFS sees them, so BFS itself doesn't need to know which
    client produced an edge."""

    source_kind: str
    source_id: str
    source_protein: Optional[str]
    source_hmdb_id: Optional[str]
    target_kind: str
    target_id: str
    target_protein: Optional[str]
    target_hmdb_id: Optional[str]
    interaction_type: str
    edge_id: str


def _omnipath_to_raw(interactions: List[Interaction]) -> List[_RawEdge]:
    # Input: OmniPath Interactions (always gene->gene). Output: _RawEdge list.
    return [
        _RawEdge(
            source_kind="gene",
            source_id=i.source_gene,
            source_protein=i.source_uniprot,
            source_hmdb_id=None,
            target_kind="gene",
            target_id=i.target_gene,
            target_protein=i.target_uniprot,
            target_hmdb_id=None,
            interaction_type=i.interaction_type,
            edge_id=i.omnipath_edge_id,
        )
        for i in interactions
    ]


def _metalinks_to_raw(edges: List[MetaboliteEdge]) -> List[_RawEdge]:
    # Input: MetalinksDB MetaboliteEdges (always gene<->metabolite, one end each). Output: _RawEdge list.
    return [
        _RawEdge(
            source_kind=e.source_kind,
            source_id=e.source_id,
            source_protein=e.uniprot_id if e.source_kind == "gene" else None,
            source_hmdb_id=e.hmdb_id if e.source_kind == "metabolite" else None,
            target_kind=e.target_kind,
            target_id=e.target_id,
            target_protein=e.uniprot_id if e.target_kind == "gene" else None,
            target_hmdb_id=e.hmdb_id if e.target_kind == "metabolite" else None,
            interaction_type=e.interaction_type,
            edge_id=e.edge_id,
        )
        for e in edges
    ]


def _score_maps(agent1_result: Agent1DiscoveryResult) -> Tuple[Dict[str, float], Dict[str, float]]:
    # Input: Agent1DiscoveryResult. Output: (disease1 gene->score dict, disease2 gene->score dict).
    d1 = {t.gene_symbol: t.disease_assoc_score for t in agent1_result.disease1_targets}
    d2 = {t.gene_symbol: t.disease_assoc_score for t in agent1_result.disease2_targets}
    return d1, d2


def _bfs_shortest_paths(
    omnipath_client,
    metalinks_client,
    start_genes: List[str],
    goal_genes: set,
    max_hops: int,
    max_paths: int,
) -> List[List[_RawEdge]]:
    # Input: OmniPath client, MetalinksDB client (None = protein-only search, no metabolite hops),
    # start gene pool, goal gene set, hop limit, max paths to collect.
    # Output: shortest paths (each a list of _RawEdge, >= 1 edge long). A path's first and last
    # node are always genes; a metabolite (when metalinks_client is given) can only be a middle hop.
    visited = {("gene", g) for g in start_genes}
    frontier: List[Tuple[Tuple[str, str], List[_RawEdge]]] = [(("gene", g), []) for g in start_genes]
    found: List[List[_RawEdge]] = []
    depth = 0

    while frontier and depth < max_hops and not found:
        depth += 1

        if len(frontier) > _MAX_FRONTIER_SIZE:
            print(
                f"WARNING: Agent 2 BFS depth {depth}: frontier of {len(frontier)} nodes exceeds "
                f"the {_MAX_FRONTIER_SIZE}-node safety cap; truncating (sorted, deterministic).",
                flush=True,
            )
            frontier = sorted(frontier, key=lambda item: item[0])[:_MAX_FRONTIER_SIZE]

        gene_ids = [node_id for (kind, node_id), _ in frontier if kind == "gene"]
        metabolite_ids = [node_id for (kind, node_id), _ in frontier if kind == "metabolite"]
        paths_by_node = dict(frontier)

        raw_edges: List[_RawEdge] = []
        if gene_ids:
            raw_edges.extend(_omnipath_to_raw(omnipath_client.get_outgoing_interactions(gene_ids)))
            if metalinks_client is not None:
                raw_edges.extend(_metalinks_to_raw(metalinks_client.get_gene_to_metabolite(gene_ids)))
        if metabolite_ids and metalinks_client is not None:
            raw_edges.extend(_metalinks_to_raw(metalinks_client.get_metabolite_to_gene(metabolite_ids)))

        by_source: Dict[Tuple[str, str], List[_RawEdge]] = {}
        for edge in raw_edges:
            by_source.setdefault((edge.source_kind, edge.source_id), []).append(edge)

        next_frontier: List[Tuple[Tuple[str, str], List[_RawEdge]]] = []
        for node in paths_by_node:
            path = paths_by_node[node]
            for edge in by_source.get(node, []):
                target_node = (edge.target_kind, edge.target_id)
                if edge.target_kind == "gene" and edge.target_id in goal_genes:
                    found.append(path + [edge])
                    if len(found) >= max_paths:
                        return found
                elif target_node not in visited:
                    visited.add(target_node)
                    next_frontier.append((target_node, path + [edge]))
        frontier = next_frontier

    return found


def _has_unknown_direction(path: List[_RawEdge]) -> bool:
    # Input: a path (list of edges). Output: whether any edge lacks a curated direction flag.
    return any(edge.interaction_type == "unknown" for edge in path)


_UNKNOWN_DIRECTION_CAVEAT = (
    "Caveat: at least one interaction in this pathway has no curated activation/inhibition "
    "flag in OmniPath/MetalinksDB, so its causal direction is not established by curated "
    "evidence -- treat that step's directionality as unconfirmed, not as a known "
    "activate/inhibit relationship, pending further literature review."
)


def _with_unknown_direction_caveat(path: List[_RawEdge], rationale: Optional[str]) -> Optional[str]:
    # Input: path + its rationale (may be None). Output: rationale with the fixed caveat
    # sentence appended if the path has any unknown-direction edge; unchanged otherwise.
    if not _has_unknown_direction(path):
        return rationale
    if rationale:
        return f"{rationale} {_UNKNOWN_DIRECTION_CAVEAT}"
    return _UNKNOWN_DIRECTION_CAVEAT


_SELECTION_PROMPT_TEMPLATE = """You are a biomedical researcher evaluating several candidate \
molecular pathways, all independently confirmed to exist in real interaction databases \
(OmniPath for protein-protein steps, MetalinksDB for gene<->metabolite steps -- a step written \
as "[metabolite] <name>" is a small-molecule metabolite, not a protein). Each pathway links a \
protein associated with {disease1} to a protein associated with {disease2}, through the same \
number of intermediate steps.

Some genes below have supporting literature citations; use them as evidence where available. \
Where none are listed for a gene, rely on your own domain knowledge of it -- but if you are not \
confident about a gene's identity, function, or any alternative name/alias for it, say so \
explicitly rather than asserting a specific claim you are not sure of.

Some edges below are marked "UNKNOWN DIRECTION": there is no curated activation/inhibition mode \
for that interaction, meaning it is not established by curated evidence whether the source \
activates or inhibits the target. Treat any candidate containing an UNKNOWN DIRECTION edge with \
default skepticism -- only endorse it if you can either (a) supply independent evidence (a \
citation above, or well-established domain knowledge) for that edge's direction, or (b) say \
plainly in your rationale that this step's direction is not confirmed and the mechanism should \
be read as tentative.

{candidates_text}

A candidate is biologically plausible if, taken as a whole, it describes a coherent biological \
story connecting {disease1} to {disease2} -- you do NOT need a fully specified, named mechanism \
for every single step to endorse it. General biological reasoning is enough for an individual \
step: shared pathway membership, a well-established broad functional relationship, or a \
plausible indirect role for a gene/metabolite in the relevant biology all count, as long as you \
say so plainly when a step is on that weaker footing (e.g. "the direct mechanism between X and Y \
isn't fully specified here, but both are established players in cholesterol trafficking"). \
Reject a candidate only if it requires a genuine, unsupported logical leap -- a step with no \
known or reasonably inferable relationship at all between its endpoints, even at a general \
pathway level. It's still fine, and expected, for some candidates to be rejected on that basis; \
do not endorse a candidate just because it is structurally present in the list, and do not \
invent any new steps, genes, or metabolites -- judge only the candidates given.

Respond with exactly one line per plausible candidate, in this format:
<candidate number>: <one-sentence rationale for the chain overall, noting explicitly which \
steps (if any) rest on general reasoning rather than a fully specified mechanism, and any \
UNKNOWN DIRECTION caveat>

If none of the candidates are plausible, respond with a single line starting with the word NONE, \
followed by a colon and one sentence explaining what was missing or unconvincing across the \
candidates as a whole -- e.g. "NONE: none of the candidates named a specific enough mechanism for \
every step." This reason is for transparency, not optional decoration -- always include it."""


def _node_label(kind: str, node_id: str) -> str:
    # Input: a node's kind and id. Output: the string used to render it in the LLM prompt.
    return node_id if kind == "gene" else f"[metabolite] {node_id}"


def _format_candidate(index: int, path: List[_RawEdge], citations: Dict[str, List[Citation]]) -> str:
    # Input: candidate index, path edges, gene->citations dict. Output: prompt text block for this candidate.
    def _step(e: _RawEdge) -> str:
        label = "UNKNOWN DIRECTION" if e.interaction_type == "unknown" else e.interaction_type
        return f"{_node_label(e.source_kind, e.source_id)} --{label}--> {_node_label(e.target_kind, e.target_id)}"

    steps = " -> ".join(_step(e) for e in path)
    # Citations are keyed by gene symbol only -- metabolites never have PubMed lookups (Agent 1 only
    # cites genes), so only gene-kind node ids are worth looking up here.
    genes_in_path = sorted(
        {e.source_id for e in path if e.source_kind == "gene"} | {e.target_id for e in path if e.target_kind == "gene"}
    )

    lines = [f"{index}. {steps}"]
    for gene in genes_in_path:
        cites = citations.get(gene) or []
        if cites:
            titles = "; ".join(f'"{c.title}"' for c in cites[:2])
            lines.append(f"   {gene} literature: {titles}")
    return "\n".join(lines)


def _select_paths(
    paths: List[List[_RawEdge]],
    disease1: str,
    disease2: str,
    citations: Dict[str, List[Citation]],
    llm_client,
) -> Tuple[List[Tuple[List[_RawEdge], Optional[str]]], Optional[str]]:
    # Input: candidate paths, disease names, citations, LLM client.
    # Output: (selections, rejection_reason). selections is a list of (path, rationale) tuples the
    # LLM endorsed -- zero, one, or many. rejection_reason is set only when selections is empty AND
    # the LLM gave a reason for its blanket "NONE" (None if it didn't, or if selections is non-empty).
    # Skips the LLM (returns the single path, rationale=None, reason=None) when there's nothing to
    # adjudicate. Callers must still run every returned tuple through _with_unknown_direction_caveat.
    if len(paths) == 1:
        return [(paths[0], None)], None

    candidates_text = "\n".join(_format_candidate(i + 1, p, citations) for i, p in enumerate(paths))
    prompt = _SELECTION_PROMPT_TEMPLATE.format(disease1=disease1, disease2=disease2, candidates_text=candidates_text)
    response = llm_client.generate(prompt, temperature=0.2)

    stripped = response.strip()
    if stripped.upper().startswith("NONE"):
        reason = stripped[len("NONE"):].strip()
        if reason.startswith(":"):
            reason = reason[1:].strip()
        return [], (reason or None)

    selections: List[Tuple[List[_RawEdge], Optional[str]]] = []
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

    return selections, None


def _to_chain_edges(
    path: List[_RawEdge],
    disease1_scores: Dict[str, float],
    disease2_scores: Dict[str, float],
) -> List[ChainEdge]:
    # Input: path edges + both diseases' gene->score maps. Output: ChainEdge list with scores
    # attached (gene-kind ends only -- metabolite ends never get a disease_assoc_score).
    return [
        ChainEdge(
            source_gene=edge.source_id,
            source_protein=edge.source_protein,
            target_gene=edge.target_id,
            target_protein=edge.target_protein,
            interaction_type=edge.interaction_type,
            omnipath_edge_id=edge.edge_id,
            source_disease_assoc_score=disease1_scores.get(edge.source_id) if edge.source_kind == "gene" else None,
            target_disease_assoc_score=disease2_scores.get(edge.target_id) if edge.target_kind == "gene" else None,
            source_kind=edge.source_kind,
            target_kind=edge.target_kind,
            source_hmdb_id=edge.source_hmdb_id,
            target_hmdb_id=edge.target_hmdb_id,
        )
        for edge in path
    ]


def run_agent2_chain(
    agent1_result: Agent1DiscoveryResult,
    omnipath_client: Optional[object] = None,
    metalinks_client: Optional[object] = None,
    llm_client: Optional[object] = None,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_candidate_paths: int = DEFAULT_MAX_CANDIDATE_PATHS,
) -> Agent2ChainResult:
    # Input: Agent 1's result, optional clients (omnipath_client defaults to real OmniPathClient;
    # metalinks_client defaults to None, meaning protein-only search -- pass a real MetalinksClient()
    # or StubMetalinksClient to include gene<->metabolite hops; llm_client defaults to real LLMClient;
    # pass Stub* clients in tests), hop/candidate limits. Output: Agent2ChainResult.
    # Raises NoChainFoundError if BFS finds nothing within max_hops.
    omnipath_client = omnipath_client or OmniPathClient()

    disease1_scores, disease2_scores = _score_maps(agent1_result)
    disease1_genes = list(disease1_scores.keys())
    disease2_genes = set(disease2_scores.keys())
    shared_genes = sorted(disease1_scores.keys() & disease2_scores.keys())

    paths = _bfs_shortest_paths(
        omnipath_client, metalinks_client, disease1_genes, disease2_genes, max_hops, max_candidate_paths
    )
    if not paths:
        raise NoChainFoundError(
            f"Agent 2: no directed path found between any {agent1_result.disease1}-associated "
            f"protein and any {agent1_result.disease2}-associated protein, within {max_hops} hops, "
            f"starting from {len(disease1_genes)} disease1 genes and {len(disease2_genes)} disease2 genes "
            f"Agent 1 surfaced (search could pass through other genes OmniPath connects them to, and "
            f"metabolites if a MetalinksDB client was provided)."
        )

    llm_client = llm_client or LLMClient()
    selections, rejection_reason = _select_paths(
        paths, agent1_result.disease1, agent1_result.disease2, agent1_result.citations, llm_client
    )

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
        rejection_reason=rejection_reason if not chains else None,
    )


def _cli():
    parser = argparse.ArgumentParser(description="Run Agent 1 then Agent 2 (Chain Construction) for one disease pair.")
    parser.add_argument("--pair", choices=[p["id"] for p in DISEASE_PAIRS], help="Disease pair id from common/config.py")
    parser.add_argument("--disease1", help="Override: disease 1 name")
    parser.add_argument("--disease2", help="Override: disease 2 name")
    parser.add_argument("--min-score", type=float, default=0.1, help="Minimum Open Targets association score, passed to Agent 1")
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS, help="Max PPI/metabolite hops to search")
    parser.add_argument(
        "--no-metabolites", action="store_true", help="Protein-only search -- skip MetalinksDB gene<->metabolite hops"
    )
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
    metalinks_client = None if args.no_metabolites else MetalinksClient()
    agent2_result = run_agent2_chain(agent1_result, metalinks_client=metalinks_client, max_hops=args.max_hops)

    if not agent2_result.chains:
        reason_text = f' LLM\'s stated reason: "{agent2_result.rejection_reason}"' if agent2_result.rejection_reason else " (no reason given)"
        print(
            f"Agent 2 found {agent2_result.candidate_path_count} structural candidate path(s), "
            f"but the LLM did not judge any of them biologically plausible.{reason_text}",
            flush=True,
        )

    print(json.dumps(agent2_result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()
