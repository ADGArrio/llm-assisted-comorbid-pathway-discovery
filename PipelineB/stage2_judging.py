"""Stage 2 Judging: path search in code, then an LLM judge and a pairwise debate.

A breadth-first search over OmniPath gene-gene edges (and optionally MetalinksDB gene-metabolite edges) finds the shortest directed
paths from disease-1 genes to disease-2 genes within max_hops; the search continues past a goal gene and applies max_paths once, as a
trim after the search. A path starts and ends at a gene; a metabolite can only be a middle hop. The LLM never invents edges: it only
marks which code-found candidates are plausible (zero, one or many), and is skipped when there is one candidate. Every chain is checked
back against the BFS edges (EdgeProvenanceError otherwise).

Judge: candidates are judged in the original and the reversed order (union rule: one endorsement keeps a candidate), optionally with a
repeat of the original order as a noise control that never affects the result. Debate: endorsed candidates are compared pairwise in both
A/B orders (a disagreement is a draw) and ranked by Elo, using a round-robin or, for large fields, a single-elimination bracket.

Each chain carries the model's rationale, labelled as unverified commentary, plus a code-built evidence record (curated edge directions,
OmniPath evidence strength, Open Targets scores, the literature titles the model saw) and any other genes the commentary names.
Edges with no curated direction are flagged. NoChainFoundError means no path exists within max_hops; a result with no chains means the
judge endorsed none.
"""
import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from common.config import DISEASE_PAIRS, get_pair
from common.llm_client import LLMClient
from PipelineB.stage1_discovery import Stage1DiscoveryResult, run_stage1_discovery
from PipelineB.metalinks_client import MetaboliteEdge, MetalinksClient
from PipelineB.omnipath_client import EXTENDED_DATASETS, Interaction, OmniPathClient
from PipelineB.pubmed_client import Citation, PubMedClient

DEFAULT_MAX_HOPS = 5
DEFAULT_MAX_CANDIDATE_PATHS = 10
DEFAULT_MAX_PER_START_GENE = 3  # at most this many of the kept candidates may share a start gene (0/None = no cap)
DEFAULT_JUDGE_PASSES = 2  # judge judgements per run: original order, then reversed (1 = single pass)
DEFAULT_JUDGE_MIN_VOTES = 1  # passes that must endorse a candidate for it to survive judge
_CITATIONS_SHOWN_PER_GENE = 2  # literature titles shown to the model per gene (and recorded in `evidence`)
# Disease abbreviations that are also gene symbols (CAD = carbamoyl-phosphate synthetase) or common words; never counted
# as another gene named in a rationale.
DISEASE_ABBREVIATIONS = {"CAD", "AD", "T2DM", "T2D"}
_GENE_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,}(?:_[A-Z0-9]+)*(?![a-z0-9])")
_MAX_FRONTIER_SIZE = 3000  # safety cap on nodes queried per BFS layer
_PARSE_ATTEMPTS = 2  # an unparseable LLM reply is retried once before raising

# Pairwise debate ranking.
_INITIAL_ELO = 1200.0  # initial Elo rating (the Co-Scientist Ranking-agent default)
_ELO_K_FACTOR = 32.0  # standard Elo K factor
_TOURNAMENT_BRACKET_THRESHOLD = 8  # above this count a bracket replaces the round-robin


# One edge of an endorsed chain, with the Open Targets scores of its gene ends.
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
    curation_effort: Optional[int] = None  # OmniPath's evidence count for this edge; None for metabolite edges / old runs
    n_sources: Optional[int] = None  # distinct OmniPath resources listing this edge

    # Output: the edge as a plain dict.
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
            "curation_effort": self.curation_effort,
            "n_sources": self.n_sources,
        }


# One judge-endorsed chain with its rationale, Elo rating and evidence record.
@dataclass
class ChainCandidate:

    edges: List[ChainEdge]
    rationale: Optional[str]
    elo_rating: Optional[float] = None  # set by the pairwise tournament; None if it did not run
    judge_votes: Optional[int] = None  # how many judge passes endorsed it; None = never judged (lone candidate)
    rationale_off_path_genes: List[str] = field(default_factory=list)  # other genes the LLM text names (unverified claims)
    evidence: dict = field(default_factory=dict)  # code-generated, grounded facts about the chain -- see _build_evidence

    # Output: True if any edge has no curated direction flag.
    @property
    def has_unconfirmed_direction(self) -> bool:
        return any(edge.interaction_type == "unknown" for edge in self.edges)

    # Output: the chain as a plain dict.
    def to_dict(self) -> dict:
        return {
            "edges": [e.to_dict() for e in self.edges],
            "rationale": self.rationale,
            "rationale_source": "LLM commentary, not verified against any database",
            "has_unconfirmed_direction": self.has_unconfirmed_direction,
            "elo_rating": self.elo_rating,
            "judge_votes": self.judge_votes,
            "rationale_off_path_genes": self.rationale_off_path_genes,
            "evidence": self.evidence,
        }


# One pairwise debate between two endorsed chains, kept in full with the model's rationale. winner_index and loser_index
# refer to the final chain order.
@dataclass
class DebateComparison:

    winner_index: int
    loser_index: int
    rationale: str
    draw: bool = False  # True when the two A/B orders disagreed -- winner/loser then just name the two sides
    rationale_swapped: Optional[str] = None  # the model's rationale from the swapped-order run (None if not swapped)
    # Index into the final chains list of the chain shown as "Candidate A" in the first run (roles swap in the second
    # run), so the A/B wording in saved rationales can be tied to a chain.
    first_a_index: Optional[int] = None
    unparsed: bool = False  # True when a run gave no parseable verdict twice; the debate then counts as a draw

    # Output: the debate as a plain dict.
    def to_dict(self) -> dict:
        return {
            "winner_index": self.winner_index,
            "loser_index": self.loser_index,
            "rationale": self.rationale,
            "draw": self.draw,
            "rationale_swapped": self.rationale_swapped,
            "first_a_index": self.first_a_index,
            "unparsed": self.unparsed,
        }


# Stage 2 output: the endorsed chains, the candidate count, the debate log and the search statistics.
@dataclass
class Stage2JudgingResult:
    disease1: str
    disease2: str
    chains: List[ChainCandidate]
    candidate_path_count: int
    shared_genes: List[str]
    rejection_reason: Optional[str] = None  # set when chains is empty and the LLM gave a reason for rejecting everything
    debate_log: List[DebateComparison] = field(default_factory=list)  # [] if the tournament did not run
    bfs_stats: dict = field(default_factory=dict)  # search funnel and data provenance; {} on older saved results

    # Output: the result as a JSON-serializable dict.
    def to_dict(self) -> dict:
        return {
            "disease1": self.disease1,
            "disease2": self.disease2,
            "chains": [c.to_dict() for c in self.chains],
            "candidate_path_count": self.candidate_path_count,
            "shared_genes": self.shared_genes,
            "rejection_reason": self.rejection_reason,
            "debate_log": [d.to_dict() for d in self.debate_log],
            "bfs_stats": self.bfs_stats,
        }


# Raised when the BFS finds no directed path within max_hops (an expected outcome, distinct from a result with no
# endorsed chains).
class NoChainFoundError(RuntimeError):
    pass


# Raised if anything in Stage 2's output is not traceable to an edge the BFS pulled from OmniPath or MetalinksDB.
class EdgeProvenanceError(RuntimeError):
    pass


# Source-agnostic edge shape the BFS works on; OmniPath and MetalinksDB edges are both converted to it.
@dataclass
class _RawEdge:

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
    curation_effort: Optional[int] = None
    n_sources: Optional[int] = None


# Input: OmniPath Interactions (always gene to gene). Output: a _RawEdge list.
def _omnipath_to_raw(interactions: List[Interaction]) -> List[_RawEdge]:
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
            curation_effort=getattr(i, "curation_effort", None),
            n_sources=getattr(i, "n_sources", None),
        )
        for i in interactions
    ]


# Input: MetalinksDB edges (always gene-metabolite, one end each). Output: a _RawEdge list.
def _metalinks_to_raw(edges: List[MetaboliteEdge]) -> List[_RawEdge]:
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


# Input: the Stage 1 result. Output: (disease-1 gene -> score dict, disease-2 gene -> score dict).
def _score_maps(stage1_result: Stage1DiscoveryResult) -> Tuple[Dict[str, float], Dict[str, float]]:
    d1 = {t.gene_symbol: t.disease_assoc_score for t in stage1_result.disease1_targets}
    d2 = {t.gene_symbol: t.disease_assoc_score for t in stage1_result.disease2_targets}
    return d1, d2


# Sort key for frontier truncation. Input: one frontier entry ((kind, node id), path so far) and the set of genes with
# an Open Targets score. Output: a key where scored genes (0) sort before everything else (1), with the node id breaking
# ties.
def _frontier_priority(item: Tuple[Tuple[str, str], List[_RawEdge]], scored_genes: set) -> Tuple[int, str]:
    (kind, node_id), _ = item
    is_scored = kind == "gene" and node_id in scored_genes
    return (0 if is_scored else 1, node_id)


# Breadth-first search for shortest directed paths. Input: OmniPath client, MetalinksDB client (None = protein-only),
# start genes, goal genes, hop limit, path limit, and optional stats dict (search funnel), per-start-gene cap and
# gene_universe set (every gene seen on a returned edge). Output: one shortest path (a list of _RawEdge) per goal gene
# reached, trimmed to max_paths.
def _bfs_shortest_paths(
    omnipath_client,
    metalinks_client,
    start_genes: List[str],
    goal_genes: set,
    max_hops: int,
    max_paths: int,
    stats: Optional[dict] = None,
    max_per_start_gene: Optional[int] = None,
    gene_universe: Optional[Set[str]] = None,
) -> List[List[_RawEdge]]:
    visited = {("gene", g) for g in start_genes}
    found_targets: set = set()  # goal genes already given a path
    frontier: List[Tuple[Tuple[str, str], List[_RawEdge]]] = [(("gene", g), []) for g in start_genes]
    found: List[List[_RawEdge]] = []
    depth = 0
    layers_stats: List[dict] = []
    scored_genes = set(start_genes) | goal_genes  # every gene with a real disease-1 or disease-2 association score; used only to prioritize what survives frontier truncation

    # The search always runs the full max_hops depth (or until the frontier is exhausted); max_paths is applied once, as
    # a trim, after the loop.
    while frontier and depth < max_hops:
        depth += 1
        layer = {"depth": depth, "frontier_size": len(frontier), "truncated": len(frontier) > _MAX_FRONTIER_SIZE}
        found_before = len(found)

        if len(frontier) > _MAX_FRONTIER_SIZE:
            print(
                f"WARNING: Stage 2 BFS depth {depth}: frontier of {len(frontier)} nodes exceeds "
                f"the {_MAX_FRONTIER_SIZE}-node safety cap; truncating -- keeping disease-scored "
                f"genes first, then the rest deterministically by id (see module docstring's "
                f"'frontier truncation prioritized' note).",
                flush=True,
            )
            frontier = sorted(frontier, key=lambda item: _frontier_priority(item, scored_genes))[:_MAX_FRONTIER_SIZE]

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

        layer["edges_returned"] = len(raw_edges)
        if gene_universe is not None:
            for e in raw_edges:
                if e.source_kind == "gene":
                    gene_universe.add(e.source_id)
                if e.target_kind == "gene":
                    gene_universe.add(e.target_id)
        by_source: Dict[Tuple[str, str], List[_RawEdge]] = {}
        for edge in raw_edges:
            by_source.setdefault((edge.source_kind, edge.source_id), []).append(edge)

        next_frontier: List[Tuple[Tuple[str, str], List[_RawEdge]]] = []
        for node in paths_by_node:
            path = paths_by_node[node]
            for edge in by_source.get(node, []):
                target_node = (edge.target_kind, edge.target_id)
                new_path = path + [edge]

                if (
                    edge.target_kind == "gene"
                    and edge.target_id in goal_genes
                    and edge.target_id not in found_targets
                ):
                    found_targets.add(edge.target_id)
                    found.append(new_path)

                # Keep expanding past a goal gene, since a more specific mechanism may sit further along the same chain;
                # only `visited` gates re-expansion.
                if target_node not in visited:
                    visited.add(target_node)
                    next_frontier.append((target_node, new_path))
        layer["new_goal_paths"] = len(found) - found_before
        layer["new_nodes"] = len(next_frontier)
        layers_stats.append(layer)
        frontier = next_frontier

    if stats is not None:
        stats["layers"] = layers_stats
        stats["n_paths_before_trim"] = len(found)
        # Every gene between the endpoints of some goal-reaching path, before trimming. Stage 3 draws its random-gene
        # baseline from this pool, so the baseline has the same topology bias as the chains.
        stats["candidate_bridge_genes"] = sorted({e.target_id for p in found for e in p[:-1] if e.target_kind == "gene"})
        stats["paths_by_hops"] = {str(k): v for k, v in sorted(Counter(len(p) for p in found).items())}
        stats["n_paths_with_unknown_direction"] = sum(1 for p in found if _has_unknown_direction(p))
        stats["n_distinct_start_genes"] = len({p[0].source_id for p in found})
        stats["n_distinct_goal_genes"] = len({p[-1].target_id for p in found})
        stats["n_start_genes"] = len(set(start_genes))
        stats["n_goal_genes"] = len(goal_genes)
        stats["any_layer_truncated"] = any(layer["truncated"] for layer in layers_stats)
        stats["frontier_cap"] = _MAX_FRONTIER_SIZE

    if len(found) > max_paths:
        # Trim only after the full search; see _trim_candidates for the order.
        print(
            f"WARNING: Stage 2 BFS found {len(found)} candidate path(s), more than "
            f"max_candidate_paths={max_paths}; keeping {max_paths} (longest first, fully-signed first among "
            f"equal lengths, at most {max_per_start_gene or 'unlimited'} per start gene) and dropping the rest.",
            flush=True,
        )
        found = _trim_candidates(found, max_paths, max_per_start_gene)

    if stats is not None:
        stats["n_paths_after_trim"] = len(found)
        stats["n_distinct_start_genes_after_trim"] = len({p[0].source_id for p in found})
        stats["trim_policy"] = {
            "order": "longest first, then fully-signed first, then discovery order",
            "max_per_start_gene": max_per_start_gene or None,
        }
    return found


# Cut the candidate list to the slot count. Input: all goal-reaching paths in discovery order, the slot count and an
# optional per-start-gene cap. Output: at most max_paths paths, ranked longest first, then fully signed first, then
# discovery order; with a cap, paths from a start gene already holding its share are used afterwards, only if slots
# remain. Evidence-blind apart from the signed-edge tie-break.
def _trim_candidates(
    found: List[List[_RawEdge]], max_paths: int, max_per_start_gene: Optional[int] = None
) -> List[List[_RawEdge]]:
    order = sorted(range(len(found)), key=lambda i: (-len(found[i]), _has_unknown_direction(found[i]), i))
    if not max_per_start_gene or max_per_start_gene <= 0:
        chosen = order[:max_paths]
    else:
        per_start: Counter = Counter()
        first_choice: List[int] = []
        skipped: List[int] = []
        for i in order:
            start = found[i][0].source_id
            if len(first_choice) < max_paths and per_start[start] < max_per_start_gene:
                first_choice.append(i)
                per_start[start] += 1
            else:
                skipped.append(i)
        chosen = first_choice + skipped[: max_paths - len(first_choice)]
    rank = {i: r for r, i in enumerate(order)}
    return [found[i] for i in sorted(chosen, key=rank.get)]


# Input: judge selections [(path, rationale)] and the BFS paths. Raises EdgeProvenanceError unless every selected path
# is, by identity, one of the BFS paths.
def _assert_selected_paths_are_code_found(selections, paths: List[List[_RawEdge]]) -> None:
    code_found = {id(p) for p in paths}
    for path, _ in selections:
        if id(path) not in code_found:
            raise EdgeProvenanceError("Stage 2: a selected chain is not one of the paths the BFS found.")


# Input: the final chains and the BFS paths. Raises EdgeProvenanceError if any chain edge (source, target, type, edge
# id) is not an edge the BFS pulled from OmniPath or MetalinksDB.
def _assert_chain_edges_come_from_bfs(chains, paths: List[List[_RawEdge]]) -> None:
    allowed = {(e.source_id, e.target_id, e.interaction_type, e.edge_id) for p in paths for e in p}
    for chain in chains:
        for e in chain.edges:
            if (e.source_gene, e.target_gene, e.interaction_type, e.omnipath_edge_id) not in allowed:
                raise EdgeProvenanceError(
                    f"Stage 2: chain edge {e.source_gene} -> {e.target_gene} ({e.interaction_type}) "
                    "was not found by the BFS in OmniPath/MetalinksDB."
                )


# Input: a path. Output: whether any edge lacks a curated direction flag.
def _has_unknown_direction(path: List[_RawEdge]) -> bool:
    return any(edge.interaction_type == "unknown" for edge in path)


_UNKNOWN_DIRECTION_CAVEAT = (
    "Caveat: at least one interaction in this pathway has no curated activation/inhibition "
    "flag in OmniPath/MetalinksDB, so its causal direction is not established by curated "
    "evidence -- treat that step's directionality as unconfirmed, not as a known "
    "activate/inhibit relationship, pending further literature review."
)


# Input: a path and its rationale (may be None). Output: the rationale with the fixed caveat sentence appended if any
# edge has an unknown direction, otherwise unchanged.
def _with_unknown_direction_caveat(path: List[_RawEdge], rationale: Optional[str]) -> Optional[str]:
    if not _has_unknown_direction(path):
        return rationale
    if rationale:
        return f"{rationale} {_UNKNOWN_DIRECTION_CAVEAT}"
    return _UNKNOWN_DIRECTION_CAVEAT


_SELECTION_PROMPT_TEMPLATE = """You are a biomedical researcher evaluating several candidate \
molecular pathways, all independently confirmed to exist in real interaction databases \
(OmniPath for protein-protein steps, MetalinksDB for gene<->metabolite steps -- a step written \
as "[metabolite] <name>" is a small-molecule metabolite, not a protein). Each pathway links a \
protein associated with {disease1} to a protein associated with {disease2}. Candidates can have \
different numbers of intermediate steps -- a shorter candidate is not automatically better than a \
longer one; judge each one on the evidence for it, not on its length.

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

How plausible a candidate needs to be depends on how many steps it has:

- A SINGLE-STEP candidate (just one interaction, directly linking the two proteins) must have a \
SPECIFIC, real, named mechanism for you to endorse it (e.g. "X phosphorylates Y, activating Y's \
kinase activity" or "X converts to metabolite Y via enzyme Z") -- "these are known to interact" \
or "this edge exists in a database" is NOT enough on its own. A single unqualified interaction \
has no other structure to lean on, so it has to clear this bar by itself.
- A MULTI-STEP candidate (two or more interactions) needs at least ONE of its steps to have a \
specific, named mechanism you can state confidently. The other step(s) may rest on general \
biological reasoning instead (shared pathway membership, a well-established broad functional \
relationship, a plausible indirect role) as long as you say so plainly when a step is on that \
weaker footing (e.g. "the direct mechanism between X and Y isn't fully specified here, but both \
are established players in cholesterol trafficking"). Reject it if none of its steps has a real, \
specific mechanism, or if a step requires a genuine, unsupported logical leap.

Either way, it's expected and fine for some or all candidates to be rejected -- marking a \
candidate NOT PLAUSIBLE is not a failure, it's the correct answer whenever it's true. Do not \
invent any new steps, genes, or metabolites; judge only the candidates given.

Respond with exactly one line for EVERY candidate listed above (not just the plausible ones), in \
this exact format:
<candidate number>: <PLAUSIBLE or NOT PLAUSIBLE> - <one-sentence rationale naming which step(s) \
have a specific mechanism, noting explicitly any step that instead rests on general reasoning, \
and any UNKNOWN DIRECTION caveat>

The word PLAUSIBLE or NOT PLAUSIBLE right after the colon is mandatory and is the only thing that \
determines whether a candidate is used -- a candidate you discuss but don't explicitly mark \
PLAUSIBLE is treated as rejected, regardless of how positive the rest of your sentence sounds. Do \
not skip a candidate; every one needs its own line and its own verdict.

Alternatively, if none of the candidates are plausible, you may instead respond with a single \
line starting with the word NONE, followed by a colon and one sentence explaining what was \
missing or unconvincing across the candidates as a whole."""


_DEBATE_PROMPT_TEMPLATE = """You are a biomedical researcher comparing two candidate molecular \
pathways head-to-head. Both have already been independently judged biologically plausible, and \
both link a protein associated with {disease1} to a protein associated with {disease2}.

Candidate A:
{candidate_a_text}

Candidate B:
{candidate_b_text}

Argue for which candidate is the MORE biologically plausible mechanism overall. Consider: the \
specificity and strength of the named mechanism at each step, how well-established each gene's \
role in {disease1} and {disease2} actually is, and how much any step leans on general reasoning \
rather than a specific, named interaction. A candidate having more steps is not automatically \
better or worse than a shorter one -- judge the actual evidence for each.

Respond in exactly this format:
WINNER: <A or B>
RATIONALE: <one paragraph explaining why the winner is the stronger candidate, and what \
specifically the other candidate lacks by comparison>"""

_DEBATE_WINNER_RE = re.compile(r"WINNER\s*:\s*([AB])\b", re.IGNORECASE)
_DEBATE_RATIONALE_RE = re.compile(r"RATION\w*\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


# Parse a debate reply. Input: the reply text. Output: (0 for A or 1 for B, rationale text), or None if there is no
# "WINNER: A" / "WINNER: B" line. Only the winner line is required, the rationale label is matched loosely, and the
# verdict is never guessed.
def _parse_debate_verdict(response: str) -> Optional[Tuple[int, str]]:
    winner_match = _DEBATE_WINNER_RE.search(response)
    if not winner_match:
        return None
    rationale_match = _DEBATE_RATIONALE_RE.search(response, winner_match.end())
    rationale = (rationale_match.group(1) if rationale_match else response[winner_match.end():]).strip()
    return (0 if winner_match.group(1).upper() == "A" else 1), (rationale or "(no rationale given)")


# Input: a node's kind and id. Output: the string used to render it in the LLM prompt.
def _node_label(kind: str, node_id: str) -> str:
    return node_id if kind == "gene" else f"[metabolite] {node_id}"


_EDGE_STRENGTH_NOTE = (
    "Note on evidence: a step marked [OmniPath: N resources, curation effort C] shows how many databases "
    "independently report that interaction and how heavily it has been curated. A larger number means the "
    "INTERACTION itself is better established; it says nothing about whether the interaction matters for "
    "the two diseases. A step with only 1 resource is weakly established."
)


# Input: candidate index, path edges, the gene -> citations dict and whether to print each edge's OmniPath evidence
# strength. Output: the prompt text block for this candidate.
def _format_candidate(
    index: int, path: List[_RawEdge], citations: Dict[str, List[Citation]], show_edge_strength: bool = False
) -> str:
    # Input: one edge. Output: its text line in the prompt.
    def _step(e: _RawEdge) -> str:
        label = "UNKNOWN DIRECTION" if e.interaction_type == "unknown" else e.interaction_type
        text = f"{_node_label(e.source_kind, e.source_id)} --{label}--> {_node_label(e.target_kind, e.target_id)}"
        if show_edge_strength and e.n_sources is not None:
            effort = f", curation effort {e.curation_effort}" if e.curation_effort is not None else ""
            text += f" [OmniPath: {e.n_sources} resource{'s' if e.n_sources != 1 else ''}{effort}]"
        return text

    steps = " -> ".join(_step(e) for e in path)
    # Citations are keyed by gene symbol; metabolites have none.
    genes_in_path = sorted(
        {e.source_id for e in path if e.source_kind == "gene"} | {e.target_id for e in path if e.target_kind == "gene"}
    )

    lines = [f"{index}. {steps}"]
    for gene in genes_in_path:
        cites = citations.get(gene) or []
        if cites:
            titles = "; ".join(f'"{c.title}"' for c in cites[:_CITATIONS_SHOWN_PER_GENE])
            lines.append(f"   {gene} literature: {titles}")
    return "\n".join(lines)


# Judge the candidates. Input: candidate paths, disease names, citations, LLM client, an optional stats dict (receives
# {"unjudged": n}, the candidates the reply gave no verdict for, which count as rejected) and whether to show edge
# strength. Output: (selections, rejection_reason). selections is a list of (path, rationale) for candidates the model
# marked PLAUSIBLE; rejection_reason is set only when none was endorsed and the model gave a reason. With a single
# candidate the LLM is skipped. Callers must pass each selection through _with_unknown_direction_caveat.
def _select_paths(
    paths: List[List[_RawEdge]],
    disease1: str,
    disease2: str,
    citations: Dict[str, List[Citation]],
    llm_client,
    stats: Optional[dict] = None,
    show_edge_strength: bool = False,
) -> Tuple[List[Tuple[List[_RawEdge], Optional[str]]], Optional[str]]:
    if len(paths) == 1:
        return [(paths[0], None)], None

    candidates_text = "\n".join(
        _format_candidate(i + 1, p, citations, show_edge_strength) for i, p in enumerate(paths)
    )
    prompt = _SELECTION_PROMPT_TEMPLATE.format(disease1=disease1, disease2=disease2, candidates_text=candidates_text)
    if show_edge_strength:
        prompt = _EDGE_STRENGTH_NOTE + "\n\n" + prompt
    response = llm_client.generate(prompt, temperature=0.2)
    if stats is not None:
        stats["unjudged"] = 0  # overwritten below once the verdict lines have been counted
        # Diagnostics: how long the prompt was and (truncated) what the model replied, to show why candidates went
        # unjudged.
        stats.setdefault("raw_replies", []).append(
            {"candidates_asked": len(paths), "prompt_chars": len(prompt), "reply": response[:3000],
             "finish_reason": getattr(llm_client, "last_finish_reason", None)}
        )

    stripped = response.strip()
    if stripped.upper().startswith("NONE"):
        reason = stripped[len("NONE"):].strip()
        if reason.startswith(":"):
            reason = reason[1:].strip()
        return [], (reason or None)

    # Each candidate needs an explicit PLAUSIBLE or NOT PLAUSIBLE verdict; a bare "<number>: <text>" line is not an
    # endorsement. The separator after the number may be '.', ':' or ')'. Markdown emphasis, bullets and an optional
    # "Candidate " prefix are ignored, and the first verdict for a candidate number wins.
    _VERDICT_RE = re.compile(
        r"\s*(?:candidate\s+)?(\d+)\s*[:.\)]\s*(NOT\s+PLAUSIBLE|IMPLAUSIBLE|PLAUSIBLE)\b[\s:.\-]*(.*)",
        re.IGNORECASE
    )
    seen_indices: set = set()
    selections: List[Tuple[List[_RawEdge], Optional[str]]] = []
    not_plausible: List[Tuple[int, str]] = []

    # Read verdict lines from a reply. Input: the reply and the set of candidate indices still awaiting a verdict.
    # Output: none; fills seen_indices, selections and not_plausible.
    def parse_reply(reply: str, allowed: set) -> None:
        for line in reply.strip().splitlines():
            line = re.sub(r"[*`]", "", line).lstrip(" \t#>-\u2022")  # markdown emphasis, bullets, quote marks
            match = _VERDICT_RE.match(line)
            if not match:
                continue  # tolerate stray preamble/formatting, and lines with no verdict word at all
            index = int(match.group(1)) - 1
            if index not in allowed or index in seen_indices:
                continue  # out of range, or the first verdict for a candidate already won
            seen_indices.add(index)
            verdict = match.group(2).upper()
            reason_text = match.group(3).strip()
            if verdict.startswith(("NOT", "IMPLAUSIBLE")):
                not_plausible.append((index, reason_text or "no reason given"))
            else:
                selections.append((paths[index], reason_text or None))

    parse_reply(response, set(range(len(paths))))
    if not selections and not not_plausible:
        raise RuntimeError(
            f"Stage 2: LLM path-selection response had no parseable "
            f"'<number>: PLAUSIBLE/NOT PLAUSIBLE - <rationale>' lines and wasn't 'NONE' either: {response!r}"
        )

    # Ask once more about only the candidates the reply did not address, keeping their original numbers. Anything still
    # unaddressed is reported as `unjudged` and treated as rejected, never endorsed.
    missing = [i for i in range(len(paths)) if i not in seen_indices]
    if stats is not None:
        stats["unjudged_first_reply"] = len(missing)
        stats["retried_unjudged"] = bool(missing)
    if missing:
        retry_text = "\n".join(_format_candidate(i + 1, paths[i], citations, show_edge_strength) for i in missing)
        retry_prompt = _SELECTION_PROMPT_TEMPLATE.format(disease1=disease1, disease2=disease2, candidates_text=retry_text)
        if show_edge_strength:
            retry_prompt = _EDGE_STRENGTH_NOTE + "\n\n" + retry_prompt
        retry_reply = llm_client.generate(retry_prompt, temperature=0.2)
        if stats is not None:
            stats["raw_replies"].append(
                {"candidates_asked": len(missing), "prompt_chars": len(retry_prompt), "reply": retry_reply[:3000], "retry": True,
                 "finish_reason": getattr(llm_client, "last_finish_reason", None)}
            )
        parse_reply(retry_reply, set(missing))

    if stats is not None:
        stats["unjudged"] = len(paths) - len(seen_indices)

    if not selections:
        reason = "; ".join(f"candidate {i + 1}: {reason}" for i, reason in not_plausible)
        return [], reason

    return selections, None


# Run the judge in the original and reversed candidate order. Input: the usual judge inputs plus passes (1 = single
# pass, 2 = also the reversed list), min_votes (endorsements needed to keep a candidate), control_pass (with two passes,
# judge the original order a second time as a noise control) and show_edge_strength. Output: (selections in original
# order, rejection_reason, votes mapping id(path) to endorsement count, order_stats saved into bfs_stats). Control-pass
# verdicts are recorded in the stats only and never enter votes or selections; selected paths are matched back by object
# identity.
def _select_paths_both_orders(
    paths: List[List[_RawEdge]],
    disease1: str,
    disease2: str,
    citations: Dict[str, List[Citation]],
    llm_client,
    passes: int = DEFAULT_JUDGE_PASSES,
    min_votes: int = DEFAULT_JUDGE_MIN_VOTES,
    control_pass: bool = False,
    show_edge_strength: bool = False,
):
    # Run one judge pass. Input: the candidate list in the order to show and a label for the stats. Output: the pass's
    # selections, with its statistics recorded.
    def one_pass(ordered_paths, label):
        pass_stats: dict = {}
        try:
            result = _select_paths(ordered_paths, disease1, disease2, citations, llm_client,
                                   stats=pass_stats, show_edge_strength=show_edge_strength)
        except RuntimeError as first_error:
            if "no parseable" not in str(first_error):
                raise
            pass_stats = {}
            result = _select_paths(ordered_paths, disease1, disease2, citations, llm_client,
                                   stats=pass_stats, show_edge_strength=show_edge_strength)
        unjudged[label] = pass_stats.get("unjudged", 0)
        diagnostics[label] = {
            "unjudged_first_reply": pass_stats.get("unjudged_first_reply", 0),
            "retried_unjudged": pass_stats.get("retried_unjudged", False),
            "raw_replies": pass_stats.get("raw_replies", []),
        }
        return result

    unjudged: Dict[str, int] = {}
    diagnostics: Dict[str, dict] = {}  # per pass: how often the first reply left candidates unaddressed, plus the raw replies
    first_selections, first_reason = one_pass(paths, "forward")
    if len(paths) == 1 or passes <= 1:
        votes = {id(path): 1 for path, _ in first_selections}
        stats = {"judge_passes": 1 if len(paths) > 1 else 0}
        if len(paths) > 1:
            stats["judge_unjudged_forward"] = unjudged["forward"]
            stats["judge_diagnostics"] = diagnostics
        return first_selections, first_reason, votes, stats

    second_selections, second_reason = one_pass(list(reversed(paths)), "reversed")
    votes: Dict[int, int] = {}
    rationale_of: Dict[int, Optional[str]] = {}
    for selections in (first_selections, second_selections):
        for path, rationale in selections:
            votes[id(path)] = votes.get(id(path), 0) + 1
            rationale_of.setdefault(id(path), rationale)
    in_first = {id(path) for path, _ in first_selections}
    in_second = {id(path) for path, _ in second_selections}
    kept = [(path, rationale_of[id(path)]) for path in paths if votes.get(id(path), 0) >= min_votes]
    stats = {
        "judge_passes": 2,
        "judge_min_votes": min_votes,
        "judge_endorsed_forward": len(in_first),
        "judge_endorsed_reversed": len(in_second),
        "judge_endorsed_both": len(in_first & in_second),
        "judge_order_disagreements": len(in_first ^ in_second),
        "judge_candidates": len(paths),
        "judge_unjudged_forward": unjudged["forward"],
        "judge_unjudged_reversed": unjudged["reversed"],
    }
    if control_pass:
        repeat_selections, _ = one_pass(paths, "forward_repeat")
        in_repeat = {id(path) for path, _ in repeat_selections}
        stats.update({
            "judge_control_pass": True,
            "judge_endorsed_forward_repeat": len(in_repeat),
            # forward vs forward-again: any difference here is sampling noise, not order
            "judge_noise_disagreements": len(in_first ^ in_repeat),
            "judge_unjudged_forward_repeat": unjudged["forward_repeat"],
        })
    stats["judge_diagnostics"] = diagnostics
    reason = None if kept else (first_reason or second_reason)
    return kept, reason, votes, stats


# Run one debate. Input: two (path, rationale) candidates, disease names, citations, LLM client and the edge-strength
# flag. Output: (0 if A won, 1 if B won, or None if the reply had no parseable verdict twice; rationale text). An
# unparseable reply is retried once and then reported as no verdict, never guessed.
def _run_debate(
    candidate_a: Tuple[List[_RawEdge], Optional[str]],
    candidate_b: Tuple[List[_RawEdge], Optional[str]],
    disease1: str,
    disease2: str,
    citations: Dict[str, List[Citation]],
    llm_client,
    show_edge_strength: bool = False,
) -> Tuple[Optional[int], str]:
    path_a, _ = candidate_a
    path_b, _ = candidate_b
    prompt = _DEBATE_PROMPT_TEMPLATE.format(
        disease1=disease1,
        disease2=disease2,
        candidate_a_text=_format_candidate(1, path_a, citations, show_edge_strength),
        candidate_b_text=_format_candidate(1, path_b, citations, show_edge_strength),
    )
    if show_edge_strength:
        prompt = _EDGE_STRENGTH_NOTE + "\n\n" + prompt
    response, verdict = None, None
    for _ in range(_PARSE_ATTEMPTS):  # one retry on an unparseable reply
        response = llm_client.generate(prompt, temperature=0.2)
        verdict = _parse_debate_verdict(response)
        if verdict:
            break
    if not verdict:
        return None, f"(no parseable 'WINNER: A/B' in the model's reply: {response[:300]!r})"
    return verdict


# Input: the winner's and loser's current ratings and the K factor. Output: their updated ratings after one decisive
# result.
def _update_elo(winner_rating: float, loser_rating: float, k: float = _ELO_K_FACTOR) -> Tuple[float, float]:
    expected_winner = 1.0 / (1.0 + 10 ** ((loser_rating - winner_rating) / 400.0))
    expected_loser = 1.0 - expected_winner
    new_winner = winner_rating + k * (1.0 - expected_winner)
    new_loser = loser_rating + k * (0.0 - expected_loser)
    return new_winner, new_loser


# Input: two ratings and the K factor. Output: both ratings after a draw (each scores 0.5).
def _update_elo_draw(rating_a: float, rating_b: float, k: float = _ELO_K_FACTOR) -> Tuple[float, float]:
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    return rating_a + k * (0.5 - expected_a), rating_b + k * (0.5 - (1.0 - expected_a))


# Input: candidate count. Output: every distinct pair (i, j) with i < j.
def _round_robin_pairs(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


# Input: the number of candidates left in a round. Output: that round's pairings (0v1, 2v3, ...); an odd one out gets a
# bye from the caller.
def _single_elimination_bracket_pairs(n: int) -> List[Tuple[int, int]]:
    pairs = []
    i = 0
    while i + 1 < n:
        pairs.append((i, i + 1))
        i += 2
    return pairs


# Debate two candidates from the selections list. Input: the selections, two positions i and j, disease names,
# citations, LLM client, swap_order and the edge-strength flag. Output: (winner index or None, loser index or None,
# rationale, swapped-run rationale, draw, unparsed). Candidate i is shown as "A" in the first run and as "B" in the
# swapped run; with swap_order a win counts only if the same candidate wins both times, otherwise it is a draw.
def _debate_pair(
    selections: List[Tuple[List[_RawEdge], Optional[str]]],
    i: int,
    j: int,
    disease1: str,
    disease2: str,
    citations: Dict[str, List[Citation]],
    llm_client,
    swap_order: bool,
    show_edge_strength: bool = False,
):
    first_offset, first_rationale = _run_debate(
        selections[i], selections[j], disease1, disease2, citations, llm_client, show_edge_strength
    )
    if first_offset is None:
        return None, None, first_rationale, None, True, True
    if not swap_order:
        return (i, j, first_rationale, None, False, False) if first_offset == 0 else (j, i, first_rationale, None, False, False)
    second_offset, second_rationale = _run_debate(
        selections[j], selections[i], disease1, disease2, citations, llm_client, show_edge_strength
    )
    if second_offset is None:
        return None, None, first_rationale, second_rationale, True, True
    first_winner = i if first_offset == 0 else j
    second_winner = j if second_offset == 0 else i  # in the swapped run, "A" is candidate j
    if first_winner == second_winner:
        loser = j if first_winner == i else i
        return first_winner, loser, first_rationale, second_rationale, False, False
    return None, None, first_rationale, second_rationale, True, False


# Rank the endorsed candidates by pairwise debate. Input: selections, disease names, citations, LLM client, swap_order
# and the edge-strength flag. Output: (Elo ratings parallel to selections, full debate log). Few candidates get a
# round-robin and larger fields a single-elimination bracket; a lone candidate skips the tournament and keeps the
# initial rating.
def _run_tournament(
    selections: List[Tuple[List[_RawEdge], Optional[str]]],
    disease1: str,
    disease2: str,
    citations: Dict[str, List[Citation]],
    llm_client,
    swap_order: bool = True,
    show_edge_strength: bool = False,
) -> Tuple[List[float], List[DebateComparison]]:
    n = len(selections)
    ratings = [_INITIAL_ELO] * n
    debate_log: List[DebateComparison] = []
    if n <= 1:
        return ratings, debate_log

    if n <= _TOURNAMENT_BRACKET_THRESHOLD:
        for i, j in _round_robin_pairs(n):
            winner_idx, loser_idx, rationale, swapped, draw, unparsed = _debate_pair(
                selections, i, j, disease1, disease2, citations, llm_client, swap_order, show_edge_strength
            )
            if draw:
                ratings[i], ratings[j] = _update_elo_draw(ratings[i], ratings[j])
                debate_log.append(DebateComparison(i, j, rationale, draw=True, rationale_swapped=swapped,
                                                   first_a_index=i, unparsed=unparsed))
            else:
                ratings[winner_idx], ratings[loser_idx] = _update_elo(ratings[winner_idx], ratings[loser_idx])
                debate_log.append(DebateComparison(winner_idx, loser_idx, rationale, rationale_swapped=swapped,
                                                   first_a_index=i))
        return ratings, debate_log

    # Single-elimination bracket for larger fields.
    remaining = list(range(n))
    while len(remaining) > 1:
        pairs = _single_elimination_bracket_pairs(len(remaining))
        paired_positions = {pos for pair in pairs for pos in pair}
        next_round = []
        for a, b in pairs:
            idx_a, idx_b = remaining[a], remaining[b]
            winner_idx, loser_idx, rationale, swapped, draw, unparsed = _debate_pair(
                selections, idx_a, idx_b, disease1, disease2, citations, llm_client, swap_order, show_edge_strength
            )
            if draw:
                ratings[idx_a], ratings[idx_b] = _update_elo_draw(ratings[idx_a], ratings[idx_b])
                debate_log.append(DebateComparison(idx_a, idx_b, rationale, draw=True, rationale_swapped=swapped,
                                                   first_a_index=idx_a, unparsed=unparsed))
                # A bracket needs one advancer: the higher-rated side (after the draw update), else the earlier one.
                next_round.append(idx_b if ratings[idx_b] > ratings[idx_a] else idx_a)
            else:
                ratings[winner_idx], ratings[loser_idx] = _update_elo(ratings[winner_idx], ratings[loser_idx])
                debate_log.append(DebateComparison(winner_idx, loser_idx, rationale, rationale_swapped=swapped,
                                                   first_a_index=idx_a))
                next_round.append(winner_idx)
        for pos, idx in enumerate(remaining):
            if pos not in paired_positions:
                next_round.append(idx)  # odd one out this round -- advances untouched, no debate
        remaining = next_round
    return ratings, debate_log


# Input: path edges and both diseases' gene -> score maps. Output: a ChainEdge list with scores attached to gene ends
# only.
def _to_chain_edges(
    path: List[_RawEdge],
    disease1_scores: Dict[str, float],
    disease2_scores: Dict[str, float],
) -> List[ChainEdge]:
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
            curation_effort=edge.curation_effort,
            n_sources=edge.n_sources,
        )
        for edge in path
    ]


# Build the grounded record for a chain. Input: a code-found path, both score maps and the gene -> citations dict.
# Output: a dict of facts taken from the databases and from what the model was shown (each edge's curated direction and
# evidence strength, each gene's Open Targets scores, the literature titles in the prompt), with nothing the LLM wrote.
def _build_evidence(
    path: List[_RawEdge],
    disease1_scores: Dict[str, float],
    disease2_scores: Dict[str, float],
    citations: Dict[str, List[Citation]],
) -> dict:
    edges = []
    for e in path:
        edges.append({
            "step": f"{_node_label(e.source_kind, e.source_id)} -> {_node_label(e.target_kind, e.target_id)}",
            "interaction_type": e.interaction_type,
            "direction_curated": e.interaction_type != "unknown",
            "curation_effort": e.curation_effort,
            "n_sources": e.n_sources,
        })
    genes = sorted(
        {e.source_id for e in path if e.source_kind == "gene"} | {e.target_id for e in path if e.target_kind == "gene"}
    )
    gene_rows = []
    for gene in genes:
        gene_rows.append({
            "gene": gene,
            "disease1_assoc_score": disease1_scores.get(gene),
            "disease2_assoc_score": disease2_scores.get(gene),
            "literature_shown_to_model": [
                {"pmid": c.pmid, "title": c.title} for c in (citations.get(gene) or [])[:_CITATIONS_SHOWN_PER_GENE]
            ],
        })
    efforts = [e.curation_effort for e in path if e.curation_effort is not None]
    return {
        "n_edges": len(path),
        "n_edges_with_unknown_direction": sum(1 for e in path if e.interaction_type == "unknown"),
        "weakest_edge_curation_effort": min(efforts) if efforts else None,
        "edges": edges,
        "genes": gene_rows,
    }


# Input: a rationale, the chain it describes and the set of gene symbols seen. Output: sorted gene symbols named in the
# text that are not on this chain (disease abbreviations excluded); they are listed for checking, not used to reject the
# chain.
def _off_path_genes(rationale: Optional[str], path: List[_RawEdge], universe: Set[str]) -> List[str]:
    if not rationale:
        return []
    on_path = {e.source_id for e in path} | {e.target_id for e in path}
    mentioned = {tok for tok in _GENE_TOKEN.findall(rationale) if tok in universe and tok not in DISEASE_ABBREVIATIONS}
    return sorted(mentioned - on_path)


# Add literature for genes Stage 1 did not cover. Input: candidate paths, Stage 1's gene -> citations dict, both disease
# names, a PubMed client and the bfs_stats dict. Output: a copy of the dict extended with up to one paper per disease
# for each path gene that had none. It is a gene-and-disease search, not an edge co-mention search, so Stage 3's
# edge-literature check stays independent; a failed search is skipped and counted.
def _fetch_bridge_citations(paths, citations, disease1, disease2, pubmed_client, stats) -> Dict[str, List[Citation]]:
    merged = {gene: list(cs) for gene, cs in citations.items()}
    genes = sorted({e.source_id for p in paths for e in p if e.source_kind == "gene"}
                   | {e.target_id for p in paths for e in p if e.target_kind == "gene"})
    searched = found_count = errors = 0
    for gene in genes:
        if merged.get(gene):
            continue
        found: List[Citation] = []
        for disease in (disease1, disease2):
            try:
                for c in pubmed_client.search_citations(gene, disease, retmax=1):
                    if all(c.pmid != f.pmid for f in found):
                        found.append(c)
            except Exception:
                errors += 1
        searched += 1
        if found:
            merged[gene] = found
            found_count += 1
    stats.update({"bridge_genes_searched": searched, "bridge_genes_with_citations": found_count,
                  "bridge_citation_search_errors": errors})
    return merged


# Run Stage 2. Input: the Stage 1 result; optional clients (OmniPath defaults to the real client, MetalinksDB to None
# for a protein-only search, LLM to the real client; tests pass stubs); hop and candidate limits;
# enable_debate_tournament; max_per_start_gene; judge_passes and judge_min_votes; swap_debate_order; judge_control_pass;
# judge_evidence (show edge strength and add bridge-gene titles, using pubmed_client); and the PubMed client. The last
# three options default to off here; the CLI and scripts/run_all.py turn them on. Output: a Stage2JudgingResult; raises
# NoChainFoundError if the BFS finds nothing within max_hops.
def run_stage2_judging(
    stage1_result: Stage1DiscoveryResult,
    omnipath_client: Optional[object] = None,
    metalinks_client: Optional[object] = None,
    llm_client: Optional[object] = None,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_candidate_paths: int = DEFAULT_MAX_CANDIDATE_PATHS,
    enable_debate_tournament: bool = True,
    max_per_start_gene: Optional[int] = DEFAULT_MAX_PER_START_GENE,
    judge_passes: int = DEFAULT_JUDGE_PASSES,
    judge_min_votes: int = DEFAULT_JUDGE_MIN_VOTES,
    swap_debate_order: bool = True,
    judge_control_pass: bool = False,
    judge_evidence: bool = False,
    pubmed_client: Optional[object] = None,
) -> Stage2JudgingResult:
    omnipath_client = omnipath_client or OmniPathClient()

    disease1_scores, disease2_scores = _score_maps(stage1_result)
    disease1_genes = list(disease1_scores.keys())
    disease2_genes = set(disease2_scores.keys())
    shared_genes = sorted(disease1_scores.keys() & disease2_scores.keys())

    bfs_stats: dict = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "omnipath_datasets": getattr(omnipath_client, "datasets", None),
        "metabolites_included": metalinks_client is not None,
        "max_hops": max_hops,
        "max_candidate_paths": max_candidate_paths,
    }
    gene_universe: Set[str] = set()
    paths = _bfs_shortest_paths(
        omnipath_client, metalinks_client, disease1_genes, disease2_genes, max_hops, max_candidate_paths,
        stats=bfs_stats, max_per_start_gene=max_per_start_gene, gene_universe=gene_universe,
    )
    if not paths:
        raise NoChainFoundError(
            f"Stage 2: no directed path found between any {stage1_result.disease1}-associated "
            f"protein and any {stage1_result.disease2}-associated protein, within {max_hops} hops, "
            f"starting from {len(disease1_genes)} disease1 genes and {len(disease2_genes)} disease2 genes "
            f"Stage 1 surfaced (search could pass through other genes OmniPath connects them to, and "
            f"metabolites if a MetalinksDB client was provided)."
        )

    llm_client = llm_client or LLMClient()
    judge_citations = stage1_result.citations
    bfs_stats["judge_evidence"] = "none"
    if judge_evidence:
        bfs_stats["judge_evidence"] = "omnipath_edge_strength"
        if pubmed_client is not None:
            judge_citations = _fetch_bridge_citations(
                paths, stage1_result.citations, stage1_result.disease1, stage1_result.disease2, pubmed_client, bfs_stats
            )
            bfs_stats["judge_evidence"] += "+bridge_gene_citations"
    selections, rejection_reason, judge_votes, order_stats = _select_paths_both_orders(
        paths, stage1_result.disease1, stage1_result.disease2, judge_citations, llm_client,
        passes=judge_passes, min_votes=judge_min_votes, control_pass=judge_control_pass,
        show_edge_strength=judge_evidence,
    )
    bfs_stats.update(order_stats)
    _assert_selected_paths_are_code_found(selections, paths)
    bfs_stats["n_judge_endorsed"] = len(selections)
    bfs_stats["llm_model"] = getattr(llm_client, "model", None)

    ratings = [None] * len(selections)
    debate_log: List[DebateComparison] = []
    if selections and enable_debate_tournament and len(selections) > 1:
        ratings, raw_debate_log = _run_tournament(
            selections, stage1_result.disease1, stage1_result.disease2, judge_citations, llm_client,
            swap_order=swap_debate_order, show_edge_strength=judge_evidence,
        )
        bfs_stats["debates_run"] = len(raw_debate_log)
        bfs_stats["debates_unparsed"] = sum(1 for d in raw_debate_log if d.unparsed)
        bfs_stats["debates_swapped_order"] = swap_debate_order
        bfs_stats["debates_position_inconsistent"] = (
            sum(1 for d in raw_debate_log if d.draw and not d.unparsed) if swap_debate_order else None
        )
    else:
        raw_debate_log = []

    # Sort by Elo rating (highest first) when the tournament ran; the sort is stable, so ties and the no-tournament case
    # keep discovery order.
    indexed = list(enumerate(selections))
    tournament_ran = any(r is not None for r in ratings)
    sort_key = (lambda pair: -ratings[pair[0]]) if tournament_ran else (lambda pair: 0)
    indexed_sorted = sorted(indexed, key=sort_key)
    old_to_new_index = {old_i: new_i for new_i, (old_i, _) in enumerate(indexed_sorted)}

    known_genes = (
        gene_universe | set(disease1_scores) | set(disease2_scores) | set(judge_citations)
    )
    chains = [
        ChainCandidate(
            edges=_to_chain_edges(path, disease1_scores, disease2_scores),
            rationale=_with_unknown_direction_caveat(path, rationale),
            elo_rating=ratings[old_i] if ratings and ratings[old_i] is not None else None,
            judge_votes=judge_votes.get(id(path)) if len(paths) > 1 else None,
            rationale_off_path_genes=_off_path_genes(rationale, path, known_genes),
            evidence=_build_evidence(path, disease1_scores, disease2_scores, judge_citations),
        )
        for old_i, (path, rationale) in indexed_sorted
    ]
    bfs_stats["n_chains_with_off_path_gene_in_rationale"] = sum(1 for c in chains if c.rationale_off_path_genes)
    _assert_chain_edges_come_from_bfs(chains, paths)
    debate_log = [
        DebateComparison(
            winner_index=old_to_new_index[d.winner_index],
            loser_index=old_to_new_index[d.loser_index],
            rationale=d.rationale,
            draw=d.draw,
            rationale_swapped=d.rationale_swapped,
            first_a_index=old_to_new_index[d.first_a_index] if d.first_a_index is not None else None,
            unparsed=d.unparsed,
        )
        for d in raw_debate_log
    ]

    return Stage2JudgingResult(
        disease1=stage1_result.disease1,
        disease2=stage1_result.disease2,
        chains=chains,
        candidate_path_count=len(paths),
        shared_genes=shared_genes,
        rejection_reason=rejection_reason if not chains else None,
        debate_log=debate_log,
        bfs_stats=bfs_stats,
    )


# Command-line entry point: run Stages 1 and 2 for a disease pair and print the Stage 2 result as JSON.
def _cli():
    parser = argparse.ArgumentParser(description="Run Stage 1 then Stage 2 (LLM-based Judging) for one disease pair.")
    parser.add_argument("--pair", choices=[p["id"] for p in DISEASE_PAIRS], help="Disease pair id from common/config.py")
    parser.add_argument("--disease1", help="Override: disease 1 name")
    parser.add_argument("--disease2", help="Override: disease 2 name")
    parser.add_argument("--min-score", type=float, default=0.1, help="Minimum Open Targets association score, passed to Stage 1")
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS, help="Max PPI/metabolite hops to search")
    parser.add_argument(
        "--with-metabolites", action="store_true", help="Opt in to MetalinksDB gene<->metabolite hops (default: protein-only)"
    )
    parser.add_argument("--no-metabolites", action="store_true", help="Deprecated no-op: protein-only is now the default")
    parser.add_argument(
        "--no-debate-tournament",
        action="store_true",
        help="Skip the pairwise debate tournament -- keep only the judge's independent PLAUSIBLE/NOT PLAUSIBLE filter (see module docstring)",
    )
    parser.add_argument("--max-per-start-gene", type=int, default=DEFAULT_MAX_PER_START_GENE,
                        help="Trim diversity cap: most kept candidates that may share one start gene (0 = no cap)")
    parser.add_argument("--judge-passes", type=int, choices=[1, 2], default=DEFAULT_JUDGE_PASSES,
                        help="2 = also judge candidates in reversed order (measures order bias); 1 = single judgement")
    parser.add_argument("--no-swap-debates", action="store_true",
                        help="Run each debate once instead of in both A/B orders")
    parser.add_argument("--no-judge-control", action="store_true",
                        help="Skip the extra forward-order judge pass that measures sampling noise")
    parser.add_argument("--no-judge-evidence", action="store_true",
                        help="Do not show the judge OmniPath edge strength or bridge-gene PubMed titles")
    parser.add_argument(
        "--omnipath-datasets",
        default=None,
        help=(
            "OmniPath dataset scope. Omit for the default core 'omnipath' set. Pass 'extended' "
            f"as shorthand for '{EXTENDED_DATASETS}' (adds pathwayextra + kinaseextra, still-"
            "curated but broader coverage), or a raw comma-separated dataset list. See "
            "omnipath_client.py's module docstring and the Stage 2 BFS and prompt recalibration note."
        ),
    )
    args = parser.parse_args()

    if args.pair:
        pair = get_pair(args.pair)
        disease1, disease2 = pair["disease1"], pair["disease2"]
    elif args.disease1 and args.disease2:
        disease1, disease2 = args.disease1, args.disease2
    else:
        parser.error("Pass --pair, or both --disease1 and --disease2.")

    print(f"Running Stage 1 (Discovery) for {disease1} / {disease2}...", flush=True)
    stage1_result = run_stage1_discovery(disease1, disease2, min_score=args.min_score)

    print("Running Stage 2 (LLM-based Judging)...", flush=True)
    metalinks_client = MetalinksClient() if args.with_metabolites else None
    if args.omnipath_datasets == "extended":
        omnipath_client = OmniPathClient(datasets=EXTENDED_DATASETS)
    elif args.omnipath_datasets:
        omnipath_client = OmniPathClient(datasets=args.omnipath_datasets)
    else:
        omnipath_client = None  # OmniPathClient()'s own default (the core "omnipath" set)
    stage2_result = run_stage2_judging(
        stage1_result,
        omnipath_client=omnipath_client,
        metalinks_client=metalinks_client,
        max_hops=args.max_hops,
        enable_debate_tournament=not args.no_debate_tournament,
        max_per_start_gene=args.max_per_start_gene,
        judge_passes=args.judge_passes,
        swap_debate_order=not args.no_swap_debates,
        judge_control_pass=not args.no_judge_control,
        judge_evidence=not args.no_judge_evidence,
        pubmed_client=None if args.no_judge_evidence else PubMedClient(),
    )

    if not stage2_result.chains:
        reason_text = f' LLM\'s stated reason: "{stage2_result.rejection_reason}"' if stage2_result.rejection_reason else " (no reason given)"
        print(
            f"Stage 2 found {stage2_result.candidate_path_count} structural candidate path(s), "
            f"but the LLM did not judge any of them biologically plausible.{reason_text}",
            flush=True,
        )

    print(json.dumps(stage2_result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()
