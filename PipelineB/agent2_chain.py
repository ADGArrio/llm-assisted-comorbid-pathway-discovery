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
could in theory miss a legitimate longer path through it. Each goal gene
is only *recorded* once, the first time it's reached (giving each goal
its own shortest path, tracked via `found_targets`) -- but the search
now keeps expanding PAST a goal gene too (see "goal-gene expansion" note
below), and stops overall once `max_hops` is reached, the frontier is
exhausted, or `max_paths` results are collected.

Goal-gene expansion, fixed 2026-09-14: a goal gene is any gene Open
Targets independently scored >= min_score for disease2 (see
opentargets_client.py) -- reaching one is a "hit" in the sense that it
gets recorded as a found path, but a gene can clear that bar on its own
while being a weak, generic, or upstream player, with a much more
specific and well-characterized mechanism sitting one or two hops
further down the exact same real edge chain. Concretely: on a real
T2DM->AD run, INSR --activates--> PIK3R1 was a genuine 1-hop candidate
(PIK3R1 is the real PI3K subunit and the literal textbook second step
of the insulin-signaling->AD cascade) and PIK3R1 itself independently
cleared Open Targets' AD-association threshold -- so the old code
recorded that hit and, critically, never queried PIK3R1's own outgoing
edges, meaning AKT1->GSK3B->MAPT(tau) -- the well-established, more
disease-defining continuation of that exact pathway -- was never even
attempted, regardless of `max_hops`. The old code enforced this by
putting a goal-gene target into `found` and an `elif` that only added
non-goal targets to `next_frontier`, so a goal-gene node could never be
expanded past. Fixed by expanding EVERY newly-visited node regardless of
goal-gene membership (only `visited` gates re-expansion now); a goal
gene still gets exactly one recorded path (via `found_targets`, checked
independently of expansion), but the search now also continues past it
looking for other, possibly more specific, goal genes reachable deeper
through it. This does not privilege longer chains over shorter ones --
both the short hit and any longer one found by continuing past it are
offered to the LLM as separate candidates, and `_SELECTION_PROMPT_TEMPLATE`
explicitly tells it length is not a plausibility signal either way.

BFS depth-exhaustion bug, fixed 2026-09-14: the loop used to also stop
entirely the moment ANY goal gene was reached, even at depth 1, never
proceeding to depth 2+ to look for other goal genes only reachable
deeper -- regardless of `max_hops`. A real T2DM->AD run surfaced this:
Agent 1's gene pools happened to have several disease2 genes one
generic, uncurated-direction OmniPath interaction away from a disease1
gene (e.g. INSR -unknown-> AXL/ATP1A1/CLTC/NEU1/PIKFYVE), so BFS reported
those 6 depth-1 candidates and quit, never even attempting depth 2-5
where a more specific, better-supported mechanism might connect other
gene pairs. Fixed by only skipping *already-satisfied* goal genes
(`found_targets`) rather than halting the whole search on first success,
so a deeper, more specific candidate to a *different* goal gene now gets
a chance to be found too, up to `max_hops`/`max_paths`. This does not
remove the shallow candidates (they're often real edges, just weakly
characterized) -- it adds better ones alongside them for the LLM to
actually choose between. See claude/agent2_bfs_and_prompt_recalibration.md.

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

Plausibility bar loosened, then recalibrated (both 2026-09-14): the
original bar required a specific, named mechanism for every single hop,
which was rejecting all 10 candidates outright on real T2DM->AD runs (see
claude/agent2_findings_maxhops10_run.md). Loosening it to "a coherent
overall story, general reasoning is enough per hop" over-corrected: on
the very next real run it endorsed all 6 depth-1, uncurated-direction
candidates from the BFS bug above with near-identical templated
rationale text -- effectively the same 100%-acceptance,
no-real-discrimination failure mode already documented in
claude/agent2_findings_maxhops10_run.md's Finding 1, just reproduced
under a different prompt. The bar is now hop-count-tiered: a single-hop
candidate (the whole "chain" is one interaction) must have a specific,
real, named mechanism to be endorsed -- "these proteins are known to
interact" / a bare database edge is not enough on its own, which is
exactly what the BFS bug's depth-1 junk was. A multi-hop candidate only
needs ONE of its hops to have a specific, named mechanism; the rest may
rest on general biological reasoning if the LLM says so explicitly.
Rationale: a single unqualified interaction has no internal structure to
lean on, so it needs to clear the high bar on its own; a real multi-step
mechanism can have a genuinely strong anchor hop plus weaker-but-plausible
supporting hops, which is a different (more defensible) thing than a
lone guess. See claude/agent2_bfs_and_prompt_recalibration.md for the
full diagnosis (this is the second time this same local model has
defaulted to near-0% or near-100% acceptance depending on prompt
wording, worth flagging honestly in the thesis methods/limitations
section). `claude/possible_metrics.md` separately tracks the
not-yet-built plan to compare strict/lenient variants as a metric --
this recalibration is a direct change to the one production prompt, not
that experiment.

Explicit per-candidate verdicts required (2026-09-14, third pass same
day): the hop-tiered bar above had zero effect on a real run -- the LLM
endorsed all 10 candidates again, and 4 of the 10 rationales *literally
said* the mechanism "isn't specifically detailed" while still being
counted as endorsed. Root cause: the model, given a numbered list of 10
candidates, wrote one line commenting on every single one (rather than
only the plausible ones, as the old prompt asked), and the old parser
treated any "<number>: <anything>" line as an endorsement -- it had no
way to tell "here's why I'm rejecting this" apart from "here's why I
endorse this," so a candidate discussed-and-rejected got endorsed
anyway. Fixed by requiring an explicit PLAUSIBLE / NOT PLAUSIBLE verdict
word right after the candidate number on every line, and having
`_select_paths` only treat a line as a selection when that verdict is
literally PLAUSIBLE -- a candidate the model discusses without that
exact word is rejected, no matter how positive its sentence reads. This
also plays to the model's own apparent habit of commenting on every
item in a numbered list, instead of fighting it.
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
    # Output: shortest paths (each a list of _RawEdge, >= 1 edge long) -- one shortest path per
    # distinct goal gene reached (tracked via found_targets, so a goal already satisfied at a
    # shallower depth can't also be found again at a deeper one). A path's first and last node
    # are always genes; a metabolite (when metalinks_client is given) can only be a middle hop.
    # Reaching a goal gene records it but does NOT stop the search from continuing past it --
    # see the module docstring's "goal-gene expansion" note.
    visited = {("gene", g) for g in start_genes}
    found_targets: set = set()  # goal genes already given a path -- see module docstring's BFS note
    frontier: List[Tuple[Tuple[str, str], List[_RawEdge]]] = [(("gene", g), []) for g in start_genes]
    found: List[List[_RawEdge]] = []
    depth = 0

    while frontier and depth < max_hops and len(found) < max_paths:
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
                new_path = path + [edge]

                if (
                    edge.target_kind == "gene"
                    and edge.target_id in goal_genes
                    and edge.target_id not in found_targets
                ):
                    found_targets.add(edge.target_id)
                    found.append(new_path)
                    if len(found) >= max_paths:
                        return found

                # Keep exploring past this node regardless of whether it was just recorded as a
                # goal gene -- a gene independently clearing Open Targets' disease2 threshold
                # doesn't mean a more specific mechanism doesn't sit further down the same real
                # chain (see the module docstring's "goal-gene expansion" note). Only `visited`
                # gates re-expansion now, not goal-gene membership.
                if target_node not in visited:
                    visited.add(target_node)
                    next_frontier.append((target_node, new_path))
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
    # LLM explicitly marked PLAUSIBLE -- zero, one, or many. rejection_reason is set only when
    # selections is empty AND the LLM gave a reason (its blanket "NONE" reason, or, if it instead
    # tagged every candidate individually, its NOT PLAUSIBLE reasons joined together). Skips the LLM
    # (returns the single path, rationale=None, reason=None) when there's nothing to adjudicate.
    # Callers must still run every returned tuple through _with_unknown_direction_caveat.
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

    # Requires an explicit PLAUSIBLE/NOT PLAUSIBLE verdict per line -- a bare "<number>: <text>"
    # line (no verdict word) is NOT treated as an endorsement. This was a real bug: the model
    # would write a line for every candidate, including ones whose own sentence said the mechanism
    # "isn't specifically detailed" -- i.e. a rejection in substance -- and the old regex (which
    # accepted any "<number>: <anything>" line) endorsed all of them anyway, because it had no way
    # to tell a discussed-but-rejected candidate apart from an endorsed one. See
    # claude/agent2_bfs_and_prompt_recalibration.md.
    _VERDICT_RE = re.compile(r"\s*(\d+)\s*:\s*(NOT\s+PLAUSIBLE|PLAUSIBLE)\b[\s:.\-]*(.*)", re.IGNORECASE)
    selections: List[Tuple[List[_RawEdge], Optional[str]]] = []
    not_plausible: List[Tuple[int, str]] = []
    for line in response.strip().splitlines():
        match = _VERDICT_RE.match(line)
        if not match:
            continue  # tolerate stray preamble/formatting, and lines with no verdict word at all
        index = int(match.group(1)) - 1
        if not (0 <= index < len(paths)):
            continue
        verdict = match.group(2).upper()
        reason_text = match.group(3).strip()
        if "NOT" in verdict:
            not_plausible.append((index, reason_text or "no reason given"))
        else:
            selections.append((paths[index], reason_text or None))

    if not selections and not not_plausible:
        raise RuntimeError(
            f"Agent 2: LLM path-selection response had no parseable "
            f"'<number>: PLAUSIBLE/NOT PLAUSIBLE - <rationale>' lines and wasn't 'NONE' either: {response!r}"
        )

    if not selections:
        reason = "; ".join(f"candidate {i + 1}: {reason}" for i, reason in not_plausible)
        return [], reason

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
