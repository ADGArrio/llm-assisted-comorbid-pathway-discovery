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
exactly; `_cli()` is also protein-only unless `--with-metabolites` is given: the
metabolite graph was evaluated and removed from the default search (it added
two calcium paths and nothing else; see the thesis). The LLM never invents edges -- it only selects which of the
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
nodes queried per BFS layer (logs a warning + truncates if exceeded, with
disease-scored genes protected -- see "frontier truncation prioritized"
note below) to keep request size/runtime bounded; a truncated unscored
node could in theory miss a legitimate longer path through it. Each goal gene
is only *recorded* once, the first time it's reached (giving each goal
its own shortest path, tracked via `found_targets`) -- but the search
now keeps expanding PAST a goal gene too (see "goal-gene expansion" note
below). The search runs the FULL `max_hops` depth (or until the frontier
is exhausted) regardless of how many candidates have already been found
-- `max_paths` is enforced once, as a trim, only after the search
completes (see "max_paths cutting off exploration too early" note
below), not as an early-stop condition during the search itself.

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
through it. Both the short hit and any longer one found by continuing
past it are offered to the LLM as separate candidates -- unless there
are more than `max_paths` total, in which case the post-search trim
below prefers to keep the longer ones (this is the one place length IS
used as a signal, and only to decide what the LLM even gets to see;
`_SELECTION_PROMPT_TEMPLATE` still tells the LLM itself that length is
not a plausibility signal once a candidate is in front of it).

`max_paths` cutting off exploration too early, fixed 2026-09-14 (same
day as the goal-gene expansion fix above, found on the very next real
run): the goal-gene expansion fix meant the search could keep going past
a hit -- but the loop also still stopped as soon as `len(found) >=
max_paths` (10 by default), *during* the search, the instant that many
candidates existed. On a real T2DM->AD run, depth 1 alone produced 10
distinct goal-gene hits (the same shallow single-hop hits described
above -- PPARG->CDKN1A, PPARG->NR1H2, INSR->PIK3R1, etc.), which filled
`max_paths` before depth 2 was ever reached -- so the goal-gene expansion
fix never got a chance to run in practice; the search returned right
after depth 1 regardless. Fixed by dropping `max_paths` as a loop
condition entirely -- the search now always runs the full `max_hops`
depth (bounded compute-wise by `_MAX_FRONTIER_SIZE` per layer, not by
how many candidates exist) -- and enforcing `max_paths` exactly once,
after the search, by keeping the candidates with the *most* hops and
dropping the rest (a stable sort, so same-length candidates keep their
original discovery order). This is a deliberate bias: the whole point of
searching past a hit is to find a more specific mechanism, so if there
turn out to be more real candidates than `max_paths` allows, the ones
worth keeping are the ones that motivated the search in the first place,
not an arbitrary first-10 subset dominated by shallow single-hop edges
(which the plausibility bar already treats most skeptically). Tradeoff,
stated honestly: a single, very strong 1-hop mechanism could in theory
be trimmed out in favor of several longer, weaker ones -- this hasn't
been observed in practice (the real failure mode so far has been the
opposite: many weak short candidates crowding out any long one), but
it's a real possible cost of this policy, worth noting in Limitations.

Verdict-line separator widened, fixed 2026-09-14 (same day, found on the
same real run as the fix above): the run also raised a RuntimeError
("no parseable ... lines") even though the LLM answered in good faith --
it wrote "1. NOT PLAUSIBLE - ..." (an ordinary numbered-list period)
instead of "1: NOT PLAUSIBLE - ..." (the colon the prompt's example
uses). `_VERDICT_RE` required a literal colon right after the digit, so
it matched nothing on any of the 10 lines, and the code's fallback logic
(no `NONE`, but also literally zero parsed lines) had no way to
distinguish "the model refused the format" from "the model used
different punctuation for a well-formed verdict," so it raised rather
than silently guessing. This is the same local model that used a colon
on an earlier real run and a period on this one, under similar prompts
-- consistent with the model's own habits being the real source of
format drift, not a one-off fluke -- so the fix is parser tolerance, not
another prompt reword (same philosophy as the verdict-word fix above):
'.', ':', and ')' are all now accepted as the separator between the
candidate number and its verdict word.

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

Frontier truncation prioritized, fixed 2026-09-14 (found on a real run
with all prior fixes in): the truncation sort key (`_frontier_priority`)
keeps every gene with a real disease1 or disease2 Open Targets score
(`scored_genes = set(start_genes) | goal_genes`) ahead of any unscored
gene or metabolite, falling back to the node id only to break ties
deterministically within each tier -- instead of the old plain
alphabetical sort, which was a real, silent sampling bias with no
correlation to biological relevance (a bridge or goal gene sorting late
in the alphabet, e.g. GSK3B or MAPT, could be dropped from a crowded
frontier before its own outgoing edges were ever queried, undermining
the whole point of the goal-gene-expansion fix above). This guarantees a
scored gene is never dropped by the cap as long as the total scored-gene
count doesn't itself exceed `_MAX_FRONTIER_SIZE`. It does NOT fully
solve the underlying problem: the bulk of each frontier is still
unscored bridge genes, and ties among those still break alphabetically
-- an unscored bridge gene needed to reach a deeper target can still be
dropped if it sorts late among the unscored majority.

`_MAX_FRONTIER_SIZE` raised from 300 to 3000, same day, per explicit
user direction: real per-depth frontier sizes on a T2DM->AD run were
599-3308 (depths 2-5) against the old 300-node cap -- as little as ~13%
of a given depth's real graph was ever queried, meaning most of what
biology actually connects the two diseases with was being silently
discarded regardless of the truncation-priority fix above. The cap
existed partly out of a (reasonable at the time) concern about OmniPath
request size/runtime with one big query string; that concern is now
handled separately -- see omnipath_client.py's `get_outgoing_interactions`
batching note -- so the cap no longer needs to stay this low for that
reason. 3000 comfortably clears both real depth-2/3 frontiers observed
so far without removing the safety net entirely (an actually-unbounded
frontier risks runaway growth at depth 4-5 on a dense graph, especially
with `omnipath_datasets="extended"`). Still genuinely untested at this
new size from this network-blocked sandbox -- if real runs keep hitting
even this cap, raising it further is a reasonable next step, watching
for OmniPath request/runtime issues as the signal to stop.

Complex nodes: tried filtering them out the same day (they can't carry
an Open Targets score, can't be confirmed by Agent 3, and rendered as
confusing long strings), then reverted per explicit direction -- see
omnipath_client.py's module docstring's "Complex nodes are kept" note
and claude/open_issues_for_committee.md. BFS sees them again as ordinary
edges (their "gene" is just never a `goal_genes` member, so they can
only ever be bridge hops, same as before this note was added).

Pairwise debate ranking (added 2026-09-27, per committee feedback -- see
claude/committee_feedback_round2_coscientist.md section 3b, and Gottweis
et al. 2026's "Co-Scientist" Ranking agent): `_select_paths` above (now
"stage 1") judges every candidate independently against a fixed rubric
-- it never compares two candidates against each other, so nothing in
its output actually answers "why would a scientist support this pathway
OVER that one," only "is this pathway plausible in isolation." Stage 2,
`_run_tournament` below, adds exactly that: every pair of stage-1-
endorsed candidates is compared head-to-head via a simulated debate
(`_run_debate`), the LLM picks a winner with a rationale, and an
Elo-style rating (starting at 1200, Co-Scientist's own stated default)
is updated after each debate. Every debate's winner AND its rationale
are kept in `Agent2ChainResult.debate_log`, not just the final ranking
-- the rationale is the actual answer to "why," a real head-to-head
justification on file for every comparison, not one isolated verdict.
Cost/scaling: comparing every pair costs O(n^2) LLM calls, fine for the
typical n<=3-4 candidates this pipeline actually produces, but a
single-elimination bracket (`_single_elimination_bracket_pairs`) is used
instead once a run has more than `_TOURNAMENT_BRACKET_THRESHOLD`
candidates, trading full pairwise coverage for bounded cost (~n debates
instead of ~n^2/2). A single stage-1 selection skips the tournament
entirely (nothing to compare) and keeps the initial 1200 rating.
`run_agent2_chain(..., enable_debate_tournament=False)` / `_cli()`'s
`--no-debate-tournament` disable this stage outright -- it multiplies
LLM calls on top of an already-slow local Ollama backend, so it's worth
timing on a real run before treating it as free.

Hardening changes, 2026-10-18 (full reasoning and measurements in
claude/agent2_hardening_provenance_maxhops.md and thesis Ch. 2/3):
  * OmniPath tie rule: an edge with both the stimulation and inhibition flag set is now resolved by
    OmniPath's own consensus columns, else "unknown" (omnipath_client.py). It used to be "activates".
  * Candidate trim (`_trim_candidates`): still longest-first, but among equal lengths fully-signed paths
    come first, and no start gene may supply more than `max_per_start_gene` (default 3) of the kept
    candidates unless slots would otherwise go empty -- a counter to INSR/PPARG-rooted candidates
    filling every slot.
  * Stage 1 is judged in two candidate orders (original and reversed; `stage1_passes`); a candidate
    endorsed in either pass survives, and `stage1_votes` records how many passes endorsed it. The
    share of candidates whose verdict flips with order is saved as a measured order-bias statistic.
  * Stage 2 debates are run in both A/B orders (`swap_debate_order`); a win counts only if the same
    candidate wins both times, otherwise the debate is a draw (Elo score 0.5). The share of draws is
    the measured position-bias statistic.
  * Every chain carries a code-generated `evidence` block (edge directions, OmniPath curation effort,
    sources, disease scores, the literature titles actually shown to the model); the LLM rationale is
    labelled model commentary (unverified) and any other gene it names is listed in
    `rationale_off_path_genes`. The prompts were deliberately NOT changed: earlier prompt edits swung
    this model's acceptance between ~0% and ~100%, so unverified text is detected after the fact
    instead of being discouraged in the prompt.
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
from PipelineB.agent1_discovery import Agent1DiscoveryResult, run_agent1_discovery
from PipelineB.metalinks_client import MetaboliteEdge, MetalinksClient
from PipelineB.omnipath_client import EXTENDED_DATASETS, Interaction, OmniPathClient
from PipelineB.pubmed_client import Citation, PubMedClient

DEFAULT_MAX_HOPS = 5
DEFAULT_MAX_CANDIDATE_PATHS = 10
DEFAULT_MAX_PER_START_GENE = 3  # at most this many of the kept candidates may share a start gene (0/None = no cap)
DEFAULT_STAGE1_PASSES = 2  # stage-1 judgements per run: original order, then reversed (1 = single pass)
DEFAULT_STAGE1_MIN_VOTES = 1  # passes that must endorse a candidate for it to survive stage 1
_CITATIONS_SHOWN_PER_GENE = 2  # literature titles shown to the model per gene (and recorded in `evidence`)
# Disease abbreviations that are also gene symbols (CAD = carbamoyl-phosphate synthetase) or common words;
# never counted as an "other gene named in a rationale".
DISEASE_ABBREVIATIONS = {"CAD", "AD", "T2DM", "T2D"}
_GENE_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,}(?:_[A-Z0-9]+)*(?![a-z0-9])")
_MAX_FRONTIER_SIZE = 3000  # safety cap per BFS layer; see module docstring's "_MAX_FRONTIER_SIZE raised" note
_PARSE_ATTEMPTS = 2  # an LLM reply that cannot be parsed is retried once before raising (added 2026-10-15)

# Pairwise debate ranking (stage 2) -- see module docstring's "Pairwise debate ranking" note.
_INITIAL_ELO = 1200.0  # matches Co-Scientist's own stated Ranking-agent default (Gottweis et al. 2026)
_ELO_K_FACTOR = 32.0  # standard chess/Elo default; not tuned against real pipeline data yet
_TOURNAMENT_BRACKET_THRESHOLD = 8  # candidate count above which a single-elimination bracket
# replaces full round-robin, bounding cost to ~n debates instead of ~n^2/2 -- see module docstring.


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


@dataclass
class ChainCandidate:
    """One LLM-endorsed plausible chain, with its rationale. A single
    Agent2ChainResult can hold more than one of these."""

    edges: List[ChainEdge]
    rationale: Optional[str]
    elo_rating: Optional[float] = None  # set by the stage-2 pairwise tournament; None if the
    # tournament didn't run (single-candidate bypass, or enable_debate_tournament=False) -- see
    # module docstring's "Pairwise debate ranking" note.
    stage1_votes: Optional[int] = None  # how many stage-1 passes endorsed it; None = never judged (lone candidate)
    rationale_off_path_genes: List[str] = field(default_factory=list)  # other genes the LLM text names (unverified claims)
    evidence: dict = field(default_factory=dict)  # code-generated, grounded facts about the chain -- see _build_evidence

    @property
    def has_unconfirmed_direction(self) -> bool:
        # True if any edge has no curated direction flag. See module docstring.
        return any(edge.interaction_type == "unknown" for edge in self.edges)

    def to_dict(self) -> dict:
        return {
            "edges": [e.to_dict() for e in self.edges],
            "rationale": self.rationale,
            "rationale_source": "LLM commentary, not verified against any database",
            "has_unconfirmed_direction": self.has_unconfirmed_direction,
            "elo_rating": self.elo_rating,
            "stage1_votes": self.stage1_votes,
            "rationale_off_path_genes": self.rationale_off_path_genes,
            "evidence": self.evidence,
        }


@dataclass
class DebateComparison:
    """One pairwise debate between two stage-1-endorsed candidate chains, kept in full so the
    model's rationale for preferring one over the other stays on record -- see module docstring's
    "Pairwise debate ranking" note. `winner_index`/`loser_index` refer to position in the final
    `Agent2ChainResult.chains` list (i.e. already re-sorted by final Elo rating), not the
    mid-tournament ordering, so a comparison can always be matched back to the chains it judged."""

    winner_index: int
    loser_index: int
    rationale: str
    draw: bool = False  # True when the two A/B orders disagreed -- winner/loser then just name the two sides
    rationale_swapped: Optional[str] = None  # the model's rationale from the swapped-order run (None if not swapped)
    # Which chain (index into the final `chains` list) was shown to the model as "Candidate A" in the FIRST run;
    # in the swapped run the roles are reversed. Without this, the words "Candidate A/B" in the saved rationale
    # texts cannot be tied to a chain (added 2026-10-18 after a saved debate said "Candidate B" with winner_index 0).
    first_a_index: Optional[int] = None
    unparsed: bool = False  # True when a run gave no parseable verdict twice; the debate then counts as a draw

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


@dataclass
class Agent2ChainResult:
    disease1: str
    disease2: str
    chains: List[ChainCandidate]
    candidate_path_count: int
    shared_genes: List[str]
    rejection_reason: Optional[str] = None  # set when chains is empty and the LLM gave a reason for rejecting everything
    debate_log: List[DebateComparison] = field(default_factory=list)  # [] if the tournament didn't run; see module docstring
    bfs_stats: dict = field(default_factory=dict)  # search funnel + data provenance, added 2026-10-15; {} on results saved before then

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


class NoChainFoundError(RuntimeError):
    """Raised when BFS finds no directed path at all within max_hops --
    a real, expected possible outcome, not a bug. Distinct from a
    successful result with an empty `chains` list, which means BFS found
    candidates but the LLM judged none of them plausible."""


class EdgeProvenanceError(RuntimeError):
    """Raised if anything in Agent 2's output is not traceable to an edge the BFS pulled from
    OmniPath/MetalinksDB. This should be impossible by construction (the LLM only returns candidate
    numbers and text); the check exists so that a future code change that breaks the invariant fails
    loudly instead of silently shipping an ungrounded edge."""


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
    curation_effort: Optional[int] = None
    n_sources: Optional[int] = None


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
            curation_effort=getattr(i, "curation_effort", None),
            n_sources=getattr(i, "n_sources", None),
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


def _frontier_priority(item: Tuple[Tuple[str, str], List[_RawEdge]], scored_genes: set) -> Tuple[int, str]:
    # Input: one frontier entry ((kind, node_id), path-so-far), the set of genes with a real
    # disease1 or disease2 Open Targets score. Output: a sort key for frontier truncation --
    # see the module docstring's "frontier truncation prioritized" note. Lower sorts first
    # (kept); a scored gene is always priority 0 (kept over any unscored node), everything else
    # (unscored genes, metabolites) is priority 1; node id breaks ties deterministically.
    (kind, node_id), _ = item
    is_scored = kind == "gene" and node_id in scored_genes
    return (0 if is_scored else 1, node_id)


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
    # Input: OmniPath client, MetalinksDB client (None = protein-only search, no metabolite hops),
    # start gene pool, goal gene set, hop limit, max paths to collect.
    # Output: shortest paths (each a list of _RawEdge, >= 1 edge long) -- one shortest path per
    # distinct goal gene reached (tracked via found_targets, so a goal already satisfied at a
    # shallower depth can't also be found again at a deeper one). If `stats` (a dict) is given it is
    # filled with the search funnel -- per-layer sizes, how many paths existed BEFORE the max_paths
    # trim, hop-count distribution, unknown-direction count -- see run_agent2_chain's bfs_stats.
    # `max_per_start_gene` (None/0 = off) and the trim order: see _trim_candidates. If `gene_universe` (a set)
    # is given, every gene symbol the search saw on any returned edge is added to it (used to spot gene
    # names in LLM text that are not on the chain). A path's first and last node
    # are always genes; a metabolite (when metalinks_client is given) can only be a middle hop.
    # Reaching a goal gene records it but does NOT stop the search from continuing past it --
    # see the module docstring's "goal-gene expansion" note.
    visited = {("gene", g) for g in start_genes}
    found_targets: set = set()  # goal genes already given a path -- see module docstring's BFS note
    frontier: List[Tuple[Tuple[str, str], List[_RawEdge]]] = [(("gene", g), []) for g in start_genes]
    found: List[List[_RawEdge]] = []
    depth = 0
    layers_stats: List[dict] = []
    scored_genes = set(start_genes) | goal_genes  # every gene with a real disease1 or disease2
    # association score -- used only to prioritize what survives frontier truncation, see below.

    # NOTE: the loop no longer stops early once len(found) reaches max_paths -- see the module
    # docstring's "max_paths cutting off exploration too early" note. It only stops on
    # `max_hops` or an exhausted frontier; `max_paths` is enforced once, as a trim, after the
    # loop -- see the trim below.
    while frontier and depth < max_hops:
        depth += 1
        layer = {"depth": depth, "frontier_size": len(frontier), "truncated": len(frontier) > _MAX_FRONTIER_SIZE}
        found_before = len(found)

        if len(frontier) > _MAX_FRONTIER_SIZE:
            print(
                f"WARNING: Agent 2 BFS depth {depth}: frontier of {len(frontier)} nodes exceeds "
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
                    # No max_paths check here -- see the note above the while loop and the
                    # module docstring. Cutting off here (the old behavior) meant a run with
                    # >= max_paths distinct depth-1 hits could exhaust the cap before ever
                    # reaching depth 2, silently discarding the exact deeper/more-specific
                    # candidates the goal-gene-expansion fix above exists to find.

                # Keep exploring past this node regardless of whether it was just recorded as a
                # goal gene -- a gene independently clearing Open Targets' disease2 threshold
                # doesn't mean a more specific mechanism doesn't sit further down the same real
                # chain (see the module docstring's "goal-gene expansion" note). Only `visited`
                # gates re-expansion now, not goal-gene membership.
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
        # Every gene that sits BETWEEN the endpoints of some goal-reaching path, before trimming. Agent 3 draws its
        # random-gene baseline from this pool (genes the search could have used as bridges), so the baseline has the
        # same topology bias (hub-like, well-connected genes) as the chains it is compared against.
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
        # Trim only now, once the full max_hops search is done -- see _trim_candidates for the order
        # (longest first; then fully-signed first; then discovery order; at most `max_per_start_gene`
        # per start gene). Rationale for longest-first: shallow, single-hop hits are exactly the kind the
        # plausibility bar treats most skeptically (a single unqualified interaction needs a specific named
        # mechanism to survive on its own, per _SELECTION_PROMPT_TEMPLATE), so letting them silently fill
        # every max_paths slot before a deeper, more specific mechanism is ever found defeats the point of
        # searching past a goal gene at all.
        print(
            f"WARNING: Agent 2 BFS found {len(found)} candidate path(s), more than "
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


def _trim_candidates(
    found: List[List[_RawEdge]], max_paths: int, max_per_start_gene: Optional[int] = None
) -> List[List[_RawEdge]]:
    # Input: all goal-reaching paths in discovery order, the slot count, an optional per-start-gene cap.
    # Output: at most `max_paths` of them, in rank order. Rank = longest first; among equal lengths a path
    # with no unknown-direction edge first; then discovery order (stable, so fully deterministic). With a
    # cap, a path is skipped while its start gene already holds `max_per_start_gene` slots; skipped paths
    # are used afterwards, in rank order, if slots remain -- so the cap re-balances the kept set but never
    # leaves a slot empty. Evidence-blind apart from the signed-edge tie-break (it never reads literature
    # or disease scores); that blindness is a stated limitation, not an oversight.
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


def _assert_selected_paths_are_code_found(selections, paths: List[List[_RawEdge]]) -> None:
    # Input: stage-1 selections [(path, rationale)] and the BFS's own path list. Raises
    # EdgeProvenanceError unless every selected path is, by identity, one of the BFS paths. The LLM only
    # ever supplies a candidate NUMBER (looked up in `paths`) and rationale text, so this holds by
    # construction; it is checked so a future refactor cannot silently break it.
    code_found = {id(p) for p in paths}
    for path, _ in selections:
        if id(path) not in code_found:
            raise EdgeProvenanceError("Agent 2: a selected chain is not one of the paths the BFS found.")


def _assert_chain_edges_come_from_bfs(chains, paths: List[List[_RawEdge]]) -> None:
    # Input: the final ChainCandidates and the BFS paths. Raises EdgeProvenanceError if any chain edge
    # (source, target, interaction type, edge id) is not an edge the BFS pulled from OmniPath/MetalinksDB.
    allowed = {(e.source_id, e.target_id, e.interaction_type, e.edge_id) for p in paths for e in p}
    for chain in chains:
        for e in chain.edges:
            if (e.source_gene, e.target_gene, e.interaction_type, e.omnipath_edge_id) not in allowed:
                raise EdgeProvenanceError(
                    f"Agent 2: chain edge {e.source_gene} -> {e.target_gene} ({e.interaction_type}) "
                    "was not found by the BFS in OmniPath/MetalinksDB."
                )


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


def _parse_debate_verdict(response: str) -> Optional[Tuple[int, str]]:
    # Input: a debate reply. Output: (0 for A / 1 for B, rationale text), or None if no clear
    # "WINNER: A" / "WINNER: B" line exists. ONLY the winner line is required: the rationale label is
    # matched loosely (RATIONALE:, and the misspelling "RATIONE:" that gemma4 emitted in a real run on
    # 2026-10-18, which crashed an entire Agent 2 run), and if no rationale label is present at all the
    # text after the winner line is used. The verdict itself is never guessed -- no winner line, no parse.
    winner_match = _DEBATE_WINNER_RE.search(response)
    if not winner_match:
        return None
    rationale_match = _DEBATE_RATIONALE_RE.search(response, winner_match.end())
    rationale = (rationale_match.group(1) if rationale_match else response[winner_match.end():]).strip()
    return (0 if winner_match.group(1).upper() == "A" else 1), (rationale or "(no rationale given)")


def _node_label(kind: str, node_id: str) -> str:
    # Input: a node's kind and id. Output: the string used to render it in the LLM prompt.
    return node_id if kind == "gene" else f"[metabolite] {node_id}"


_EDGE_STRENGTH_NOTE = (
    "Note on evidence: a step marked [OmniPath: N resources, curation effort C] shows how many databases "
    "independently report that interaction and how heavily it has been curated. A larger number means the "
    "INTERACTION itself is better established; it says nothing about whether the interaction matters for "
    "the two diseases. A step with only 1 resource is weakly established."
)


def _format_candidate(
    index: int, path: List[_RawEdge], citations: Dict[str, List[Citation]], show_edge_strength: bool = False
) -> str:
    # Input: candidate index, path edges, gene->citations dict, and whether to print each edge's OmniPath evidence
    # strength (added 2026-10-18, see run_agent2_chain's judge_evidence). Output: prompt text block for this candidate.
    def _step(e: _RawEdge) -> str:
        label = "UNKNOWN DIRECTION" if e.interaction_type == "unknown" else e.interaction_type
        text = f"{_node_label(e.source_kind, e.source_id)} --{label}--> {_node_label(e.target_kind, e.target_id)}"
        if show_edge_strength and e.n_sources is not None:
            effort = f", curation effort {e.curation_effort}" if e.curation_effort is not None else ""
            text += f" [OmniPath: {e.n_sources} resource{'s' if e.n_sources != 1 else ''}{effort}]"
        return text

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
            titles = "; ".join(f'"{c.title}"' for c in cites[:_CITATIONS_SHOWN_PER_GENE])
            lines.append(f"   {gene} literature: {titles}")
    return "\n".join(lines)


def _select_paths(
    paths: List[List[_RawEdge]],
    disease1: str,
    disease2: str,
    citations: Dict[str, List[Citation]],
    llm_client,
    stats: Optional[dict] = None,
    show_edge_strength: bool = False,
) -> Tuple[List[Tuple[List[_RawEdge], Optional[str]]], Optional[str]]:
    # Input: candidate paths, disease names, citations, LLM client. Optional: `stats` (a dict that receives
    # {"unjudged": n} -- how many candidates the reply gave no verdict line for; those are treated as rejected),
    # and show_edge_strength (print each edge's OmniPath evidence strength in the prompt).
    # Output: (selections, rejection_reason). selections is a list of (path, rationale) tuples the
    # LLM explicitly marked PLAUSIBLE -- zero, one, or many. rejection_reason is set only when
    # selections is empty AND the LLM gave a reason (its blanket "NONE" reason, or, if it instead
    # tagged every candidate individually, its NOT PLAUSIBLE reasons joined together). Skips the LLM
    # (returns the single path, rationale=None, reason=None) when there's nothing to adjudicate.
    # Callers must still run every returned tuple through _with_unknown_direction_caveat.
    #
    # This is "stage 1" of a two-stage selection process -- see module docstring's "Pairwise
    # debate ranking" note for stage 2 (_run_tournament), which compares these selections against
    # each other rather than judging each in isolation.
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
        # Diagnostics added 2026-10-18: a real run left 7-9 of 10 candidates without any verdict line and the saved
        # result could not show why. Keep how long the prompt was and (truncated) what the model actually replied.
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

    # Requires an explicit PLAUSIBLE/NOT PLAUSIBLE verdict per line -- a bare "<number>: <text>"
    # line (no verdict word) is NOT treated as an endorsement. This was a real bug: the model
    # would write a line for every candidate, including ones whose own sentence said the mechanism
    # "isn't specifically detailed" -- i.e. a rejection in substance -- and the old regex (which
    # accepted any "<number>: <anything>" line) endorsed all of them anyway, because it had no way
    # to tell a discussed-but-rejected candidate apart from an endorsed one. See
    # claude/agent2_bfs_and_prompt_recalibration.md.
    #
    # Number/verdict separator, widened 2026-09-14 (same day): the prompt asks for
    # "<number>: <verdict>", but a real run had the model write "<number>. <verdict>" instead
    # (an ordinary numbered-list period, not the requested colon) on every single line -- the
    # old regex required a literal ':' right after the digit, matched nothing at all, and the
    # whole response was (wrongly) treated as unparseable, raising a RuntimeError instead of
    # the correct "all candidates rejected" outcome. This model has now used two different
    # separators across two real runs with similar prompts, so the fix is tolerance in the
    # parser, not another prompt reword: '.', ':', and ')' are all accepted after the number.
    #
    # Markdown / "Candidate N" tolerance and de-duplication, added 2026-10-15: a reply formatted as
    # "**1: PLAUSIBLE** - ..." or "- Candidate 1: PLAUSIBLE ..." used to match nothing (and so raised);
    # asterisks/backticks, leading bullets and an optional "Candidate " prefix are now ignored. If the
    # model writes more than one verdict line for the same candidate number, the FIRST one wins -- before,
    # two "3: PLAUSIBLE" lines would have listed candidate 3 twice.
    _VERDICT_RE = re.compile(
        r"\s*(?:candidate\s+)?(\d+)\s*[:.\)]\s*(NOT\s+PLAUSIBLE|IMPLAUSIBLE|PLAUSIBLE)\b[\s:.\-]*(.*)",
        re.IGNORECASE
    )
    seen_indices: set = set()
    selections: List[Tuple[List[_RawEdge], Optional[str]]] = []
    not_plausible: List[Tuple[int, str]] = []

    def parse_reply(reply: str, allowed: set) -> None:
        # Reads verdict lines from `reply`, accepting only candidate numbers in `allowed` (0-based indices into
        # `paths`) that have no verdict yet; fills seen_indices / selections / not_plausible.
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
            f"Agent 2: LLM path-selection response had no parseable "
            f"'<number>: PLAUSIBLE/NOT PLAUSIBLE - <rationale>' lines and wasn't 'NONE' either: {response!r}"
        )

    # Added 2026-10-18: candidates the reply never addressed used to be silently treated as rejected, which in real
    # runs meant 7-9 of 10 candidates were dropped for a formatting reason, not a scientific one (the chain sets came
    # back empty). Ask once more about ONLY the unaddressed candidates, keeping their original numbers, so every
    # candidate gets a real verdict if the model is able to give one. Anything still unaddressed after the retry is
    # reported as `unjudged` (and is still treated as rejected -- never as endorsed).
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


def _select_paths_both_orders(
    paths: List[List[_RawEdge]],
    disease1: str,
    disease2: str,
    citations: Dict[str, List[Citation]],
    llm_client,
    passes: int = DEFAULT_STAGE1_PASSES,
    min_votes: int = DEFAULT_STAGE1_MIN_VOTES,
    control_pass: bool = False,
    show_edge_strength: bool = False,
):
    # Input: candidate paths and the usual stage-1 inputs, plus `passes` (1 = the original single judgement;
    # 2 = also judge the REVERSED candidate list), `min_votes` (passes that must endorse a candidate),
    # `control_pass` (with passes=2: judge the ORIGINAL order a second time, as a noise control) and
    # show_edge_strength (see _format_candidate).
    # Output: (selections, rejection_reason, votes, order_stats). `selections` are (path, rationale) in original
    # discovery order; `votes` maps id(path) -> number of passes that endorsed it; `order_stats` is saved into
    # bfs_stats. Why: an LLM reading a numbered list tends to favour candidates near the top, so one verdict per
    # candidate confounds "plausible" with "listed first". Judging the reversed list and recording where the two
    # verdicts differ measures that bias, and the union rule (min_votes=1) keeps recall. The control pass answers
    # the follow-up question "is a forward-vs-reversed difference an ORDER effect, or just sampling noise?": the same
    # forward list judged twice should agree if the model is stable. Its verdicts are recorded in the stats only and
    # NEVER enter `votes` or `selections`. Each pass gets the existing one-retry-on-unparseable behaviour. Selected
    # paths are matched back by object identity, never by the (order-dependent) candidate number, so the provenance
    # guard still applies.
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
    diagnostics: Dict[str, dict] = {}  # per pass: how often the first reply left candidates unaddressed + raw replies
    first_selections, first_reason = one_pass(paths, "forward")
    if len(paths) == 1 or passes <= 1:
        votes = {id(path): 1 for path, _ in first_selections}
        stats = {"stage1_passes": 1 if len(paths) > 1 else 0}
        if len(paths) > 1:
            stats["stage1_unjudged_forward"] = unjudged["forward"]
            stats["stage1_diagnostics"] = diagnostics
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
        "stage1_passes": 2,
        "stage1_min_votes": min_votes,
        "stage1_endorsed_forward": len(in_first),
        "stage1_endorsed_reversed": len(in_second),
        "stage1_endorsed_both": len(in_first & in_second),
        "stage1_order_disagreements": len(in_first ^ in_second),
        "stage1_candidates": len(paths),
        "stage1_unjudged_forward": unjudged["forward"],
        "stage1_unjudged_reversed": unjudged["reversed"],
    }
    if control_pass:
        repeat_selections, _ = one_pass(paths, "forward_repeat")
        in_repeat = {id(path) for path, _ in repeat_selections}
        stats.update({
            "stage1_control_pass": True,
            "stage1_endorsed_forward_repeat": len(in_repeat),
            # forward vs forward-again: any difference here is sampling noise, not order
            "stage1_noise_disagreements": len(in_first ^ in_repeat),
            "stage1_unjudged_forward_repeat": unjudged["forward_repeat"],
        })
    stats["stage1_diagnostics"] = diagnostics
    reason = None if kept else (first_reason or second_reason)
    return kept, reason, votes, stats


def _run_debate(
    candidate_a: Tuple[List[_RawEdge], Optional[str]],
    candidate_b: Tuple[List[_RawEdge], Optional[str]],
    disease1: str,
    disease2: str,
    citations: Dict[str, List[Citation]],
    llm_client,
    show_edge_strength: bool = False,
) -> Tuple[Optional[int], str]:
    # Input: two stage-1-selected (path, rationale) candidates, disease names, citations, LLM client.
    # Output: (0 if A won / 1 if B won / None if the reply had no parseable verdict twice, rationale text). Changed
    # 2026-10-18: this used to RAISE on an unparseable reply, which aborted a whole ~20-minute Agent 2 run over one
    # garbled debate (a real run, "RATIONE:"). Now a missing verdict is returned as None and the caller records the
    # debate as a draw flagged `unparsed`: "no verdict" is reported as exactly that, no winner is guessed, and the
    # unparseable reply (truncated) is kept in the rationale text. Contrast stage 1, where an unparseable reply still
    # raises because there is nothing sensible to fall back to.
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


def _update_elo(winner_rating: float, loser_rating: float, k: float = _ELO_K_FACTOR) -> Tuple[float, float]:
    # Input: the two candidates' current ratings (winner first). Output: their updated ratings
    # after one decisive win/loss (no draws -- every debate names exactly one winner). Standard
    # Elo update; see module docstring's "Pairwise debate ranking" note for the initial-rating
    # convention this borrows from Co-Scientist.
    expected_winner = 1.0 / (1.0 + 10 ** ((loser_rating - winner_rating) / 400.0))
    expected_loser = 1.0 - expected_winner
    new_winner = winner_rating + k * (1.0 - expected_winner)
    new_loser = loser_rating + k * (0.0 - expected_loser)
    return new_winner, new_loser


def _update_elo_draw(rating_a: float, rating_b: float, k: float = _ELO_K_FACTOR) -> Tuple[float, float]:
    # Input: two ratings. Output: both updated after a DRAW (each scores 0.5): the stronger side loses a little,
    # the weaker gains a little, equal ratings do not move. Used when the two A/B orders of a debate disagree.
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    return rating_a + k * (0.5 - expected_a), rating_b + k * (0.5 - (1.0 - expected_a))


def _round_robin_pairs(n: int) -> List[Tuple[int, int]]:
    # Input: candidate count. Output: every distinct pair (i, j), i<j -- a full round-robin.
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _single_elimination_bracket_pairs(n: int) -> List[Tuple[int, int]]:
    # Input: a round's remaining-candidate count. Output: that round's pairings only (0v1, 2v3,
    # ...) -- an odd one out is left unpaired here and given a bye by the caller. Later rounds
    # depend on who wins each pairing, so `_run_tournament` calls this fresh for every round
    # rather than precomputing a whole bracket up front.
    pairs = []
    i = 0
    while i + 1 < n:
        pairs.append((i, i + 1))
        i += 2
    return pairs


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
    # Input: the selections list and two positions in it. Output: (winner_idx or None, loser_idx or None,
    # rationale, swapped_rationale, draw, unparsed). Candidate i is shown as "A" in the first run and as "B" in the
    # swapped run. With swap_order a win counts only if the same candidate wins both times, else it is a draw
    # (winner/loser None). This removes "always pick A"/"always pick B" from the ranking and measures how often the
    # model's verdict depends on presentation order alone. A run with no parseable verdict makes the debate a draw
    # with unparsed=True (see _run_debate).
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


def _run_tournament(
    selections: List[Tuple[List[_RawEdge], Optional[str]]],
    disease1: str,
    disease2: str,
    citations: Dict[str, List[Citation]],
    llm_client,
    swap_order: bool = True,
    show_edge_strength: bool = False,
) -> Tuple[List[float], List[DebateComparison]]:
    # Input: stage-1-selected candidates (path, rationale), disease names, citations, LLM client.
    # Output: (elo_ratings parallel to `selections`, full debate log). Every debate's winner and
    # rationale is kept, not just the final ranking -- see module docstring. A full round-robin is
    # used when there are few enough candidates to keep cost bounded (n <= _TOURNAMENT_BRACKET_THRESHOLD),
    # else a single-elimination bracket. A single candidate skips the tournament entirely (nothing
    # to compare) and keeps the initial rating. `winner_index`/`loser_index` in the returned
    # DebateComparisons refer to positions in `selections` (the caller re-maps these to the final,
    # Elo-sorted `chains` order -- see run_agent2_chain).
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

    # Single-elimination bracket for larger fields -- bounds cost to ~n debates total instead of
    # ~n^2/2, at the real cost of not comparing every pair. See module docstring.
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
            curation_effort=edge.curation_effort,
            n_sources=edge.n_sources,
        )
        for edge in path
    ]


def _build_evidence(
    path: List[_RawEdge],
    disease1_scores: Dict[str, float],
    disease2_scores: Dict[str, float],
    citations: Dict[str, List[Citation]],
) -> dict:
    # Input: one code-found path, both disease score maps, the gene->citations dict. Output: a plain dict of
    # facts that come straight from the databases and from what the model was actually shown -- nothing the LLM
    # wrote. This (not the LLM rationale) is the grounded record of why a chain exists: each edge's curated
    # direction and OmniPath evidence strength, each gene's Open Targets scores, and the literature titles
    # that were in the prompt for each gene (at most _CITATIONS_SHOWN_PER_GENE, exactly as `_format_candidate`
    # shows them).
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


def _off_path_genes(rationale: Optional[str], path: List[_RawEdge], universe: Set[str]) -> List[str]:
    # Input: an LLM rationale, the chain it describes, the set of gene symbols the search saw (plus Agent 1's
    # lists and the cited genes). Output: sorted gene symbols that appear in the text but are NOT on this chain
    # (disease abbreviations excluded). Such mentions are claims about biology the chain's data do not contain;
    # they are listed so a reader or a later check can verify them, not used to reject the chain.
    if not rationale:
        return []
    on_path = {e.source_id for e in path} | {e.target_id for e in path}
    mentioned = {tok for tok in _GENE_TOKEN.findall(rationale) if tok in universe and tok not in DISEASE_ABBREVIATIONS}
    return sorted(mentioned - on_path)


def _fetch_bridge_citations(paths, citations, disease1, disease2, pubmed_client, stats) -> Dict[str, List[Citation]]:
    # Input: the candidate paths, Agent 1's gene->citations dict, both disease names, a PubMed client, and the
    # bfs_stats dict. Output: a COPY of the citations dict extended with up to one paper per disease (so at most 2)
    # for every gene on a candidate path that Agent 1 gave no literature for -- in practice the bridge genes in the
    # middle of chains, which previously reached the judge with nothing but the model's own memory. Same gene-AND-
    # disease search Agent 1 uses (titles, PubMed's default order). Deliberately NOT an edge co-mention search
    # ("geneA AND geneB"): that is exactly Agent 3's independent edge-literature check, and showing it to the Agent 2
    # judge would make Agent 3's confirmation circular. A failed search is skipped and counted, never fatal.
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


def run_agent2_chain(
    agent1_result: Agent1DiscoveryResult,
    omnipath_client: Optional[object] = None,
    metalinks_client: Optional[object] = None,
    llm_client: Optional[object] = None,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_candidate_paths: int = DEFAULT_MAX_CANDIDATE_PATHS,
    enable_debate_tournament: bool = True,
    max_per_start_gene: Optional[int] = DEFAULT_MAX_PER_START_GENE,
    stage1_passes: int = DEFAULT_STAGE1_PASSES,
    stage1_min_votes: int = DEFAULT_STAGE1_MIN_VOTES,
    swap_debate_order: bool = True,
    stage1_control_pass: bool = False,
    judge_evidence: bool = False,
    pubmed_client: Optional[object] = None,
) -> Agent2ChainResult:
    # Input: Agent 1's result, optional clients (omnipath_client defaults to real OmniPathClient;
    # metalinks_client defaults to None, meaning protein-only search -- pass a real MetalinksClient()
    # or StubMetalinksClient to include gene<->metabolite hops; llm_client defaults to real LLMClient;
    # pass Stub* clients in tests), hop/candidate limits, and whether to run the stage-2 pairwise
    # debate tournament (see module docstring's "Pairwise debate ranking" note -- on by default,
    # but multiplies LLM calls; set False to skip it and keep only stage-1 independent selection).
    # 2026-10-18 options: max_per_start_gene (trim diversity cap, None/0 = off), stage1_passes (2 = also judge the
    # reversed candidate order; 1 = single pass), stage1_min_votes, swap_debate_order (run each debate in both
    # A/B orders; disagreement = draw). Defaults = the hardened behaviour; pass 1/1/False for the old one.
    # Also added 2026-10-18: stage1_control_pass (judge the original order a second time and record how often the two
    # forward verdicts differ -- separates sampling noise from order bias; does not affect which chains are kept) and
    # judge_evidence (show the judge each edge's OmniPath evidence strength and, when pubmed_client is given, one
    # PubMed title per disease for every path gene Agent 1 gave no literature; see _fetch_bridge_citations). All three
    # default to OFF here so the function behaves as before for existing callers; the CLI and scripts/run_all turn them on.
    # Output: Agent2ChainResult. Raises NoChainFoundError if BFS finds nothing within max_hops.
    omnipath_client = omnipath_client or OmniPathClient()

    disease1_scores, disease2_scores = _score_maps(agent1_result)
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
            f"Agent 2: no directed path found between any {agent1_result.disease1}-associated "
            f"protein and any {agent1_result.disease2}-associated protein, within {max_hops} hops, "
            f"starting from {len(disease1_genes)} disease1 genes and {len(disease2_genes)} disease2 genes "
            f"Agent 1 surfaced (search could pass through other genes OmniPath connects them to, and "
            f"metabolites if a MetalinksDB client was provided)."
        )

    llm_client = llm_client or LLMClient()
    judge_citations = agent1_result.citations
    bfs_stats["judge_evidence"] = "none"
    if judge_evidence:
        bfs_stats["judge_evidence"] = "omnipath_edge_strength"
        if pubmed_client is not None:
            judge_citations = _fetch_bridge_citations(
                paths, agent1_result.citations, agent1_result.disease1, agent1_result.disease2, pubmed_client, bfs_stats
            )
            bfs_stats["judge_evidence"] += "+bridge_gene_citations"
    selections, rejection_reason, stage1_votes, order_stats = _select_paths_both_orders(
        paths, agent1_result.disease1, agent1_result.disease2, judge_citations, llm_client,
        passes=stage1_passes, min_votes=stage1_min_votes, control_pass=stage1_control_pass,
        show_edge_strength=judge_evidence,
    )
    bfs_stats.update(order_stats)
    _assert_selected_paths_are_code_found(selections, paths)
    bfs_stats["n_stage1_endorsed"] = len(selections)
    bfs_stats["llm_model"] = getattr(llm_client, "model", None)

    ratings = [None] * len(selections)
    debate_log: List[DebateComparison] = []
    if selections and enable_debate_tournament and len(selections) > 1:
        ratings, raw_debate_log = _run_tournament(
            selections, agent1_result.disease1, agent1_result.disease2, judge_citations, llm_client,
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

    # Sort by Elo rating (highest first) when the tournament ran; stable, so ties (and the
    # untouched case where the tournament never ran, ratings all None) keep original discovery
    # order -- matches existing tests that assume selection order when no tournament runs.
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
            stage1_votes=stage1_votes.get(id(path)) if len(paths) > 1 else None,
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

    return Agent2ChainResult(
        disease1=agent1_result.disease1,
        disease2=agent1_result.disease2,
        chains=chains,
        candidate_path_count=len(paths),
        shared_genes=shared_genes,
        rejection_reason=rejection_reason if not chains else None,
        debate_log=debate_log,
        bfs_stats=bfs_stats,
    )


def _cli():
    parser = argparse.ArgumentParser(description="Run Agent 1 then Agent 2 (Chain Construction) for one disease pair.")
    parser.add_argument("--pair", choices=[p["id"] for p in DISEASE_PAIRS], help="Disease pair id from common/config.py")
    parser.add_argument("--disease1", help="Override: disease 1 name")
    parser.add_argument("--disease2", help="Override: disease 2 name")
    parser.add_argument("--min-score", type=float, default=0.1, help="Minimum Open Targets association score, passed to Agent 1")
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS, help="Max PPI/metabolite hops to search")
    parser.add_argument(
        "--with-metabolites", action="store_true", help="Opt in to MetalinksDB gene<->metabolite hops (default: protein-only)"
    )
    parser.add_argument("--no-metabolites", action="store_true", help="Deprecated no-op: protein-only is now the default")
    parser.add_argument(
        "--no-debate-tournament",
        action="store_true",
        help="Skip stage 2's pairwise debate tournament -- keep only stage 1's independent PLAUSIBLE/NOT PLAUSIBLE filter (see module docstring)",
    )
    parser.add_argument("--max-per-start-gene", type=int, default=DEFAULT_MAX_PER_START_GENE,
                        help="Trim diversity cap: most kept candidates that may share one start gene (0 = no cap)")
    parser.add_argument("--stage1-passes", type=int, choices=[1, 2], default=DEFAULT_STAGE1_PASSES,
                        help="2 = also judge candidates in reversed order (measures order bias); 1 = single judgement")
    parser.add_argument("--no-swap-debates", action="store_true",
                        help="Run each stage-2 debate once instead of in both A/B orders")
    parser.add_argument("--no-stage1-control", action="store_true",
                        help="Skip the extra forward-order stage-1 pass that measures sampling noise")
    parser.add_argument("--no-judge-evidence", action="store_true",
                        help="Do not show the judge OmniPath edge strength or bridge-gene PubMed titles")
    parser.add_argument(
        "--omnipath-datasets",
        default=None,
        help=(
            "OmniPath dataset scope. Omit for the default core 'omnipath' set. Pass 'extended' "
            f"as shorthand for '{EXTENDED_DATASETS}' (adds pathwayextra + kinaseextra, still-"
            "curated but broader coverage), or a raw comma-separated dataset list. See "
            "omnipath_client.py's module docstring and claude/agent2_bfs_and_prompt_recalibration.md."
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

    print(f"Running Agent 1 (Discovery) for {disease1} / {disease2}...", flush=True)
    agent1_result = run_agent1_discovery(disease1, disease2, min_score=args.min_score)

    print("Running Agent 2 (Chain Construction)...", flush=True)
    metalinks_client = MetalinksClient() if args.with_metabolites else None
    if args.omnipath_datasets == "extended":
        omnipath_client = OmniPathClient(datasets=EXTENDED_DATASETS)
    elif args.omnipath_datasets:
        omnipath_client = OmniPathClient(datasets=args.omnipath_datasets)
    else:
        omnipath_client = None  # OmniPathClient()'s own default (the core "omnipath" set)
    agent2_result = run_agent2_chain(
        agent1_result,
        omnipath_client=omnipath_client,
        metalinks_client=metalinks_client,
        max_hops=args.max_hops,
        enable_debate_tournament=not args.no_debate_tournament,
        max_per_start_gene=args.max_per_start_gene,
        stage1_passes=args.stage1_passes,
        swap_debate_order=not args.no_swap_debates,
        stage1_control_pass=not args.no_stage1_control,
        judge_evidence=not args.no_judge_evidence,
        pubmed_client=None if args.no_judge_evidence else PubMedClient(),
    )

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
