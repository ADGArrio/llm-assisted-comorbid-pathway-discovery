# Pipeline B -- Multi-Agent

Per `claude/project_architecture.md` and the published diagram:
https://claude.ai/code/artifact/c8a6f509-478d-4a9f-a285-c7cac49ed984

## Built so far

**Agent 1 -- Discovery** (`agent1_discovery.py`, LangGraph)
- `opentargets_client.py` -- sub-agent 1: Open Targets GraphQL API. Free-text
  disease name -> EFO id (`search_disease`), then EFO id -> associated
  genes/proteins above a configurable score threshold
  (`get_associated_targets`): gene symbol, Ensembl id, UniProt id, score.
- `pubmed_client.py` -- sub-agent 2: PubMed E-utilities. Citation/evidence
  support only for genes sub-agent 1 already surfaced -- **not** a second
  discovery source (Open Targets' own score already folds in a literature
  evidence type; re-deriving candidates from PubMed would be redundant).
- The LangGraph wiring is fan-out/fan-in, not two fully independent
  parallel sub-agents: `discover_disease1` and `discover_disease2` run
  concurrently (that's the real parallelism), but `cite_genes` genuinely
  depends on both finishing, since it needs their gene lists. See the
  module docstring in `agent1_discovery.py` for the full reasoning --
  small correction to how the original architecture sketch grouped these.

**Confirmed working against the live APIs, run by the user on their own
machine (2026-09-05)**, for `t2dm_ad`: Open Targets resolved both diseases
(`MONDO_0005148` = T2DM, `MONDO_0004975` = Alzheimer's -- the schema field
is named `efoId` but Open Targets' ontology space spans MONDO ids too),
returned biologically sound gene sets (T2DM: KCNJ11, ABCC8, GCK, PPARG,
HNF1A/4A/1B, TCF7L2, INSR, IRS1/2 -- the actual MODY/T2DM GWAS canon;
AD: APP, PSEN1/2, APOE, TREM2, CLU, PICALM, BIN1, CR1 -- the standard AD
GWAS loci), and PubMed citations came back real and on-topic. Neither the
cloud container nor the device-bridge VM this was developed in can reach
these APIs directly (network allowlist), so this had to be verified on
the user's own machine -- same pattern as debugging Pipeline A's Ollama
integration.

Two things worth noting from that real run, relevant to Agent 2 and
BioPathBench later:
- **Real overlap genes between the two disease's target lists**: APOE,
  ACE, HMGCR, JAZF1, CDKN2B, TMEM106B, plus a cluster of NDUFA/NDUFB/NDUFS/
  NDUFV mitochondrial Complex I subunits. These are exactly the kind of
  shared nodes Agent 2 needs as anchors to build an OmniPath path from
  disease1 to disease2 -- concrete evidence the T2DM/AD shared-mechanism
  premise has real molecular grounding, not just a plausible-sounding
  hypothesis.
- **The NDUF* cluster shares near-identical scores** (many at exactly
  0.4619748183558772, for example). That pattern usually means Open
  Targets scored them from one pathway/complex-level evidence source
  rather than gene-specific evidence -- not a bug, but a sign that at low
  `min_score` thresholds the target list gets diluted with generic
  "mitochondrial housekeeping" genes that are less mechanism-specific than
  they look. Worth considering a higher `min_score` or a pathway-level
  dedup once Agent 2 actually consumes this list.
- **GSK3B showed up for real** (AD score 0.554) -- matching what Pipeline
  A's ungrounded LLM already guessed at ("GSK3$\beta$") in its T2DM->AD
  narrative. A good concrete data point for the eventual Pipeline A vs. B
  comparison writeup: the orphaned LLM landed on a real gene by parametric
  knowledge alone, in this one case.

```
python -m PipelineB.agent1_discovery --pair t2dm_ad
python -m PipelineB.agent1_discovery --disease1 "X" --disease2 "Y" --min-score 0.2
```

**Agent 2 -- Chain Construction** (`agent2_chain.py`, `omnipath_client.py`)
- `omnipath_client.py` queries OmniPath's REST API for directed
  activates/inhibits protein interactions. `get_outgoing_interactions(genes)`
  returns edges FROM those genes to wherever they actually lead -- including
  genes outside the queried set -- which is what lets the search discover a
  real intermediate/bridge protein that isn't itself associated with either
  disease.
- `agent2_chain.py` runs a multi-source BFS: starting from every
  disease1-associated gene simultaneously, it fetches each depth's real
  outgoing edges live from OmniPath (up to `--max-hops`, default **5**)
  until it reaches any disease2-associated gene. It collects every shortest
  path found (not just the first), then:
  - if there's exactly one shortest path, that's the chain -- no LLM involved
    for the plausibility call, though see the unknown-direction caveat below,
    which still applies;
  - if several equally-short paths exist, a grounded LLM call (reusing
    `common/llm_client.py`) judges each one, using whatever PubMed citations
    Agent 1 already gathered for that path's genes as evidence where
    available (falling back to its own domain knowledge otherwise), and
    returns **every candidate it finds biologically plausible -- zero, one,
    or several**, each with its own one-sentence rationale. It never
    generates edges, protein ids, or `omnipath_edge_id`s itself -- it only
    ever chooses among the real BFS-discovered candidates. See the module
    docstring for why: letting an LLM "deduce" a path over a text-dumped
    graph, as the original architecture sketch suggested, invites exactly
    the kind of hallucinated structure Pipeline A exists to contrast against.
  - if no path exists within `--max-hops`, it raises `NoChainFoundError`.
    This is different from the LLM judging zero of the found candidates
    plausible (`chains: []` on an otherwise successful result) -- the error
    means BFS found no structural path at all; the empty list means paths
    existed but none held up. Both are legitimate, distinguishable outcomes.
- Output is a list of chain candidates (`Agent2ChainResult.chains`), each
  with its own `edges`, `rationale`, and a derived `has_unconfirmed_direction`
  boolean -- not a single winning chain -- so that a genuinely ambiguous
  mechanism (e.g. two independently plausible routes, each with its own
  supporting literature) doesn't get artificially collapsed into one answer.
- Edges follow project_architecture.md's schema (source_gene, source_protein,
  target_gene, target_protein, interaction_type, omnipath_edge_id,
  source_disease_assoc_score, target_disease_assoc_score), with the score
  fields read literally off the schema's own worked example:
  `source_disease_assoc_score` is the edge's source gene's score against
  disease1, `target_disease_assoc_score` is the target gene's score against
  disease2 -- either can be `None` for a middle hop that isn't itself
  disease-associated in that direction.
- `_MAX_FRONTIER_SIZE` (300 genes) caps each BFS layer's query and prints a
  warning when it truncates -- trading a small chance of missing a very
  indirect shortest path for keeping requests bounded. Not yet observed in
  practice.
- 49/49 tests pass, all against `StubOmniPathClient`/`StubLLMClient` --
  including a test that caught a real bug during development: an earlier
  version restricted every OmniPath edge to both ends being in Agent 1's
  gene lists, which silently made genuine multi-hop bridging through a
  non-disease-associated intermediate impossible. Fixed by switching to
  live, per-layer outgoing-edge queries instead of one restricted upfront
  fetch.

**Real findings from the max_hops=10 run, and what changed because of
them (2026-09-05)** -- see `claude/agent2_findings_maxhops10_run.md` in the
Claude Project for the full write-up, citations, and thesis-ready framing;
summary here:

Run against the live OmniPath + Ollama/gemma4 stack for `t2dm_ad` at
`--max-hops 10` returned 10 structural candidate paths, and the LLM
endorsed **all 10 of them (100% acceptance)** -- a strong signal the
original selection prompt wasn't discriminating at all, just rubber-stamping
whatever BFS handed it. Auditing the 10 rationales by hand also surfaced a
real factual error in one of them (an endorsed candidate asserted "NR1H2
(also known as RXR)" -- NR1H2 is actually Liver X Receptor beta/LXRbeta,
not RXR, per GeneCards and OMIM), and several endorsed candidates narrated
confident activate/inhibit language for edges OmniPath itself records as
`interaction_type: "unknown"` (no curated direction flag at all).

Framing this honestly rather than as a pure failure: the goal of this
project was never a plausibility judge that's 100% correct -- it's a
grounded pipeline that is **considerably better than an ungrounded/orphan
LLM baseline** (Pipeline A), and can be honest about where it still falls
short. These findings are exactly the kind of concrete, quantifiable
comparison point Pipeline A vs. B needs, and are being kept for the thesis
and defense rather than quietly patched away.

Two concrete changes came out of this, on top of dropping the default hop
count from 10 back to 5 (see `DEFAULT_MAX_HOPS` history in the module
docstring):

1. **Tightened the "plausible" bar.** The selection prompt now requires a
   *specific, named mechanism* for every step (e.g. "X phosphorylates Y,
   activating Y's kinase activity") rather than accepting "these proteins
   are known to interact," and explicitly tells the model to hedge rather
   than assert when it isn't confident about a gene's identity, function,
   or alias -- a direct response to the NR1H2/RXR error. This is a
   prompt-level change only -- it cannot guarantee the LLM actually applies
   the stricter bar, and the 100%-acceptance finding above should be
   re-checked against a fresh real run once one is available.
2. **`interaction_type: "unknown"` edges now always carry a caveat.** Per
   explicit instruction, a missing OmniPath activate/inhibit flag is
   treated as a signal that the causal direction isn't established, not
   as license to narrate confident directionality. Outright rejecting any
   candidate touching an "unknown" edge was considered and deliberately
   *not* chosen as the default -- OmniPath's curation coverage is uneven,
   and doing that would bias the pipeline against real biology in
   less-curated corners of the interactome for a curation reason, not an
   implausibility reason. Instead, `_with_unknown_direction_caveat` is a
   non-negotiable, code-level backstop that appends a fixed caveat sentence
   to the rationale of *any* returned chain touching an "unknown" edge --
   including the single-candidate case, which bypasses the LLM entirely
   and previously would have shipped a bare `rationale: None` with no hint
   that the direction wasn't confirmed. `ChainCandidate.to_dict()` also
   exposes this as a queryable `has_unconfirmed_direction` boolean for
   BioPathBench or the thesis write-up to use directly.

```
python -m PipelineB.agent2_chain --pair t2dm_ad
```

Worth re-running for real again with these changes in and reporting back:
does the acceptance rate come down from 100%, do any endorsed candidates
still assert an unverified gene alias, and does every `unknown`-direction
candidate now carry a visible caveat.


**Agent 3 -- Independent Confirmation** (`agent3_confirmation.py`, `tissue_client.py`, `crossdb_client.py`)

Per project_architecture.md, this checks each chain candidate against evidence
Agents 1-2 never looked at -- what keeps validation non-circular rather than
just re-confirming Open Targets/OmniPath's own say-so. Two independent checks,
run per chain candidate returned by Agent 2:

- **Tissue plausibility** (`tissue_client.py`): is a gene actually expressed
  in a tissue relevant to the disease it's nearest to in the chain? GTEx is
  the primary source (accepts a bare gene symbol, no id resolution needed);
  Human Protein Atlas is a fallback used only when GTEx has no record *and*
  an Ensembl id is known for that gene (not every gene has one -- a pure BFS
  bridge gene from OmniPath that Open Targets never surfaced has no Ensembl
  id in this pipeline at all). A gene counts as "expressed" at >=1.0 nTPM,
  a common low-expression cutoff -- see the module docstring for why this is
  a documented simplification, not an exact biological threshold.
- **Cross-database gene-disease association** (`crossdb_client.py`): for the
  chain's two disease-anchored endpoint genes specifically (the first gene,
  claimed disease1-associated by Agent 1, and the last gene, claimed
  disease2-associated), does GWAS Catalog and/or DisGeNET -- association
  sources independent of Open Targets -- also confirm gene<->disease? This
  is the committee's Metric 1 (committee_feedback_plan.md), generalized from
  "one PPI edge" to "the two ends of a possibly-longer chain." A middle
  bridge gene has no such check: Agent 1 never claimed it was
  disease-associated in the first place, so there's nothing to
  cross-confirm -- it only gets the tissue check, against *both* diseases'
  tissues (a bridge gene isn't assigned to one side of the mechanism).
- Like Agent 2, this agent doesn't collapse everything into one pass/fail
  "confirmed" boolean. A gene not expressed in GTEx, or with no GWAS
  Catalog/DisGeNET hit, is real negative evidence worth reporting on its own
  terms -- a gene can matter at the protein level (trafficking, complex
  formation) without showing up as disease-associated in a genetics-only
  database. `Agent3ConfirmationResult` reports every tissue check and both
  endpoint cross-db checks individually, plus simple counts
  (`tissue_expressed_count`/`tissue_checked_count`) -- no single collapsed
  verdict.
- **On DisGeNET specifically**: during setup, a URL was suggested for
  connecting to DisGeNET as an MCP server (`https://mcp.disgenet.com/mcp`).
  That domain doesn't match DisGeNET's actual documented API host
  (`api.disgenet.com`, confirmed via disgenet.com/Tools and real working
  client code in the ToolUniverse project), doesn't appear in Anthropic's
  MCP connector registry, and there's no tool available to add an arbitrary
  remote MCP server by URL regardless -- so `crossdb_client.py` is built
  directly against the verified real DisGeNET REST API instead. Put your
  real academic-license API key in `.env` as `DISGENET_API_KEY`; DisGeNET is
  entirely optional (`disgenet_client=None` / omit `--use-disgenet`) --
  GWAS Catalog alone still runs the cross-db check.
- GWAS Catalog's exact endpoint path could not be confirmed against a live
  docs fetch in this environment (JS-rendered reference page); built against
  the well-established v1 HAL finder-method pattern that other real
  integrations use, with a clear error naming the docs URL to check if it
  404s on a real run. GTEx and HPA's exact response field names are
  similarly unconfirmed from a live call and parsed defensively (see each
  module's docstring) -- same "strong starting point, not guaranteed final"
  posture already applied to opentargets_client.py and gap1_pull.py before
  either had a live check.

```
python -m PipelineB.agent3_confirmation --pair t2dm_ad
python -m PipelineB.agent3_confirmation --pair t2dm_ad --use-disgenet
```

Not yet run for real -- try it and report back what comes out, especially
whether GTEx/HPA/GWAS Catalog's actual response shapes match what the
defensive parsing above expects.

**Agent 4 -- MIMIC-IV Cohort Check** (`agent4_mimic.py`, `mimic_client.py`)

Per project_architecture.md, this validates the disease *pair* as a whole
(do T2DM-diagnosed patients really show elevated real-world co-occurrence of
AD or CAD), not any individual chain -- MIMIC-IV's diagnosis codes can't
confirm a specific protein-protein edge. Runs independently of Agents 2-3;
only needs a disease pair, not a chain.

- Reads MIMIC-IV's `diagnoses_icd.csv.gz` locally (PhysioNet-credentialed
  data, queried locally per project_architecture.md, not a live API) in
  500k-row chunks, matching ICD-10 codes against the hand-mapped prefixes
  in `common/config.py`'s `DISEASE_ICD10_PREFIXES` (T2DM: E11, AD: G30, CAD:
  I25 -- chronic ischemic heart disease specifically, not the full I20-I25
  ischemic heart disease block; worth revisiting if cohort counts are too
  small).
- Computes a relative risk: (rate of disease2 among disease1 patients) /
  (disease2's base rate in the whole population). **This is a crude,
  unadjusted ratio** -- no age/sex matching, no confounder adjustment -- and
  should be reported as descriptive evidence for the comorbidity premise,
  not a causal claim. Worth saying this plainly wherever the number is used,
  same honesty standard as Agent 2's findings doc holds its own results to.
- **Only needs two files, not the full MIMIC-IV download**:
  `hosp/diagnoses_icd.csv.gz` and `hosp/d_icd_diagnoses.csv.gz`, set as
  `MIMIC_DATA_DIR` in `.env` (the folder containing `hosp/`). A full
  recursive PhysioNet mirror download is known to take many hours for
  reasons unrelated to bandwidth (it fetches directory index pages for
  hundreds of thousands of files before reaching real data -- see
  https://github.com/MIT-LCP/mimic-code/issues/600); downloading just these
  two files directly by their individual PhysioNet URLs should take minutes.
  If your download is already stuck, there's no need to let it finish --
  cancel it and grab just these two files instead.
- 82/82 tests pass overall, including a real (small, synthetic) gzipped CSV
  fixture exercising `MimicClient`'s actual chunked-read logic, not just a
  stub -- the only place in this codebase so far that tests a real file
  format end to end without a network call.

```
python -m PipelineB.agent4_mimic --pair t2dm_ad
```

Not yet run for real -- needs `MIMIC_DATA_DIR` pointing at real MIMIC-IV
files on your machine first.

## Not started yet

- `synthesis.py` -- grounded LLM narrative generation
- `pipeline_b.py` -- orchestrates Agents 1-4 + Synthesis end to end

Will reuse `common/config.py` and `common/llm_client.py`, same as Pipeline A.
