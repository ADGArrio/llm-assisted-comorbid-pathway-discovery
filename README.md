# LLM-Assisted Comorbid Mechanistic Pathway Discovery

MS thesis project (Arrio Gonsalves, CMU, advisor Dr. Min Xu). Compares two
approaches to deducing the mechanistic pathway connecting a pair of
comorbid diseases -- Type 2 Diabetes Mellitus paired with Alzheimer's
Disease, and with Coronary Artery Disease:

- **Pipeline A** -- an orphaned/ungrounded LLM baseline: one prompt, the
  model's own parametric knowledge only, no retrieval.
- **Pipeline B** -- a grounded, four-agent pipeline that builds its answer
  from real biomedical data sources and independently checks it before
  reporting it.

Both pipelines run standalone or side by side via `main.py`. Scoring one
against the other (the planned **BioPathBench** benchmark) is deferred for
now -- see `claude/metrics_purge.md` -- so this repo currently just runs
both pipelines cleanly and saves their raw output. Full architecture spec
lives in the project's `claude/project_architecture.md` doc.

## Layout

```
Code/
  common/       -- shared by both pipelines: disease-pair config, LLM client
  PipelineA/    -- Orphaned LLM baseline
  PipelineB/    -- Grounded multi-agent pipeline (Agents 1-4)
  docs/figures/ -- diagrams
  main.py       -- run both pipelines cell-by-cell in VS Code's Interactive Window
  requirements.txt
  pytest.ini
```

## What's here

**Pipeline A** (`PipelineA/pipeline_a.py`) -- a single LLM call asked to
narrate a mechanistic chain between two diseases from its own knowledge,
returning the raw narrative text.

**Pipeline B** (`PipelineB/`) -- four agents, each independent enough to
run and test on its own:

- **Agent 1, Discovery** (`agent1_discovery.py`) -- Open Targets resolves
  each disease to its gene associations; PubMed supplies citation support
  for the genes found. Orchestrated with LangGraph (`discover_disease1`/
  `discover_disease2` run concurrently, `cite_genes` joins after both).
- **Agent 2, Chain Construction** (`agent2_chain.py`) -- BFS over live
  OmniPath protein-protein interactions and MetalinksDB gene<->metabolite
  interactions, finding every shortest real path from a disease1-associated
  node to a disease2-associated gene (metabolites can be intermediate hops;
  chain endpoints are always genes). When multiple equally-short candidates
  exist, an LLM judges which are biologically plausible using real
  citations as evidence -- it never invents an edge itself.
- **Agent 3, Independent Confirmation** (`agent3_confirmation.py`) --
  checks each chain against evidence Agents 1-2 never used: tissue
  expression (GTEx) and cross-database gene-disease association (GWAS
  Catalog -- real numerical study-level evidence, kept over DisGeNET
  since Agent 1's PubMed citations already cover literature evidence).
  Reports each check individually rather than collapsing to one verdict.
- **Agent 4, MIMIC-IV Cohort Check** (`agent4_mimic.py`) -- queries local
  MIMIC-IV diagnosis data for the real-world co-occurrence rate of the two
  diagnoses, as descriptive (not causal) supporting evidence for the
  comorbidity itself.

**`main.py`** -- runs both pipelines as `# %%` cells in VS Code's Python
Interactive Window: Pipeline A, each Pipeline B agent individually, all of
Pipeline B combined, and a side-by-side view of both pipelines' output.
Every run is saved as JSON under `outputs/`.

See `docs/figures/pipeline_b_diagram.png` for the Pipeline B architecture
diagram (inputs, the four-agent flow, and outputs).

## Setup

```
cd Code
python3 -m venv "MS Thesis"
source "MS Thesis/bin/activate"   # Windows: "MS Thesis\Scripts\activate"
pip install -r requirements.txt
cp .env.example .env   # then edit for your LLM endpoint, MIMIC_DATA_DIR, etc.
```

Re-run `source "MS Thesis/bin/activate"` at the start of every new
terminal session before running anything below -- the env folder is
gitignored, so it stays local to your machine and isn't part of the repo.

Pipeline A and Agent 2 need a running LLM: defaults to a local Ollama
server (`ollama serve`) with a model matching `LLM_MODEL` in `.env`, or
point `LLM_BASE_URL`/`LLM_MODEL`/`LLM_API_KEY` at any OpenAI-compatible
hosted endpoint instead.

## Run

The easiest way to run and compare both pipelines is `main.py` in VS
Code's Python Interactive Window -- open the file and run each `# %%`
cell in order (Shift+Enter). It saves every result to `outputs/` and ends
with a side-by-side view of Pipeline A's and Pipeline B's output.

Each piece is also runnable standalone from the command line:

```
python -m PipelineA.pipeline_a --pair t2dm_ad
python -m PipelineB.agent1_discovery --pair t2dm_ad
python -m PipelineB.agent2_chain --pair t2dm_ad
python -m PipelineB.agent3_confirmation --pair t2dm_ad
python -m PipelineB.agent4_mimic --pair t2dm_ad
```

`--pair` accepts `t2dm_ad` or `t2dm_cad` (see `common/config.py`);
`--disease1`/`--disease2` overrides work too. Agent 2/3 need real network
access to OmniPath/GTEx/GWAS Catalog/etc.; Agent 4 needs `MIMIC_DATA_DIR`
pointing at local MIMIC-IV files instead. See `PipelineB/README.md` for
per-agent details and real-run findings.

## Test

```
pytest -v
```

Everything runs offline against stub clients (`StubLLMClient`,
`StubOpenTargetsClient`, `StubOmniPathClient`, `StubMetalinksClient`,
`StubGTExClient`, `StubGWASCatalogClient`, `StubMimicClient`) -- no model
or network needed.
