# Comorbidity Pathway Pipeline

Code for the MS thesis pipeline. Full architecture spec: see the project's
`claude/project_architecture.md` doc, or the published diagram at
https://claude.ai/code/artifact/c8a6f509-478d-4a9f-a285-c7cac49ed984

## Layout

```
Code/
  common/       -- shared by both pipelines: disease-pair config, LLM client
  PipelineA/    -- Orphaned LLM baseline (built, tested)
  PipelineB/    -- Multi-agent pipeline: Agents 1-4 built, Synthesis not started
  metrics/      -- BioPathBench and its building blocks (gene/protein extraction so far)
  requirements.txt
  pytest.ini    -- lets `common`/`PipelineA`/`PipelineB`/`metrics` import as top-level packages
```

`common/` isn't one of the folders originally sketched out, but `config.py`
(disease pairs) and `llm_client.py` (the swappable LLM client) are needed by
both Pipeline A and the future Pipeline B, so they live here once instead of
being duplicated into each pipeline.

## What's here so far

- `common/config.py` -- the two disease pairs (T2DM-AD, T2DM-CAD)
- `common/llm_client.py` -- swappable LLM client (local Ollama by default, or
  any OpenAI-compatible hosted endpoint via `.env`)
- `metrics/extract.py` -- regex-based gene/protein symbol extraction from free
  text, feeding BioPathBench's Protein Recall / Hallucination Rate metrics
- `PipelineA/pipeline_a.py` -- **Pipeline A, the Orphaned LLM baseline**: no
  retrieval, no grounding, just the model's own knowledge
- `PipelineB/opentargets_client.py`, `PipelineB/pubmed_client.py`,
  `PipelineB/agent1_discovery.py` -- **Agent 1, Discovery**: Open Targets
  gene-disease lookup + PubMed citation support, orchestrated with LangGraph.
  Confirmed working against the live APIs (2026-09-05).
- `PipelineB/omnipath_client.py`, `PipelineB/agent2_chain.py` -- **Agent 2,
  Chain Construction**: BFS over live OmniPath protein interaction data,
  linking a disease1-associated protein to a disease2-associated one
  (through real intermediate proteins where needed), with a grounded LLM
  choosing among equally-short real candidates rather than generating chain
  data itself. See `PipelineB/README.md` for details, including a
  network-testing caveat.
- `PipelineB/tissue_client.py`, `PipelineB/crossdb_client.py`,
  `PipelineB/agent3_confirmation.py` -- **Agent 3, Independent
  Confirmation**: tissue-expression plausibility (GTEx, HPA fallback) plus
  an independent cross-database gene-disease association check (GWAS
  Catalog, DisGeNET) against each Agent 2 chain candidate's two
  disease-anchored endpoint genes. See `PipelineB/README.md` for the full
  design, including why DisGeNET is optional and why it isn't going through
  the MCP URL suggested during setup.
- `PipelineB/mimic_client.py`, `PipelineB/agent4_mimic.py` -- **Agent 4,
  MIMIC-IV Cohort Check**: local MIMIC-IV query computing whether
  disease1-diagnosed patients show elevated real-world co-occurrence of
  disease2 vs. the population base rate. Needs only two local files, not
  the full MIMIC-IV download -- see `PipelineB/README.md`.

Not built yet: Synthesis (Pipeline B), the rest of BioPathBench, the UI.

## Setup

```
cd Code
pip install -r requirements.txt
cp .env.example .env   # then edit if not using local Ollama on defaults
```

Requires a running LLM to actually generate text. Default assumes a local
Ollama server (`ollama serve`) with a pulled model matching `LLM_MODEL` in
`.env` (defaults to `gemma4:12b`, confirmed via `ollama list` on this
machine -- the only model currently pulled; a larger 31b variant was
tried and removed for RAM reasons).

## Run

```
python -m PipelineA.pipeline_a --pair t2dm_ad
python -m PipelineA.pipeline_a --pair t2dm_cad
python -m PipelineA.pipeline_a --disease1 "X" --disease2 "Y"
```

Prints the narrative and extracted gene/protein symbols as JSON.

```
python -m PipelineB.agent1_discovery --pair t2dm_ad
python -m PipelineB.agent2_chain --pair t2dm_ad
python -m PipelineB.agent3_confirmation --pair t2dm_ad
python -m PipelineB.agent4_mimic --pair t2dm_ad
```

Agent 1 prints Open Targets gene associations for both diseases plus
PubMed citations. Agent 2 runs Agent 1 internally, then searches for a
protein-protein interaction chain between the two diseases and prints it.
Agent 3 runs Agents 1-2 internally, then independently checks each
resulting chain's tissue plausibility and cross-database gene-disease
association. Agent 4 runs standalone (no chain needed) against local
MIMIC-IV files. All need real network access except Agent 4, which needs
local MIMIC-IV data instead -- see `PipelineB/README.md` for setup notes
on all four.

## Test

```
pytest -v
```

Runs everything under `PipelineA/tests/`, `PipelineB/tests/`, `metrics/tests/`,
etc. All tests run offline against stub clients (`StubLLMClient`,
`StubOpenTargetsClient`, `StubPubMedClient`, `StubOmniPathClient`,
`StubGTExClient`, `StubHPAClient`, `StubGWASCatalogClient`,
`StubDisGeNETClient`, `StubMimicClient`) -- no model or network needed,
except `test_mimic_client.py`'s one test against a real (small, synthetic)
gzipped CSV fixture, which is still local-only. 82/82 pass.

## Known limitation

`metrics/extract.py`'s symbol extraction has no gene-symbol reference list
yet, so without passing `known_symbols=`, it over-matches rather than
under-matches (see the docstring). Agent 1 (Open Targets pull) exists now
and can supply that list, but nothing wires it through yet -- Pipeline A
still calls `extract_gene_protein_mentions()` with no `known_symbols`.
