# A Multi-Stage, LLM-Assisted Pipeline for Mechanistic Deduction of Disease Comorbidity

MS thesis code (Arrio Gonsalves, Carnegie Mellon University, advisor Dr. Min Xu). The study asks one question: when a
language model proposes a mechanism linking two comorbid diseases, can the proposal be checked against real data? It
compares two approaches on Type 2 diabetes paired with Alzheimer's disease, and with coronary artery disease:

- **Pipeline A** -- the ungrounded control. One language-model call with only the two disease names, no retrieval. It
  returns free text.
- **Pipeline B** -- the grounded system. Four sequential stages. Three of them are ordinary code that calls public
  databases; one (Stage 2) calls a language model to judge candidate chains. Every link in a reported chain is a database
  record, and the chain is checked against evidence the earlier stages did not use.

Only Stage 2 uses a language model, and it is used for two jobs: a *judge* that endorses or rejects each candidate
chain, and a *debate* that compares endorsed chains two at a time. Stages 1, 3 and 4 are fixed code with no model.

## Layout

```
common/       shared by both pipelines: disease-pair config, LLM client, report formatting
PipelineA/    the ungrounded baseline
PipelineB/    the four-stage grounded pipeline, its data clients and tests
scripts/      analyses behind the thesis (run_all, hop sweep, evidence floor, model comparison, audits)
docs/figures/ figures used in the thesis and the architecture diagram
main.py       run both pipelines cell by cell in VS Code's Interactive Window
```

## Pipeline B stages

| Stage | Module | Uses a language model? | What it does |
|---|---|---|---|
| 1. Discovery | `stage1_discovery.py` | no | Open Targets gives the top 200 genes for each disease; PubMed gives titles of up to 2 papers for the top 10 genes. The two gene lookups run in parallel in a LangGraph graph. |
| 2. LLM-based Judging | `stage2_judging.py` | **yes** | A breadth-first search over OmniPath protein interactions finds directed paths (up to 5 hops) from a disease-1 gene to a disease-2 gene. The paths are trimmed to 10; the language model judge endorses or rejects each, and a pairwise debate with Elo scoring ranks the endorsed ones. The model never creates an edge; it only chooses among edges the search found. |
| 3. Confirmation | `stage3_confirmation.py` | no | Checks each chain against evidence the earlier stages did not use: GTEx tissue expression, GWAS Catalog gene-disease links, PMC full-text co-mention for bridge edges, a seeded random-gene baseline, and an evidence floor that can leave a pair with no recommended chain. |
| 4. Cohort Check | `stage4_cohort_check.py` | no | Computes the real-world co-occurrence of the two diseases in local MIMIC-IV data (a crude, unadjusted relative risk). |

## Setup

```
cd Code
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # edit for your LLM endpoint and MIMIC_DATA_DIR
```

Pipeline A and Stage 2 need a language model. The default is a local Ollama server (`ollama serve`) with the model named
by `LLM_MODEL` in `.env` (the thesis uses `gemma4:12b`). Any OpenAI-compatible endpoint also works through
`LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`. Stage 4 needs credentialed MIMIC-IV files; set `MIMIC_DATA_DIR` to the
folder that contains `hosp/`.

## Run

```
python -m scripts.run_all                      # both disease pairs, everything, saved under outputs/
python -m scripts.run_all --pairs t2dm_ad      # one pair
python -m scripts.run_all --skip-stage4        # no MIMIC-IV on this machine
python -m PipelineA.pipeline_a --pair t2dm_ad  # each piece also runs on its own
python -m PipelineB.stage1_discovery --pair t2dm_ad
python -m PipelineB.stage2_judging --pair t2dm_ad
python -m PipelineB.stage3_confirmation --pair t2dm_ad
python -m PipelineB.stage4_cohort_check --pair t2dm_ad
```

`--pair` accepts `t2dm_ad` or `t2dm_cad` (see `common/config.py`); `--disease1` and `--disease2` override it. Stages 1 to 3
need network access to Open Targets, PubMed, OmniPath, GTEx and GWAS Catalog. Saved results go to `outputs/`, which is not
tracked by git.

## Test

```
pytest -v
```

The tests run offline against stub clients, so they need neither a model nor a network.
