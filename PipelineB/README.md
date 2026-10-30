# Pipeline B -- the grounded, four-stage pipeline

Pipeline B takes two disease names and returns candidate mechanistic chains between them, each built from database
records and checked against independent evidence. It is a staged workflow, not an autonomous system: the stages run in a
fixed order, and **only Stage 2 calls a language model**.

```
disease 1, disease 2
   -> Stage 1  Discovery            (code)   genes + paper titles
   -> Stage 2  LLM-based Judging    (LLM)    path search, judge, debate
   -> Stage 3  Confirmation         (code)   independent checks, evidence floor
   -> Stage 4  Cohort Check         (code)   MIMIC-IV co-occurrence
```

## Stage 1 -- Discovery (`stage1_discovery.py`)

- `opentargets_client.py`: Open Targets GraphQL API (v4). Disease name to ID, then the top 200 associated genes with
  gene symbol, Ensembl ID, UniProt ID and a 0 to 1 association score. A minimum score (default 0.1) is applied, but in
  the saved runs the 200-gene page limit is what binds.
- `pubmed_client.py`: PubMed E-utilities. For the top 10 genes of each disease, the query is `GENE[Title/Abstract] AND
  "disease"[Title/Abstract]`; up to 2 papers (most recent first) are kept as titles and metadata only. PubMed is context,
  not a discovery source, because Open Targets already includes text mining.
- The orchestration is a small LangGraph graph: the two disease lookups run in parallel, and the paper search waits for
  both. This is the only place LangGraph is used.

## Stage 2 -- LLM-based Judging (`stage2_judging.py`, `omnipath_client.py`)

1. **Path search (code).** A multi-source breadth-first search over OmniPath protein-protein interactions, starting from
   all disease-1 genes at once and stopping at any disease-2 gene, up to `--max-hops` (default 5). Every shortest path
   found is kept.
2. **Trim (code).** The candidates are cut to 10: longest first, fully-signed paths first, at most 3 per start gene
   (`--max-per-start-gene`, 0 turns the cap off). The trim does not look at evidence.
3. **Judge (LLM).** One prompt lists all 10 candidates and asks for a PLAUSIBLE or NOT PLAUSIBLE verdict with one
   sentence for each. It runs in two candidate orders (`--judge-passes 2`) plus a repeat of the first order as a noise
   control (`--no-judge-control` turns it off); a candidate endorsed in either order survives.
4. **Debate (LLM).** Judge survivors are compared two at a time ("WINNER: A" or "WINNER: B"), in both A/B orders
   (`--no-swap-debates` turns that off); a disagreement is a draw. Ratings use Elo (start 1200, K = 32).
5. **Provenance check (code).** Every edge in a returned chain must be an edge the search pulled from OmniPath, or the run
   raises `EdgeProvenanceError`. The model's sentences are labelled commentary and any other gene it names is listed in
   `rationale_off_path_genes`.

The model never creates an edge; it chooses among edges the search found. If no path exists, `NoChainFoundError` is
raised. If paths exist and the judge endorses none, the result has an empty chain list.

## Stage 3 -- Confirmation (`stage3_confirmation.py`, `tissue_client.py`, `crossdb_client.py`, `evidence_rerank.py`)

No language model. Each chain is checked against data Stages 1 and 2 never used:

- **Tissue expression (GTEx v8):** a gene counts as expressed at a median of at least 1 TPM in the tissues fixed for each
  disease in `common/config.py`.
- **GWAS Catalog:** gene-disease links for the chain's two endpoint genes. The endpoint genes came from Open Targets, which
  itself uses GWAS data, so this check is not fully independent for the endpoints; for bridge genes it is.
- **Edge literature (PMC full text):** co-mention of the two genes of each bridge edge.
- **Random-gene baseline:** 30 seeded random genes drawn from the same search, to show what a chance chain scores.
- **Evidence floor** (`evidence_rerank.py`): a chain fails if any of its genes is expressed in none of its tissues, or if
  fewer than half of its tissue checks pass. If no chain passes, no chain is recommended.

Every check is reported on its own; there is no single collapsed verdict.

## Stage 4 -- Cohort Check (`stage4_cohort_check.py`, `mimic_client.py`)

No language model. Reads two local MIMIC-IV files (`hosp/diagnoses_icd.csv.gz`, `hosp/d_icd_diagnoses.csv.gz`; set
`MIMIC_DATA_DIR`) and computes a crude, unadjusted relative risk of disease 2 among disease-1 patients, using the ICD-10
prefixes in `common/config.py`. It describes the co-occurrence of the diseases; it says nothing about any chain.

## Other modules

- `metalinks_client.py`: optional gene-metabolite hops (MetalinksDB). Off by default; the thesis reports that adding it
  changed almost nothing. Enable with `--with-metabolites`.

## Tests

`pytest PipelineB/tests` runs everything against stub clients (`StubLLMClient`, `StubOmniPathClient`,
`StubMetalinksClient`, `StubGTExClient`, `StubGWASCatalogClient`, `StubMimicClient`). No network or model is needed.
