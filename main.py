"""Runs Pipeline A and Pipeline B side by side, cell by cell, for use in VS Code's Python Interactive Window.

Each `# %%` marker starts a runnable cell. Run the Setup cell first, then Pipeline A and/or Pipeline B in any order
(Stages 2 and 3 need Stage 1's result first; Stage 4 and Pipeline A are independent). Every run is saved as JSON under
outputs/, named by disease pair id.
"""

# %% Setup
import json
from pathlib import Path
from typing import Optional

from common.report_format import _format_chain_evidence, _format_debate_log
from common.config import get_pair
from common.llm_client import LLMClient
from PipelineA.pipeline_a import run_pipeline_a
from PipelineB.stage1_discovery import run_stage1_discovery
from PipelineB.stage2_judging import NoChainFoundError, run_stage2_judging
from PipelineB.stage3_confirmation import run_stage3_confirmation
from PipelineB.stage4_cohort_check import run_stage4_cohort_check
from PipelineB.metalinks_client import MetalinksClient
from PipelineB.omnipath_client import EXTENDED_DATASETS, OmniPathClient
from PipelineB.pubmed_client import PubMedClient

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Disease pair to run: "t2dm_ad" or "t2dm_cad".
PAIR_ID = "t2dm_cad"
pair = get_pair(PAIR_ID)
DISEASE1, DISEASE2 = pair["disease1"], pair["disease2"]
print(f"Disease pair: {DISEASE1} -> {DISEASE2}")


# Write a dict as indented JSON under outputs/. Input: the data and a file name. Output: the path written.
def save_json(data: dict, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved: {path}")
    return path


# %% Readable output formatting
# Render one Stage 2 chain as a single arrow line, e.g. GENE1 --activates--> GENE2 -> ... Input: a chain dict,
# optionally the two disease names to bookend the line. Output: the line.
def _format_chain_pathway(chain: dict, disease1: str = None, disease2: str = None) -> str:
    edges = chain["edges"]

    # Label one chain node, tagging metabolites. Input: node kind and name. Output: the display string.
    def node(kind: str, name: str) -> str:
        return f"[metabolite] {name}" if kind == "metabolite" else name

    parts = [node(edges[0]["source_kind"], edges[0]["source_gene"])]
    for edge in edges:
        parts.append(f'--{edge["interaction_type"]}-->')
        parts.append(node(edge["target_kind"], edge["target_gene"]))
    pathway = " ".join(parts)
    if disease1 and disease2:
        return f"{disease1} -> {pathway} -> {disease2}"
    return pathway


# Summarise one chain's Stage 3 result in plain sentences: tissue expression, GWAS hits for the endpoint genes and edge
# literature. Input: a Stage 3 result dict. Output: the text.
def _format_confirmation_summary(confirmation: dict) -> str:
    tissue_checks = confirmation.get("tissue_checks") or []
    expressed = [t for t in tissue_checks if t.get("expressed") is True]
    no_record = [t for t in tissue_checks if t.get("source") == "none"]

    lines = [f"Tissue expression (GTEx): {len(expressed)}/{len(tissue_checks)} gene/tissue checks confirmed expressed."]
    if expressed:
        lines.append("  Expressed: " + ", ".join(f"{t['gene_symbol']} in {t['tissue']}" for t in expressed))
    if no_record:
        genes_with_no_record = sorted({t["gene_symbol"] for t in no_record})
        lines.append(f"  No GTEx record at all for: {', '.join(genes_with_no_record)}")

    lines.append("GWAS Catalog endpoint check (consistent with the selection evidence, not independent: Stage 1 chose these "
                 "genes by an Open Targets score that already includes GWAS-derived evidence):")
    for e in confirmation.get("endpoint_checks") or []:
        if e.get("confirmed"):
            traits = ", ".join(sorted({a["trait_name"] for a in e.get("gwas_associations", [])}))
            lines.append(f"  {e['gene_symbol']} confirmed for {e['disease_name']} ({traits})")
        elif e.get("possibly_incomplete"):
            # A check that gave up after exhausting its rate-limit retries is inconclusive, not a confirmed negative.
            lines.append(
                f"  {e['gene_symbol']}: no GWAS Catalog hit for {e['disease_name']} "
                f"(rate-limited before finishing this check -- inconclusive, not a confirmed negative)"
            )
        else:
            lines.append(f"  {e['gene_symbol']}: no GWAS Catalog hit for {e['disease_name']}")

    bridge_gwas = confirmation.get("bridge_gwas_checks") or []
    if bridge_gwas:
        lines.append("GWAS Catalog bridge-gene check (independent of how the genes were selected):")
        for g in sorted({b["gene_symbol"] for b in bridge_gwas}):
            parts = []
            for b in (x for x in bridge_gwas if x["gene_symbol"] == g):
                if b.get("confirmed"):
                    parts.append(f"hit for {b['disease_name']}")
                elif b.get("possibly_incomplete"):
                    parts.append(f"{b['disease_name']}: inconclusive (rate-limited)")
                else:
                    parts.append(f"no hit for {b['disease_name']}")
            lines.append(f"  {g}: " + "; ".join(parts))
    null = confirmation.get("null_baseline")
    if null:
        # Format a fraction as a percentage string. Input: a number or None. Output: the string.
        def pct(x):
            return "n/a" if x is None else f"{x:.0%}"
        chains = null.get("chains") or {}
        lines.append(
            f"Random-gene baseline ({null['n_genes']} genes drawn from {null['pool_size']} genes the search could have used as bridges):"
        )
        lines.append(
            f"  tissue expression: chain bridge genes {pct(chains.get('tissue_expressed_rate'))} vs baseline {pct(null.get('tissue_expressed_rate'))}"
        )
        for disease, b in (null.get("gwas") or {}).items():
            c = (chains.get("gwas") or {}).get(disease, {})
            lines.append(
                f"  GWAS hit for {disease}: chain bridge genes {c.get('genes_with_hit', 0)}/{c.get('genes_checked', 0)} "
                f"vs baseline {pct(b.get('hit_rate'))} ({b.get('genes_with_hit', 0)}/{b.get('genes_checked', 0)})"
            )

    metabolites = confirmation.get("chain_metabolites") or []
    if metabolites:
        lines.append(f"  (metabolite hop(s) not tissue/GWAS-checked -- gene-centric only: {', '.join(metabolites)})")

    edge_checks = confirmation.get("edge_literature_checks") or []
    if edge_checks:
        # Bridge edges only; 1- and 2-hop chains have none.
        lines.append("Edge-level literature check (bridge edges only):")
        for check in edge_checks:
            label = f"{check['gene_a']}-{check['gene_b']}"
            if check.get("supported"):
                confirming = next((c for c in check["citations"] if c.get("both_genes_confirmed")), None)
                source = "full text" if confirming and confirming.get("full_text_checked") else "abstract"
                lines.append(f"  {label}: supported by a passage in PMID {confirming['pmid']} ({source})")
            elif check.get("citations"):
                text_kind = "full text" if check.get("any_full_text_checked") else "abstract only"
                lines.append(
                    f"  {label}: {len(check['citations'])} citation(s) found co-mentioning both genes in "
                    f"the original search, but no passage in the checked {text_kind} actually named both together"
                )
            else:
                lines.append(f"  {label}: no PubMed citation found co-mentioning both genes at all")

    return "\n".join(lines)


# Summarise Stage 1 as one line per disease: gene counts and the top genes by Open Targets score. Input: Stage 1 result
# dict and how many top genes to list. Output: the text.
def stage1_explanation(stage1_data: Optional[dict], top_n: int = 5) -> str:
    if not stage1_data:
        return ""

    # List the top gene symbols with their scores. Input: a list of scored targets. Output: the display string.
    def top_genes(targets: list) -> str:
        ranked = sorted(targets, key=lambda t: t.get("disease_assoc_score", 0.0), reverse=True)[:top_n]
        return ", ".join(f"{t['gene_symbol']} ({t['disease_assoc_score']:.3f})" for t in ranked)

    d1, d2 = stage1_data.get("disease1", ""), stage1_data.get("disease2", "")
    d1_targets = stage1_data.get("disease1_targets") or []
    d2_targets = stage1_data.get("disease2_targets") or []
    return (
        f"Stage 1 (Discovery): {len(d1_targets)} genes associated with {d1} "
        f"(top {top_n}: {top_genes(d1_targets)}); {len(d2_targets)} genes associated with {d2} "
        f"(top {top_n}: {top_genes(d2_targets)})."
    )


# Summarise the Stage 4 MIMIC-IV cohort check in one sentence. Input: Stage 4 result dict. Output: the text.
def stage4_explanation(stage4_data: Optional[dict]) -> str:
    if not stage4_data:
        return ""

    d1, d2 = stage4_data.get("disease1", ""), stage4_data.get("disease2", "")
    n_total, n1, n2, both = (stage4_data.get(k, 0) for k in
                             ("n_total_patients", "n_disease1_patients", "n_disease2_patients", "n_both_patients"))
    # Recompute the statistics from the counts for old saved results that lack them.
    from PipelineB.stage4_cohort_check import _two_by_two_stats
    stats = _two_by_two_stats(n_total, n1, n2, both) if n_total else {}
    rr, ci, orr, p = stats.get("relative_risk"), stats.get("relative_risk_ci95"), stats.get("odds_ratio"), stats.get("p_value")
    unexposed = n_total - n1
    unexposed_rate = (n2 - both) / unexposed if unexposed else 0.0
    rr_text = (f"{rr:.2f} (95% CI {ci[0]:.2f}-{ci[1]:.2f}; odds ratio {orr:.2f}"
               + ((f"; p={p:.1e}" if p > 0 else "; p<1e-300") if p is not None else "") + ")") if rr is not None and ci and orr is not None else "undefined"
    return (
        f"Stage 4 (Cohort Check): of {n_total:,} total patients, "
        f"{n1:,} had {d1}, {n2:,} had {d2}, and {both:,} had both. Patients with {d1} had {d2} at a "
        f"rate of {stage4_data.get('disease2_rate_in_disease1_cohort', 0.0):.2%}, vs. {unexposed_rate:.2%} among patients "
        f"without {d1} -- a relative risk of {rr_text}. "
        f"(Crude, unadjusted co-occurrence -- no age/sex matching, no causal claim; see stage4_cohort_check.py.)"
    )


# Summarise Stage 2 as the endorsed pathways (bookended by the two diseases) with the model's rationale and the database
# evidence line, or a statement that none was endorsed. Input: Stage 2 result dict and options. Output: the text.
def stage2_explanation(
    stage2_data: Optional[dict], disease1: str, disease2: str, stage3_data: Optional[list] = None
) -> str:
    if stage2_data is None:
        return (
            f"Stage 2 found no structural path at all connecting {disease1} to {disease2} "
            f"within the max-hop search limit."
        )

    chains = stage2_data.get("chains") or []
    if not chains:
        reason = stage2_data.get("rejection_reason")
        reason_text = f' Its stated reason: "{reason}"' if reason else " (no reason was given)."
        return (
            f"Stage 2 found {stage2_data.get('candidate_path_count', 0)} structurally real "
            f"candidate pathway(s) connecting {disease1} to {disease2}, but did not judge any "
            f"of them biologically plausible enough to endorse.{reason_text}"
        )

    blocks = []
    for i, chain in enumerate(chains, start=1):
        label = f"Pathway {i}" if len(chains) > 1 else "Pathway"
        # Chains are sorted by Elo rating, so elo_rating is also the pathway's rank.
        elo = chain.get("elo_rating")
        elo_text = f" [Elo {elo:.0f}]" if elo is not None else ""
        rationale = chain.get("rationale") or "(no rationale given)"
        # The rationale is the model's own free text, labelled as commentary; the evidence line below it is generated by
        # code from database records.
        rationale = f"Model commentary (unverified): {rationale}" if chain.get("rationale") else rationale
        evidence_line = _format_chain_evidence(chain)
        if evidence_line:
            rationale = f"{evidence_line}\n{rationale}"
        caveat = (
            "\n(Note: at least one step's direction of effect is not confirmed by curated "
            "evidence -- treat this pathway as tentative.)"
            if chain.get("has_unconfirmed_direction")
            else ""
        )
        block = f"{label}{elo_text}: {_format_chain_pathway(chain, disease1, disease2)}\n{rationale}{caveat}"
        if stage3_data is not None and i - 1 < len(stage3_data):
            block += "\n" + _format_confirmation_summary(stage3_data[i - 1])
        blocks.append(block)

    debate_log = stage2_data.get("debate_log") or []
    if debate_log:
        blocks.append(_format_debate_log(debate_log, len(chains)))

    return "\n\n".join(blocks)


# Build the full readable Pipeline B report: Stage 1 top genes, each endorsed pathway with its Stage 3 confirmation, and
# Stage 4 numbers. Input: the combined result dict. Output: the text.
def pipeline_b_report(b_data: dict) -> str:
    disease1, disease2 = b_data.get("disease1", ""), b_data.get("disease2", "")
    sections = [f"{disease1} -> {disease2}"]

    a1_line = stage1_explanation(b_data.get("stage1"))
    if a1_line:
        sections.append(a1_line)

    sections.append(stage2_explanation(b_data.get("stage2"), disease1, disease2, stage3_data=b_data.get("stage3")))

    a4_line = stage4_explanation(b_data.get("stage4"))
    if a4_line:
        sections.append(a4_line)

    return "\n\n".join(sections)


# %% Pipeline A -- Orphaned LLM baseline
# Run Pipeline A once and save the result. Input: the two disease names and an optional Ollama model name that overrides
# LLM_MODEL. Output: the result dict.
def run_pipeline_a_and_save(disease1: str = DISEASE1, disease2: str = DISEASE2, llm_model: str = None) -> dict:
    client = LLMClient(model=llm_model) if llm_model else None
    result = run_pipeline_a(disease1, disease2, client=client)
    data = result.to_dict()
    save_json(data, f"pipeline_a_{PAIR_ID}.json")
    return data


# Display Pipeline A's narrative as formatted Markdown (plain print outside a notebook). Input: a Pipeline A result
# dict. Output: none.
def show_narrative(data: dict) -> None:
    header = f"### {data['disease1']} → {data['disease2']}"
    try:
        from IPython.display import Markdown, display
        display(Markdown(f"{header}\n\n{data['narrative']}"))
    except ImportError:
        print(f"{data['disease1']} -> {data['disease2']}\n")
        print(data["narrative"])


pipeline_a_output = run_pipeline_a_and_save()
show_narrative(pipeline_a_output)


# %% Pipeline B -- Stage 1 (Discovery)
# Run Stage 1 (Open Targets discovery and PubMed citations). Input: the two disease names and a minimum association
# score. Output: the Stage 1 result.
def run_stage1(disease1: str = DISEASE1, disease2: str = DISEASE2, min_score: float = 0.1):
    return run_stage1_discovery(disease1, disease2, min_score=min_score)


stage1_result = run_stage1()
stage1_result.to_dict()


# %% Pipeline B -- Stage 2 (LLM-based Judging)
# Run Stage 2 (OmniPath breadth-first search, judge, control pass and debate). Input: the Stage 1 result and options
# (hop limit, metabolite hops, OmniPath dataset set, debate on/off, model name). Output: the Stage 2 result, or None if
# no path exists within the hop limit.
def run_stage2(
    stage1_result,
    max_hops: int = None,
    include_metabolites: bool = False,
    omnipath_datasets: Optional[str] = None,
    enable_debate_tournament: bool = True,
    llm_model: str = None,
):
    kwargs = {"max_hops": max_hops} if max_hops else {}
    kwargs["enable_debate_tournament"] = enable_debate_tournament
    # Options shared with scripts/run_all.py: a repeat judge pass as a noise control, and edge strength plus bridge-gene
    # titles in the judge prompt.
    kwargs.update(judge_control_pass=True, judge_evidence=True, pubmed_client=PubMedClient())
    if llm_model:
        kwargs["llm_client"] = LLMClient(model=llm_model)
    metalinks_client = MetalinksClient() if include_metabolites else None
    if omnipath_datasets == "extended":
        kwargs["omnipath_client"] = OmniPathClient(datasets=EXTENDED_DATASETS)
    elif omnipath_datasets:
        kwargs["omnipath_client"] = OmniPathClient(datasets=omnipath_datasets)
    try:
        result = run_stage2_judging(stage1_result, metalinks_client=metalinks_client, **kwargs)
    except NoChainFoundError as e:
        print(f"Stage 2: {e}")
        return None

    if not result.chains:
        reason = (
            f' LLM\'s stated reason: "{result.rejection_reason}"'
            if result.rejection_reason
            else " (no reason given)"
        )
        print(
            f"Stage 2 found {result.candidate_path_count} structural candidate path(s) "
            f"(see shared_genes/candidate_path_count below), but the LLM did not judge any "
            f"of them biologically plausible enough to endorse.{reason} This is a real, "
            f"expected possible outcome -- not a bug -- see stage2_judging.py's module docstring."
        )
    return result


stage2_result = run_stage2(stage1_result)
stage2_result.to_dict() if stage2_result else None


# %% Pipeline B -- Stage 3 (Confirmation)
# Run Stage 3 (GTEx, GWAS Catalog and edge-literature confirmation) for every Stage 2 chain. Input: the Stage 1 and
# Stage 2 results and a flag to skip the edge-literature check. Output: a list of per-chain results, empty if Stage 2
# found no chain.
def run_stage3(stage1_result, stage2_result, enable_edge_literature: bool = True):
    if stage2_result is None:
        return []
    pubmed_client = None if enable_edge_literature else _NoOpPubMedClient()
    return run_stage3_confirmation(stage1_result, stage2_result, pubmed_client=pubmed_client)


# PubMed stand-in that returns no citations, used when the edge-literature check is turned off.
class _NoOpPubMedClient:

    # Input: two gene symbols and a result limit. Output: an empty list.
    def search_edge_citations(self, gene_a: str, gene_b: str, retmax: int = 3):
        return []


stage3_results = run_stage3(stage1_result, stage2_result)
[r.to_dict() for r in stage3_results]


# %% Pipeline B -- Stage 4 (Cohort Check)
# Run Stage 4 (MIMIC-IV cohort check; needs MIMIC_DATA_DIR in .env). Input: the two disease names. Output: the Stage 4
# result.
def run_stage4(disease1: str = DISEASE1, disease2: str = DISEASE2):
    return run_stage4_cohort_check(disease1, disease2)


stage4_result = run_stage4()
stage4_result.to_dict()


# %% Pipeline B -- run all four stages together and save one combined result
# Run Stages 1-4 in sequence and save one combined JSON result. Input: the two disease names. Output: the combined
# result dict.
def run_pipeline_b_and_save(disease1: str = DISEASE1, disease2: str = DISEASE2) -> dict:
    a1 = run_stage1(disease1, disease2)
    a2 = run_stage2(a1)
    a3 = run_stage3(a1, a2)
    a4 = run_stage4(disease1, disease2)

    data = {
        "disease1": disease1,
        "disease2": disease2,
        "stage1": a1.to_dict(),
        "stage2": a2.to_dict() if a2 else None,
        "stage3": [c.to_dict() for c in a3],
        "stage4": a4.to_dict(),
    }
    save_json(data, f"pipeline_b_{PAIR_ID}.json")
    return data


pipeline_b_output = run_pipeline_b_and_save()
print(pipeline_b_report(pipeline_b_output))


# %% Side-by-side comparison: Pipeline A output vs. Pipeline B output
# Convert an LLM narrative (headers, bold, arrows, rules, paragraphs) to simple HTML after escaping it. Input: the
# narrative text. Output: an HTML string.
def _narrative_to_html(narrative: str) -> str:
    import re
    import html as html_lib

    text = html_lib.escape(narrative)
    text = text.replace(r"$\rightarrow$", " → ").replace(r"\rightarrow", " → ")
    text = re.sub(r"^### (.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"^---$", "<hr>", text, flags=re.MULTILINE)
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return "".join(b if b.startswith(("<h", "<hr")) else f"<p>{b}</p>" for b in blocks)


# Display Pipeline A's narrative next to Pipeline B's readable report (plain sequential print outside a notebook).
# Input: the two result dicts. Output: none.
def show_side_by_side(a_data: dict, b_data: dict) -> None:
    b_explanation = stage2_explanation(
        b_data.get("stage2"), b_data.get("disease1", ""), b_data.get("disease2", ""), stage3_data=b_data.get("stage3")
    )

    try:
        from IPython.display import HTML, display
        import html as html_lib

        a_html = _narrative_to_html(a_data.get("narrative", ""))
        subtitle = f"{html_lib.escape(a_data.get('disease1', ''))} → {html_lib.escape(a_data.get('disease2', ''))}"
        b_html = "".join(f"<p>{html_lib.escape(b)}</p>" for b in b_explanation.split("\n\n"))

        display(HTML(f"""
        <div style="display:flex; gap:16px; font-family:sans-serif; font-size:13px;">
          <div style="flex:1; min-width:0;">
            <h3>Pipeline A -- Orphaned LLM</h3>
            <div style="background:#f5f5f5; color:#111; padding:12px; border-radius:6px;
                        max-height:600px; overflow:auto; line-height:1.5;">
              <p><em>{subtitle}</em></p>
              {a_html}
            </div>
          </div>
          <div style="flex:1; min-width:0;">
            <h3>Pipeline B -- Grounded Multi-Stage</h3>
            <div style="background:#f5f5f5; color:#111; padding:12px; border-radius:6px;
                        max-height:600px; overflow:auto; line-height:1.5; white-space:pre-wrap;">
              <p><em>{subtitle}</em></p>
              {b_html}
            </div>
          </div>
        </div>
        """))
    except ImportError:
        print("=" * 30, "PIPELINE A", "=" * 30)
        print(f"{a_data.get('disease1')} -> {a_data.get('disease2')}\n")
        print(a_data.get("narrative", ""))
        print("=" * 30, "PIPELINE B", "=" * 30)
        print(b_explanation)


show_side_by_side(pipeline_a_output, pipeline_b_output)

# %%
