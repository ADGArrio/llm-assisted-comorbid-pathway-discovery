"""
main.py -- Run Pipeline A and Pipeline B side by side, cell by cell.

Meant to be run in VS Code's Python Interactive Window: each `# %%` marker
below is its own runnable cell (Run Cell / Shift+Enter). Run the Setup
cell first, then Pipeline A and/or Pipeline B in any order -- Agent 4 and
Pipeline A don't depend on each other and can run independently; Agents
2 and 3 need Agent 1's result first.

Every run is saved as JSON under outputs/, named by disease pair id, so
results persist across interactive sessions and are ready for comparison
and the thesis writeup.

Metrics (BioPathBench) were purged 2026-09-14 to get the bare pipelines
running cleanly first -- see claude/metrics_purge.md. This file only
runs and saves both pipelines' raw output side by side; no scoring yet.
"""

# %% Setup
import json
from pathlib import Path
from typing import Optional

from common.config import get_pair
from PipelineA.pipeline_a import run_pipeline_a
from PipelineB.agent1_discovery import run_agent1_discovery
from PipelineB.agent2_chain import NoChainFoundError, run_agent2_chain
from PipelineB.agent3_confirmation import run_agent3_confirmation
from PipelineB.agent4_mimic import run_agent4_mimic
from PipelineB.metalinks_client import MetalinksClient
from PipelineB.omnipath_client import EXTENDED_DATASETS, OmniPathClient

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Change this to "t2dm_cad" to run the other disease pair.
PAIR_ID = "t2dm_ad"
pair = get_pair(PAIR_ID)
DISEASE1, DISEASE2 = pair["disease1"], pair["disease2"]
print(f"Disease pair: {DISEASE1} -> {DISEASE2}")


def save_json(data: dict, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved: {path}")
    return path


# %% Readable output formatting -- turns raw result dicts into plain text
# (disease1 -> gene1 --activates--> gene2 -> ... -> disease2, plus
# rationale and confirmation summaries) instead of raw JSON/dict dumps.
# Defined early (before any agent runs) since run_pipeline_b_and_save's
# cell below prints through pipeline_b_report immediately after running.
def _format_chain_pathway(chain: dict, disease1: str = None, disease2: str = None) -> str:
    """One endorsed Agent 2 chain (a ChainCandidate.to_dict()) as a single
    arrow-diagram line, e.g. 'APOE --activates--> [metabolite] Cholesterol
    --produces--> ABCA1' -- the mechanistic pathway itself, without the
    edge ids/disease-assoc-scores/hmdb ids that only matter for the saved
    JSON, not for reading it. Pass disease1/disease2 to bookend the line
    with the two diseases the chain is actually claiming to connect, e.g.
    'Type 2 Diabetes Mellitus -> IRS1 --activates--> MAPK8 -> Alzheimer's
    Disease' -- the full claim, not just the gene-to-gene middle."""
    edges = chain["edges"]

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


def _format_confirmation_summary(confirmation: dict) -> str:
    """One chain's Agent 3 result (an Agent3ConfirmationResult.to_dict())
    as plain-language sentences -- which genes were confirmed expressed
    in disease-relevant tissue (GTEx) and which endpoint genes have real
    GWAS Catalog hits for their disease -- instead of the full per-tissue/
    per-SNP JSON array, which is still saved to outputs/ but is not what
    you want to read at a glance. 'No GTEx/GWAS record' is real negative
    evidence, not proof the chain is wrong -- see agent3_confirmation.py's
    module docstring -- so this reports counts and specifics rather than
    a single pass/fail verdict."""
    tissue_checks = confirmation.get("tissue_checks") or []
    expressed = [t for t in tissue_checks if t.get("expressed") is True]
    no_record = [t for t in tissue_checks if t.get("source") == "none"]

    lines = [f"Tissue expression (GTEx): {len(expressed)}/{len(tissue_checks)} gene/tissue checks confirmed expressed."]
    if expressed:
        lines.append("  Expressed: " + ", ".join(f"{t['gene_symbol']} in {t['tissue']}" for t in expressed))
    if no_record:
        genes_with_no_record = sorted({t["gene_symbol"] for t in no_record})
        lines.append(f"  No GTEx record at all for: {', '.join(genes_with_no_record)}")

    lines.append("GWAS Catalog endpoint check:")
    for e in confirmation.get("endpoint_checks") or []:
        if e.get("confirmed"):
            traits = ", ".join(sorted({a["trait_name"] for a in e.get("gwas_associations", [])}))
            lines.append(f"  {e['gene_symbol']} confirmed for {e['disease_name']} ({traits})")
        else:
            lines.append(f"  {e['gene_symbol']}: no GWAS Catalog hit for {e['disease_name']}")

    metabolites = confirmation.get("chain_metabolites") or []
    if metabolites:
        lines.append(f"  (metabolite hop(s) not tissue/GWAS-checked -- gene-centric only: {', '.join(metabolites)})")

    return "\n".join(lines)


def agent1_explanation(agent1_data: Optional[dict], top_n: int = 5) -> str:
    """Agent 1's raw result dict (hundreds of scored genes per disease) as
    one plain-language summary line per disease -- gene counts plus the
    top few by Open Targets disease-association score -- instead of the
    full nested target lists, which are still saved to outputs/ in full."""
    if not agent1_data:
        return ""

    def top_genes(targets: list) -> str:
        ranked = sorted(targets, key=lambda t: t.get("disease_assoc_score", 0.0), reverse=True)[:top_n]
        return ", ".join(f"{t['gene_symbol']} ({t['disease_assoc_score']:.3f})" for t in ranked)

    d1, d2 = agent1_data.get("disease1", ""), agent1_data.get("disease2", "")
    d1_targets = agent1_data.get("disease1_targets") or []
    d2_targets = agent1_data.get("disease2_targets") or []
    return (
        f"Agent 1 (Discovery): {len(d1_targets)} genes associated with {d1} "
        f"(top {top_n}: {top_genes(d1_targets)}); {len(d2_targets)} genes associated with {d2} "
        f"(top {top_n}: {top_genes(d2_targets)})."
    )


def agent4_explanation(agent4_data: Optional[dict]) -> str:
    """Agent 4's raw result dict (an MimicCohortResult.to_dict()) as one
    plain-language sentence -- the real-world MIMIC-IV comorbidity check,
    independent of Agents 1-3 -- instead of the bare rate/count fields."""
    if not agent4_data:
        return ""

    d1, d2 = agent4_data.get("disease1", ""), agent4_data.get("disease2", "")
    rr = agent4_data.get("relative_risk")
    rr_text = f"{rr:.2f}x" if rr is not None else "undefined (disease2 base rate is 0 in this cohort)"
    return (
        f"Agent 4 (MIMIC-IV Cohort Check): of {agent4_data.get('n_total_patients', 0):,} total patients, "
        f"{agent4_data.get('n_disease1_patients', 0):,} had {d1}, {agent4_data.get('n_disease2_patients', 0):,} "
        f"had {d2}, and {agent4_data.get('n_both_patients', 0):,} had both. Patients with {d1} had {d2} at a "
        f"rate of {agent4_data.get('disease2_rate_in_disease1_cohort', 0.0):.2%}, vs. a base rate of "
        f"{agent4_data.get('disease2_base_rate', 0.0):.2%} across the full cohort -- a relative risk of {rr_text}. "
        f"(Crude, unadjusted co-occurrence -- no age/sex matching, no causal claim; see agent4_mimic.py.)"
    )


def agent2_explanation(
    agent2_data: Optional[dict], disease1: str, disease2: str, agent3_data: Optional[list] = None
) -> str:
    """Turns Agent 2's raw result dict into the same kind of plain-language
    explanation Pipeline A gives: the mechanistic pathway(s) it endorsed,
    bookended by the two diseases themselves (disease1 -> gene1
    --activates--> gene2 -> ... -> disease2), plus its rationale for each
    -- or a direct statement of why it endorsed none -- instead of the
    full nested JSON (edge ids, disease-assoc scores, hmdb ids, etc.),
    which is still saved to outputs/ but isn't what you want to read at a
    glance. Pass agent3_data (the list of Agent3ConfirmationResult dicts,
    same order as agent2_data['chains']) to append each chain's tissue/
    GWAS confirmation summary right under its pathway and rationale."""
    if agent2_data is None:
        return (
            f"Agent 2 found no structural path at all connecting {disease1} to {disease2} "
            f"within the max-hop search limit."
        )

    chains = agent2_data.get("chains") or []
    if not chains:
        reason = agent2_data.get("rejection_reason")
        reason_text = f' Its stated reason: "{reason}"' if reason else " (no reason was given)."
        return (
            f"Agent 2 found {agent2_data.get('candidate_path_count', 0)} structurally real "
            f"candidate pathway(s) connecting {disease1} to {disease2}, but did not judge any "
            f"of them biologically plausible enough to endorse.{reason_text}"
        )

    blocks = []
    for i, chain in enumerate(chains, start=1):
        label = f"Pathway {i}" if len(chains) > 1 else "Pathway"
        rationale = chain.get("rationale") or "(no rationale given)"
        caveat = (
            "\n(Note: at least one step's direction of effect is not confirmed by curated "
            "evidence -- treat this pathway as tentative.)"
            if chain.get("has_unconfirmed_direction")
            else ""
        )
        block = f"{label}: {_format_chain_pathway(chain, disease1, disease2)}\n{rationale}{caveat}"
        if agent3_data is not None and i - 1 < len(agent3_data):
            block += "\n" + _format_confirmation_summary(agent3_data[i - 1])
        blocks.append(block)
    return "\n\n".join(blocks)


def pipeline_b_report(b_data: dict) -> str:
    """The full Pipeline B combined result (as saved by
    run_pipeline_b_and_save: agent1/agent2/agent3/agent4 keys) as one
    readable plain-text report -- Agent 1's top genes, each endorsed
    pathway (disease1 -> gene1 --activates--> ... -> disease2) with its
    rationale and Agent 3 confirmation summary, and Agent 4's MIMIC-IV
    comorbidity numbers -- instead of printing the raw dict, which is a
    wall of nested JSON-shaped Python data most useful for saved output,
    not for reading right after a run. The full nested detail is still
    saved to outputs/ in full by run_pipeline_b_and_save; this is what to
    read instead of printing pipeline_b_output directly."""
    disease1, disease2 = b_data.get("disease1", ""), b_data.get("disease2", "")
    sections = [f"{disease1} -> {disease2}"]

    a1_line = agent1_explanation(b_data.get("agent1"))
    if a1_line:
        sections.append(a1_line)

    sections.append(agent2_explanation(b_data.get("agent2"), disease1, disease2, agent3_data=b_data.get("agent3")))

    a4_line = agent4_explanation(b_data.get("agent4"))
    if a4_line:
        sections.append(a4_line)

    return "\n\n".join(sections)


# %% Pipeline A -- Orphaned LLM baseline
def run_pipeline_a_and_save(disease1: str = DISEASE1, disease2: str = DISEASE2) -> dict:
    """Runs Pipeline A (one ungrounded LLM call) and saves its output."""
    result = run_pipeline_a(disease1, disease2)
    data = result.to_dict()
    save_json(data, f"pipeline_a_{PAIR_ID}.json")
    return data


def show_narrative(data: dict) -> None:
    """Renders a pipeline-A-shaped result's narrative as actual formatted
    Markdown (headers, bold, arrows) instead of a raw Python dict repr --
    the LLM writes real Markdown/LaTeX (###, **bold**, $\\rightarrow$),
    which just shows up as literal syntax if you print the dict directly.
    Falls back to plain print outside a notebook/IPython context."""
    header = f"### {data['disease1']} → {data['disease2']}"
    try:
        from IPython.display import Markdown, display
        display(Markdown(f"{header}\n\n{data['narrative']}"))
    except ImportError:
        print(f"{data['disease1']} -> {data['disease2']}\n")
        print(data["narrative"])


pipeline_a_output = run_pipeline_a_and_save()
show_narrative(pipeline_a_output)


# %% Pipeline B -- Agent 1 (Discovery)
def run_agent1(disease1: str = DISEASE1, disease2: str = DISEASE2, min_score: float = 0.1):
    """Agent 1: Open Targets gene-disease discovery + PubMed citations."""
    return run_agent1_discovery(disease1, disease2, min_score=min_score)


agent1_result = run_agent1()
agent1_result.to_dict()


# %% Pipeline B -- Agent 2 (Chain Construction)
def run_agent2(
    agent1_result,
    max_hops: int = None,
    include_metabolites: bool = True,
    omnipath_datasets: Optional[str] = None,
):
    """Agent 2: BFS chain construction over OmniPath (+ MetalinksDB metabolite
    hops by default), with an LLM adjudicating between equally-short candidates.
    Returns None if no structural path exists within max_hops (NoChainFoundError
    from BFS itself). A non-None result with an EMPTY chains list is a different,
    equally valid outcome: BFS found real candidates, but the LLM judged none of
    them biologically plausible enough to endorse -- printed explicitly below so
    it's not mistaken for main.py silently doing nothing.

    omnipath_datasets (added 2026-09-14): None (default) queries OmniPath's core
    curated set only. Pass "extended" to widen to core + pathwayextra + kinaseextra
    (still curated, broader coverage -- see omnipath_client.py's module docstring),
    or a raw comma-separated dataset string for anything else. This is the lever for
    the core-vs-extended OmniPath coverage comparison -- e.g.
    run_agent2(agent1_result) vs run_agent2(agent1_result, omnipath_datasets="extended")
    on the same agent1_result, then compare candidate_path_count and the chains
    themselves. See claude/agent2_bfs_and_prompt_recalibration.md."""
    kwargs = {"max_hops": max_hops} if max_hops else {}
    metalinks_client = MetalinksClient() if include_metabolites else None
    if omnipath_datasets == "extended":
        kwargs["omnipath_client"] = OmniPathClient(datasets=EXTENDED_DATASETS)
    elif omnipath_datasets:
        kwargs["omnipath_client"] = OmniPathClient(datasets=omnipath_datasets)
    try:
        result = run_agent2_chain(agent1_result, metalinks_client=metalinks_client, **kwargs)
    except NoChainFoundError as e:
        print(f"Agent 2: {e}")
        return None

    if not result.chains:
        reason = (
            f' LLM\'s stated reason: "{result.rejection_reason}"'
            if result.rejection_reason
            else " (no reason given)"
        )
        print(
            f"Agent 2 found {result.candidate_path_count} structural candidate path(s) "
            f"(see shared_genes/candidate_path_count below), but the LLM did not judge any "
            f"of them biologically plausible enough to endorse.{reason} This is a real, "
            f"expected possible outcome -- not a bug -- see agent2_chain.py's module docstring."
        )
    return result


agent2_result = run_agent2(agent1_result)
agent2_result.to_dict() if agent2_result else None


# %% Pipeline B -- Agent 3 (Independent Confirmation)
def run_agent3(agent1_result, agent2_result):
    """Agent 3: tissue-expression (GTEx) + cross-database (GWAS Catalog)
    confirmation for each Agent 2 chain candidate. Returns [] if Agent 2
    found no chain."""
    if agent2_result is None:
        return []
    return run_agent3_confirmation(agent1_result, agent2_result)


agent3_results = run_agent3(agent1_result, agent2_result)
[r.to_dict() for r in agent3_results]


# %% Pipeline B -- Agent 4 (MIMIC-IV Cohort Check)
def run_agent4(disease1: str = DISEASE1, disease2: str = DISEASE2):
    """Agent 4: real-world comorbidity check against local MIMIC-IV data.
    Independent of Agents 1-3 -- needs MIMIC_DATA_DIR set in .env."""
    return run_agent4_mimic(disease1, disease2)


agent4_result = run_agent4()
agent4_result.to_dict()


# %% Pipeline B -- run all four agents together and save one combined result
def run_pipeline_b_and_save(disease1: str = DISEASE1, disease2: str = DISEASE2) -> dict:
    """Runs Agents 1-4 in sequence and saves one combined JSON result."""
    a1 = run_agent1(disease1, disease2)
    a2 = run_agent2(a1)
    a3 = run_agent3(a1, a2)
    a4 = run_agent4(disease1, disease2)

    data = {
        "disease1": disease1,
        "disease2": disease2,
        "agent1": a1.to_dict(),
        "agent2": a2.to_dict() if a2 else None,
        "agent3": [c.to_dict() for c in a3],
        "agent4": a4.to_dict(),
    }
    save_json(data, f"pipeline_b_{PAIR_ID}.json")
    return data


pipeline_b_output = run_pipeline_b_and_save()
print(pipeline_b_report(pipeline_b_output))


# %% Side-by-side comparison: Pipeline A output vs. Pipeline B output
def _narrative_to_html(narrative: str) -> str:
    """Small, dependency-free Markdown-ish -> HTML conversion for an LLM
    narrative -- just enough of the subset these prompts actually produce
    (###/## headers, **bold**, $\\rightarrow$/\\rightarrow arrows, ---
    rules, paragraph breaks) to make it readable in a raw HTML display,
    instead of one escaped JSON string with literal \\n and ** in it.
    Escapes the raw text first, then applies these substitutions, so any
    stray '<'/'&' in the narrative itself renders safely."""
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


def show_side_by_side(a_data: dict, b_data: dict) -> None:
    """Renders Pipeline A's narrative and Pipeline B's full readable report
    (pathways bookended by the disease names, rationale, and Agent 3
    confirmation summary -- see agent2_explanation/pipeline_b_report) side
    by side, both as readable text rather than raw JSON. The full nested
    output (all four agents, edge ids, scores, etc.) is still saved under
    outputs/ by run_pipeline_b_and_save -- this view is deliberately just
    the headline explanation. In VS Code's Interactive Window this
    displays as a real two-pane view; falls back to plain sequential
    printing outside a notebook/IPython context."""
    b_explanation = agent2_explanation(
        b_data.get("agent2"), b_data.get("disease1", ""), b_data.get("disease2", ""), agent3_data=b_data.get("agent3")
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
            <h3>Pipeline B -- Grounded Multi-Agent</h3>
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