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

from common.config import get_pair
from PipelineA.pipeline_a import run_pipeline_a
from PipelineB.agent1_discovery import run_agent1_discovery
from PipelineB.agent2_chain import NoChainFoundError, run_agent2_chain
from PipelineB.agent3_confirmation import run_agent3_confirmation
from PipelineB.agent4_mimic import run_agent4_mimic
from PipelineB.metalinks_client import MetalinksClient

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
def run_agent2(agent1_result, max_hops: int = None, include_metabolites: bool = True):
    """Agent 2: BFS chain construction over OmniPath (+ MetalinksDB metabolite
    hops by default), with an LLM adjudicating between equally-short candidates.
    Returns None if no structural path exists within max_hops."""
    kwargs = {"max_hops": max_hops} if max_hops else {}
    metalinks_client = MetalinksClient() if include_metabolites else None
    try:
        return run_agent2_chain(agent1_result, metalinks_client=metalinks_client, **kwargs)
    except NoChainFoundError as e:
        print(f"Agent 2: {e}")
        return None


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
pipeline_b_output


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
    """Renders Pipeline A's narrative (formatted, not raw JSON) and
    Pipeline B's structured output (JSON -- it's nested chain/agent data,
    not prose, so JSON is already the readable form) as two columns. In
    VS Code's Interactive Window this displays as a real two-pane view;
    falls back to plain sequential printing outside a notebook/IPython
    context."""
    b_text = json.dumps(b_data, indent=2)

    try:
        from IPython.display import HTML, display
        import html as html_lib

        a_html = _narrative_to_html(a_data.get("narrative", ""))
        subtitle = f"{html_lib.escape(a_data.get('disease1', ''))} → {html_lib.escape(a_data.get('disease2', ''))}"

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
            <pre style="white-space:pre-wrap; background:#f5f5f5; color:#111; padding:8px;
                        border-radius:6px; max-height:600px; overflow:auto; font-family:monospace;">{html_lib.escape(b_text)}</pre>
          </div>
        </div>
        """))
    except ImportError:
        print("=" * 30, "PIPELINE A", "=" * 30)
        print(f"{a_data.get('disease1')} -> {a_data.get('disease2')}\n")
        print(a_data.get("narrative", ""))
        print("=" * 30, "PIPELINE B", "=" * 30)
        print(b_text)


show_side_by_side(pipeline_a_output, pipeline_b_output)
