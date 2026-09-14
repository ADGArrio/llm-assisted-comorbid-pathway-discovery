"""
main.py -- Run Pipeline A and Pipeline B side by side, cell by cell.

Meant to be run in VS Code's Python Interactive Window: each `# %%` marker
below is its own runnable cell (Run Cell / Shift+Enter). Run the Setup
cell first, then Pipeline A and/or Pipeline B in any order -- Agent 4 and
Pipeline A don't depend on each other and can run independently; Agents
2 and 3 need Agent 1's result first.

Every run is saved as JSON under outputs/, named by disease pair id, so
results persist across interactive sessions and are ready for comparison
(BioPathBench scoring, the thesis writeup, etc.).
"""

# %% Setup
import json
from pathlib import Path

from common.config import get_pair
from metrics.benchmark import hallucination_rate, protein_recall
from metrics.gene_reference import load_known_gene_symbols
from PipelineA.pipeline_a import run_pipeline_a
from PipelineB.agent1_discovery import run_agent1_discovery
from PipelineB.agent2_chain import NoChainFoundError, run_agent2_chain
from PipelineB.agent3_confirmation import run_agent3_confirmation
from PipelineB.agent4_mimic import run_agent4_mimic
from PipelineB.crossdb_client import DisGeNETClient
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


pipeline_a_output = run_pipeline_a_and_save()
pipeline_a_output


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
def run_agent3(agent1_result, agent2_result, use_disgenet: bool = False):
    """Agent 3: tissue-expression + cross-database confirmation for each
    Agent 2 chain candidate. Returns [] if Agent 2 found no chain."""
    if agent2_result is None:
        return []
    disgenet_client = DisGeNETClient() if use_disgenet else None
    return run_agent3_confirmation(agent1_result, agent2_result, disgenet_client=disgenet_client)


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


# %% BioPathBench: score Pipeline A against Agent 1's grounded gene set
def score_pipeline_a(pipeline_a_data: dict, agent1_result) -> dict:
    """Protein Recall (vs. Agent 1's discovered genes) and Hallucination
    Rate (vs. the full HGNC gene list -- set HGNC_GENE_LIST_PATH in .env)."""
    cited = pipeline_a_data["cited_symbols"]
    scores = {"protein_recall": protein_recall(cited, agent1_result)}
    try:
        known_symbols = load_known_gene_symbols()
        scores["hallucination_rate"] = hallucination_rate(cited, known_symbols)
    except FileNotFoundError as e:
        scores["hallucination_rate"] = None
        print(f"Hallucination rate skipped: {e}")
    return scores


pipeline_a_scores = score_pipeline_a(pipeline_a_output, agent1_result)
pipeline_a_scores


# %% Side-by-side comparison: Pipeline A output vs. Pipeline B output
def show_side_by_side(a_data: dict, b_data: dict) -> None:
    """Renders Pipeline A's output and Pipeline B's output as two columns.
    In VS Code's Interactive Window this displays as a real two-pane view;
    falls back to plain sequential printing outside a notebook/IPython context."""
    a_text = json.dumps(a_data, indent=2)
    b_text = json.dumps(b_data, indent=2)

    try:
        from IPython.display import HTML, display
        import html as html_lib

        display(HTML(f"""
        <div style="display:flex; gap:16px; font-family:monospace; font-size:12px;">
          <div style="flex:1; min-width:0;">
            <h3>Pipeline A -- Orphaned LLM</h3>
            <pre style="white-space:pre-wrap; background:#f5f5f5; color:#111; padding:8px;
                        border-radius:6px; max-height:600px; overflow:auto;">{html_lib.escape(a_text)}</pre>
          </div>
          <div style="flex:1; min-width:0;">
            <h3>Pipeline B -- Grounded Multi-Agent</h3>
            <pre style="white-space:pre-wrap; background:#f5f5f5; color:#111; padding:8px;
                        border-radius:6px; max-height:600px; overflow:auto;">{html_lib.escape(b_text)}</pre>
          </div>
        </div>
        """))
    except ImportError:
        print("=" * 30, "PIPELINE A", "=" * 30)
        print(a_text)
        print("=" * 30, "PIPELINE B", "=" * 30)
        print(b_text)


show_side_by_side(pipeline_a_output, pipeline_b_output)
