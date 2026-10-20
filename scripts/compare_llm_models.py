"""
Multi-model comparison harness -- Action item 1 from committee feedback round 2
("Have multiple models to compare outputs" / "compare singular model outputs
Gemma vs. DeepSeek vs. Qwen, etc."). See
claude/committee_feedback_round2_coscientist.md section 4 for the full writeup.

MUST be run on the user's own machine, not through the Claude device bridge:
both LLM calls here go to a local Ollama server at LLM_BASE_URL (default
http://localhost:11434), and that host is only reachable from the machine
Ollama is actually running on -- the cloud sandbox and the device-bridge shell
used to edit this repo cannot reach it (see claude/agent2_bfs_and_prompt_recalibration.md
and this project's other real-run docs for the same limitation).

Before running, pull each candidate model with Ollama, e.g.:
    ollama pull qwen3:14b
    ollama pull deepseek-r1:14b
(Model tag names on Ollama's library change over time -- confirm the exact tag
with `ollama list`/`ollama pull <name>` yourself; these are the two recommended
in claude/committee_feedback_round2_coscientist.md as of 2026-10-04, chosen to
be roughly comparable in size to the existing default, gemma4:12b.)

What this measures, concretely, per model per disease pair:
  - Agent 2 stage-1 endorsement rate: of the BFS-found structural candidates,
    how many did this model's independent PLAUSIBLE/NOT-PLAUSIBLE pass endorse.
  - The debate tournament's winning chain and its Elo rating (if >=2 candidates
    were endorsed).
  - Whether different models endorse the SAME underlying chain (agreement) or
    different ones -- a stricter/looser model may accept more or fewer of the
    same real candidates, and two models may disagree about which one is best.
  - Pipeline A's narrative for the same model, saved alongside for a qualitative
    read (no automated narrative-quality score yet -- see possible_metrics.md).

This script does NOT re-run Agent 1 (Open Targets/PubMed) or Agent 3
(GTEx/GWAS/PubMed-PMC) per model -- neither uses the LLM, so Agent 1 runs once
per disease pair and is reused across every model; Agent 3 is left for the user
to run separately per model's winning chain if they want full per-model
confirmation data (keeps this script's own runtime and API-rate-limit exposure
down, since Agent 1/3 are the parts bound by external-service rate limits, not
by which LLM is being compared).

Usage:
    python3 -m scripts.compare_llm_models --pair t2dm_cad --models gemma4:12b,qwen3:14b,deepseek-r1:14b
    python3 -m scripts.compare_llm_models --pair t2dm_ad --models gemma4:12b,qwen3:14b,deepseek-r1:14b

Output: outputs/model_comparison/{pair_id}__{model_tag}.json per model, plus a
printed summary table, plus outputs/model_comparison/{pair_id}__summary.json.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

from PipelineA.pipeline_a import run_pipeline_a
from PipelineB.agent1_discovery import run_agent1_discovery
from PipelineB.agent2_chain import NoChainFoundError, run_agent2_chain
from PipelineB.metalinks_client import MetalinksClient
from common.config import get_pair
from common.llm_client import LLMClient

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "model_comparison"


def _safe_tag(model: str) -> str:
    # Input: a raw model tag (e.g. "deepseek-r1:14b"). Output: a filesystem-safe version.
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model)


def _unload_model(model: str) -> None:
    # Ollama's own default keep_alive (5 minutes idle) can leave a model resident in memory
    # while the next one loads -- across three 12-14B models run back to back, that can stack
    # up to more RAM/VRAM than the machine actually has, rather than ever holding just one at a
    # time. `ollama stop` is the supported, explicit way to force an immediate unload; call it
    # right after a model's work is done and before the next model starts. If the `ollama` CLI
    # isn't on PATH or the call fails for any reason, this is a soft warning, not a fatal error --
    # the comparison should still proceed even if a stray model stays loaded.
    try:
        subprocess.run(["ollama", "stop", model], check=False, capture_output=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  (warning: couldn't unload {model} via 'ollama stop': {e})")


def run_one_model(pair_id: str, disease1: str, disease2: str, model: str, agent1_result) -> dict:
    # Input: one model tag + Agent 1's result (shared across models, see module docstring).
    # Output: a dict summarizing that model's Pipeline A narrative and Agent 2 chain output.
    print(f"\n=== {pair_id} :: {model} ===")
    client = LLMClient(model=model)

    pipeline_a_result = run_pipeline_a(disease1, disease2, client=client)

    metalinks_client = None  # protein-only: the metabolite graph was evaluated and removed from the default search
    try:
        agent2_result = run_agent2_chain(
            agent1_result, metalinks_client=metalinks_client, llm_client=client
        )
    except NoChainFoundError as e:
        print(f"  Agent 2: {e}")
        agent2_result = None

    record = {
        "pair_id": pair_id,
        "model": model,
        "pipeline_a_narrative": pipeline_a_result.narrative,
        "agent2_candidate_path_count": agent2_result.candidate_path_count if agent2_result else 0,
        "agent2_endorsed_count": len(agent2_result.chains) if agent2_result else 0,
        "agent2_rejection_reason": agent2_result.rejection_reason if agent2_result else None,
        "agent2_chains": [c.to_dict() for c in agent2_result.chains] if agent2_result else [],
        "agent2_debate_log": [d.to_dict() for d in agent2_result.debate_log] if agent2_result else [],
    }

    out_path = OUTPUT_DIR / f"{pair_id}__{_safe_tag(model)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(f"  endorsed {record['agent2_endorsed_count']}/{record['agent2_candidate_path_count']} candidates")
    print(f"  saved -> {out_path}")
    return record


def _top_chain_signature(record: dict) -> str:
    # Input: one model's result record. Output: a short string identifying its top (highest-Elo,
    # or first if no tournament ran) endorsed chain's gene sequence -- used to compare whether two
    # models landed on the "same" mechanism, not just the same endorsement count.
    chains = record["agent2_chains"]
    if not chains:
        return "(none endorsed)"
    top = max(chains, key=lambda c: c.get("elo_rating", 0.0))
    genes = [top["edges"][0]["source_gene"]] + [e["target_gene"] for e in top["edges"]]
    return " -> ".join(genes)


def summarize(records: list) -> dict:
    # Input: one record per model for one disease pair. Output: a cross-model summary --
    # endorsement rates, top-chain signatures, and a simple agreement count (how many distinct
    # models landed on the exact same gene sequence as their top chain).
    signatures = {r["model"]: _top_chain_signature(r) for r in records}
    distinct_signatures = set(signatures.values())
    return {
        "pair_id": records[0]["pair_id"] if records else None,
        "per_model": {
            r["model"]: {
                "endorsed": r["agent2_endorsed_count"],
                "candidates": r["agent2_candidate_path_count"],
                "top_chain": signatures[r["model"]],
            }
            for r in records
        },
        "distinct_top_chains": len(distinct_signatures),
        "models_agree_on_top_chain": len(distinct_signatures) <= 1 and "(none endorsed)" not in distinct_signatures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", required=True, choices=["t2dm_ad", "t2dm_cad"])
    parser.add_argument(
        "--models", required=True,
        help="Comma-separated Ollama model tags, e.g. gemma4:12b,qwen3:14b,deepseek-r1:14b",
    )
    args = parser.parse_args()

    pair = get_pair(args.pair)
    disease1, disease2 = pair["disease1"], pair["disease2"]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"Running Agent 1 once for {disease1} -> {disease2} (shared across all models)...")
    agent1_result = run_agent1_discovery(disease1, disease2)

    records = []
    for model in models:
        records.append(run_one_model(args.pair, disease1, disease2, model, agent1_result))
        _unload_model(model)  # free this model's memory before the next one loads

    summary = summarize(records)
    summary_path = OUTPUT_DIR / f"{args.pair}__summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== Summary ({args.pair}) ===")
    for model, info in summary["per_model"].items():
        print(f"  {model}: endorsed {info['endorsed']}/{info['candidates']} -- top chain: {info['top_chain']}")
    print(f"  distinct top chains across models: {summary['distinct_top_chains']}")
    print(f"  saved -> {summary_path}")


if __name__ == "__main__":
    main()
