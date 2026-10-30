"""Multi-model comparison on the HARDENED pipeline.

Runs Stages 2 and 3 and the evidence floor once per run, per model, per disease pair, on the same Stage 1 output
(the one saved in outputs/pipeline_b_<pair>.json, so every model sees the candidates the final gemma4 runs saw).
Stage 2 uses the full hardened settings: two-order judge plus the repeat (noise-control) pass, edge strength and
bridge-gene PubMed titles shown to the judge, and a two-order debate. Stage 3 runs without the random-gene baseline
(that baseline does not depend on the language model). The evidence floor is applied to every run.

MUST be run on the machine where Ollama runs: the language-model calls go to LLM_BASE_URL (default
http://localhost:11434), which the cloud sandbox and the device-bridge shell cannot reach.

Each model is first copied to a 16,384-token-context version (`<tag>-16k`, via `ollama create`), because the default
context truncates judge replies on the larger prompts (the same reason the final gemma4 runs use gemma4-16k).

    python3 -m scripts.compare_llm_models                                   # all three models, 3 runs each, both pairs
    python3 -m scripts.compare_llm_models --pairs t2dm_ad --models qwen3:14b --runs 1      # quick test

Gemma4 is included by default so that all three models are run under exactly the same harness and settings.

Output, in outputs/model_comparison_hardened/:
    <pair>__<model>__run<k>.json     one saved pipeline_b-style result per run (stage1, stage2, stage3)
    summary.json                     one row per (pair, model) and one row per run
    multimodel_table.tex             LaTeX table fragment the thesis \\input's
    run_log.txt                      printed summary
"""
import argparse
import json
import re
import statistics
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from PipelineB.stage1_discovery import Stage1DiscoveryResult, run_stage1_discovery
from PipelineB.stage2_judging import NoChainFoundError, run_stage2_judging
from PipelineB.stage3_confirmation import run_stage3_confirmation
from PipelineB.evidence_rerank import chain_label, rerank_saved
from PipelineB.pubmed_client import PubMedClient
from common.config import get_pair
from common.llm_client import LLMClient

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "model_comparison_hardened"
DEFAULT_MODELS = "gemma4:12b,qwen3:14b,deepseek-r1:14b"
PAIRS = ["t2dm_ad", "t2dm_cad"]
PAIR_LABEL = {"t2dm_ad": "T2DM to AD", "t2dm_cad": "T2DM to CAD"}


def _safe_tag(model: str) -> str:
    # Input: a model tag such as "deepseek-r1:14b". Output: a filesystem-safe version.
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model)


def make_16k_copy(model: str, ctx: int = 16384) -> str:
    # Input: an Ollama model tag. Output: the tag of a copy with a `ctx`-token context window (created if missing).
    name = f"{model.replace(':', '-')}-{ctx // 1024}k"
    with tempfile.TemporaryDirectory() as d:
        mf = Path(d) / "Modelfile"
        mf.write_text(f"FROM {model}\nPARAMETER num_ctx {ctx}\n")
        r = subprocess.run(["ollama", "create", name, "-f", str(mf)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ollama create {name} failed: {r.stderr.strip()}")
    return name


def unload(model: str) -> None:
    # Free the model's memory before the next one loads; a failure here is only a warning.
    try:
        subprocess.run(["ollama", "stop", model], check=False, capture_output=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  (warning: could not unload {model}: {e})")


def load_stage1(pair_id: str, disease1: str, disease2: str) -> Stage1DiscoveryResult:
    # The saved Stage 1 output, so every model judges the same candidates; Stage 1 is run only if none is saved.
    saved = ROOT / "outputs" / f"pipeline_b_{pair_id}.json"
    if saved.exists():
        return Stage1DiscoveryResult.from_dict(json.loads(saved.read_text())["stage1"])
    print(f"  no saved Stage 1 for {pair_id}; running it")
    return run_stage1_discovery(disease1, disease2)


def summarise_run(result: dict) -> dict:
    # Input: one pipeline_b-style result dict. Output: the numbers the comparison reports for that run.
    a2 = result.get("stage2")
    row = {"endorsed": 0, "candidates": 0, "fwd": None, "rev": None, "rpt": None, "disagree_order": None,
           "disagree_noise": None, "top_label": None, "floor_top_label": None, "chains": 0, "chains_pass_floor": 0,
           "abstains_after_floor": False, "top_passes_floor": None}
    if not a2:
        return row
    stats = a2.get("bfs_stats", {})
    chains = a2.get("chains", [])
    row.update({
        "endorsed": len(chains),
        "candidates": a2.get("candidate_path_count", 0),
        "fwd": stats.get("judge_endorsed_forward"),
        "rev": stats.get("judge_endorsed_reversed"),
        "rpt": stats.get("judge_endorsed_forward_repeat"),
        "disagree_order": stats.get("judge_order_disagreements"),
        "disagree_noise": stats.get("judge_noise_disagreements"),
        "chains": len(chains),
        "top_label": chain_label(chains[0]) if chains else None,
    })
    rr = rerank_saved(result)
    if rr:
        row["chains_pass_floor"] = rr["n_pass"]
        row["abstains_after_floor"] = rr["abstains"]
        row["top_passes_floor"] = rr["top_original_passes"]
        if rr["n_pass"]:
            row["floor_top_label"] = rr["chains"][0]["label"]
    return row


def summarise_model(rows: list) -> dict:
    # Input: the per-run rows for one (pair, model). Output: stability numbers across those runs.
    n = len(rows)
    endorsed = [r["endorsed"] for r in rows]
    tops = [r["top_label"] or "(none)" for r in rows]
    floor_tops = [r["floor_top_label"] or "(none)" for r in rows]
    modal = Counter(tops).most_common(1)[0] if rows else ("(none)", 0)
    with_chain = [r for r in rows if r["chains"]]
    return {
        "runs": n,
        "endorsed_min": min(endorsed) if endorsed else None,
        "endorsed_max": max(endorsed) if endorsed else None,
        "endorsed_median": statistics.median(endorsed) if endorsed else None,
        "empty_runs": sum(1 for e in endorsed if e == 0),
        "distinct_top": len(set(t for t in tops if t != "(none)")),
        "modal_top": modal[0],
        "modal_share": round(modal[1] / n, 2) if n else None,
        "distinct_top_after_floor": len(set(floor_tops)),
        "top_fails_floor": sum(1 for r in with_chain if r["top_passes_floor"] is False),
        "runs_with_chain": len(with_chain),
        "abstain_after_floor": sum(1 for r in rows if r["chains"] and r["abstains_after_floor"]),
    }



RUN_FILE_RE = re.compile(r"^(?P<pair>t2dm_[a-z]+)__(?P<tag>.+)__run(?P<k>\d+)\.json$")
FAIL_FILE_RE = re.compile(r"^(?P<pair>t2dm_[a-z]+)__(?P<tag>.+)__run(?P<k>\d+)\.FAILED\.txt$")


def _model_from_tag(tag: str) -> str:
    # "deepseek-r1_14b" -> "deepseek-r1:14b" (the tag is the model with ':' replaced by '_')
    return tag.rsplit("_", 1)[0] + ":" + tag.rsplit("_", 1)[1] if "_" in tag else tag


def collect_rows_from_disk() -> dict:
    # Rebuilds the per-run rows from every saved run file in the output folder, so that results from separate
    # invocations (one pair at a time, or one model at a time) are combined. Failed runs are listed separately.
    rows_by, failed_by = {}, {}
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        m = RUN_FILE_RE.match(f.name)
        if not m:
            continue
        res = json.loads(f.read_text())
        model = res.get("model") or _model_from_tag(m["tag"])
        row = summarise_run(res)
        row.update({"pair": m["pair"], "model": model, "run": int(m["k"])})
        rows_by.setdefault((m["pair"], model), []).append(row)
    for f in sorted(OUTPUT_DIR.glob("*.FAILED.txt")):
        m = FAIL_FILE_RE.match(f.name)
        if m:
            failed_by.setdefault((m["pair"], _model_from_tag(m["tag"])), []).append(f.read_text().strip())
    return rows_by, failed_by


def latex_table(per_model: dict, failed: dict = None) -> str:
    # Input: {(pair, model): summary}. Output: a LaTeX table fragment (tabular inside a table environment).
    lines = [
        r"\begin{table}[h]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Hardened pipeline run with each language model on the same Stage~1 output. ``Endorsed'' is the number of "
        r"the ten candidates the judge endorsed, as minimum to maximum over completed runs. ``Empty'' counts completed runs with no "
        r"endorsed chain. ``Failed'' counts runs that did not complete (for example a timeout); they are not counted as empty. "
        r"``Top chains'' counts the different top chains over completed runs that had a chain. ``Top fails floor'' counts runs "
        r"whose debate-top chain failed the evidence floor.}",
        r"\label{tab:multimodel-hardened}",
        r"\begin{tabular}{|>{\raggedright\arraybackslash}p{0.15\linewidth}|>{\raggedright\arraybackslash}p{0.19\linewidth}"
        r"|>{\centering\arraybackslash}p{0.07\linewidth}|>{\centering\arraybackslash}p{0.08\linewidth}"
        r"|>{\centering\arraybackslash}p{0.11\linewidth}|>{\centering\arraybackslash}p{0.07\linewidth}"
        r"|>{\centering\arraybackslash}p{0.10\linewidth}|>{\centering\arraybackslash}p{0.10\linewidth}|}", r"\hline",
        r"\textbf{Pair} & \textbf{Model} & \textbf{Runs} & \textbf{Failed} & \textbf{Endorsed} & \textbf{Empty} & "
        r"\textbf{Top chains} & \textbf{Top fails floor} \\", r"\hline",
    ]
    failed = failed or {}
    for (pair, model), s in per_model.items():
        nf = len(failed.get((pair, model), []))
        end = f"{s['endorsed_min']}" if s['endorsed_min'] == s['endorsed_max'] else f"{s['endorsed_min']} to {s['endorsed_max']}"
        lines.append(
            f"{PAIR_LABEL[pair]} & \\texttt{{{model}}} & {s['runs']} & {nf} & {end} & "
            f"{s['empty_runs']} & {s['distinct_top'] if s['distinct_top'] else 'n/a'} & {s['top_fails_floor']} of {s['runs_with_chain']} \\\\")
        lines.append(r"\hline")
    for (pair, model), msgs in failed.items():
        if (pair, model) not in per_model:
            lines.append(f"{PAIR_LABEL[pair]} & \\texttt{{{model}}} & 0 & {len(msgs)} & n/a & n/a & n/a & n/a \\\\")
            lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def run_once(a1, model_name: str, pair_id: str, tag: str, k: int, timeout: int = 300, label: str = "") -> dict:
    # One hardened Stage 2 + Stage 3 run with `model_name`. Saves and returns the pipeline_b-style result dict.
    client = LLMClient(model=model_name, timeout=timeout)
    try:
        a2 = run_stage2_judging(a1, metalinks_client=None, llm_client=client, judge_control_pass=True,
                              judge_evidence=True, pubmed_client=PubMedClient())
    except NoChainFoundError as e:
        print(f"    Stage 2: {e}")
        a2 = None
    a3 = run_stage3_confirmation(a1, a2, null_baseline_size=0) if a2 else []
    result = {"stage1": a1.to_dict(), "stage2": a2.to_dict() if a2 else None, "stage3": [c.to_dict() for c in a3],
              "llm_model": model_name, "model": label or model_name}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{pair_id}__{tag}__run{k}.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", nargs="+", default=PAIRS, choices=PAIRS)
    ap.add_argument("--models", default=DEFAULT_MODELS, help="comma-separated Ollama tags")
    ap.add_argument("--runs", type=int, default=3, help="repeat runs per model per pair (default 3)")
    ap.add_argument("--ctx", type=int, default=16384, help="context window of the model copy (default 16384)")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="seconds to wait for one model reply (default 1800; reasoning models such as deepseek-r1 are slow)")
    ap.add_argument("--start-run", type=int, default=1,
                    help="first run number to execute (default 1); use 2 to keep an existing run 1 and only fill in runs 2..--runs")
    ap.add_argument("--summary-only", action="store_true", help="rebuild the summary and table from saved runs and exit")
    ap.add_argument("--no-16k-copy", action="store_true", help="use each model as is (not recommended)")
    args = ap.parse_args()
    if args.summary_only:
        write_summary()
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pair_id in args.pairs:
        pair = get_pair(pair_id)
        print(f"\n===== {pair_id} =====", flush=True)
        a1 = load_stage1(pair_id, pair["disease1"], pair["disease2"])
        for model in models:
            name = model if args.no_16k_copy else make_16k_copy(model, args.ctx)
            tag = _safe_tag(model)
            for k in range(args.start_run, args.runs + 1):
                print(f"  [{pair_id} :: {model} :: run {k}/{args.runs}]", flush=True)
                fail_file = OUTPUT_DIR / f"{pair_id}__{tag}__run{k}.FAILED.txt"
                try:
                    res = run_once(a1, name, pair_id, tag, k, timeout=args.timeout, label=model)
                    row = summarise_run(res)
                    if fail_file.exists():
                        fail_file.unlink()  # a later successful run replaces the earlier failure record
                    print(f"    endorsed {row['endorsed']}/{row['candidates']} (fwd {row['fwd']}, rev {row['rev']}, rpt {row['rpt']}); "
                          f"top: {row['top_label']}; passes floor: {row['top_passes_floor']}", flush=True)
                except Exception as e:  # one failed run must not lose the others, and is never counted as an empty run
                    print(f"    FAILED: {type(e).__name__}: {e}")
                    fail_file.write_text(f"{type(e).__name__}: {e}\n")
            if not args.no_16k_copy:
                unload(name)

    write_summary()


def write_summary() -> None:
    # Combines every saved run in the output folder (including those from earlier invocations) into the summary files.
    rows_by, failed_by = collect_rows_from_disk()
    per_model = {k: summarise_model(v) for k, v in sorted(rows_by.items())}
    all_rows = [r for v in rows_by.values() for r in v]
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(
        {"per_model": {f"{p}|{m}": s for (p, m), s in per_model.items()},
         "failed_runs": {f"{p}|{m}": v for (p, m), v in failed_by.items()}, "runs": all_rows}, indent=2))
    (OUTPUT_DIR / "multimodel_table.tex").write_text(latex_table(per_model, failed_by))
    out = ["", "=== Summary ==="]
    for (p, m), s in per_model.items():
        out.append(f"{p:9s} {m:18s} runs={s['runs']} failed={len(failed_by.get((p, m), []))} "
                   f"endorsed {s['endorsed_min']}-{s['endorsed_max']} empty={s['empty_runs']} "
                   f"distinct_top={s['distinct_top']} modal='{s['modal_top']}' ({s['modal_share']}) "
                   f"top_fails_floor={s['top_fails_floor']}/{s['runs_with_chain']}")
    for (p, m), v in failed_by.items():
        if (p, m) not in per_model:
            out.append(f"{p:9s} {m:18s} runs=0 failed={len(v)}  ({v[0][:90]})")
    (OUTPUT_DIR / "run_log.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nsaved -> {OUTPUT_DIR}")


def summary_only() -> None:
    write_summary()


if __name__ == "__main__":
    main()
