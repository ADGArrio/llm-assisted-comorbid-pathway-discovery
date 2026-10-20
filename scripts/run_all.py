"""Run the whole study end to end for one or both disease pairs, saving the same files main.py saves.

    python -m scripts.run_all                      # both pairs, everything
    python -m scripts.run_all --pairs t2dm_ad      # one pair
    python -m scripts.run_all --skip-agent4        # no MIMIC-IV on this machine
    python -m scripts.run_all --skip-classifier    # skip the citation-evidence classifier
    python -m scripts.run_all --reuse-saved-agent1 # re-run only Agents 2-4 on the previous Agent 1 output
    python -m scripts.run_all --reuse-saved-agent1 --no-judge-evidence --out-suffix _noevidence   # ablation
    python -m scripts.run_all --reuse-saved-agent1 --with-metabolites --out-suffix _metab        # opt-in metabolite graph (default is protein-only)

Order per pair: Pipeline A -> Agent 1 -> Agent 2 -> Agent 3 -> Agent 4 -> save pipeline_b_<pair>.json
-> citation-evidence classifier (reads that file) -> edge-provenance audit. Any existing outputs are
first COPIED to outputs/archive_<timestamp>/ so nothing from the earlier run is lost.
A failure in one stage is printed and recorded, and the run carries on with what it can still do.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from common.config import DISEASE_PAIRS, get_pair
from PipelineA.pipeline_a import run_pipeline_a
from PipelineB.agent1_discovery import Agent1DiscoveryResult, run_agent1_discovery
from PipelineB.agent2_chain import NoChainFoundError, run_agent2_chain
from PipelineB.agent3_confirmation import DEFAULT_NULL_SIZE, run_agent3_confirmation
from PipelineB.agent4_mimic import run_agent4_mimic
from PipelineB.metalinks_client import MetalinksClient
from PipelineB.pubmed_client import PubMedClient

OUTPUT_DIR = Path("outputs")


def _archive(pair_id: str, stamp: str) -> None:
    # Copy (never move or delete) this pair's previous result files aside before overwriting them.
    dest = OUTPUT_DIR / f"archive_{stamp}"
    for name in (f"pipeline_a_{pair_id}.json", f"pipeline_b_{pair_id}.json"):
        src = OUTPUT_DIR / name
        if src.exists():
            dest.mkdir(exist_ok=True)
            shutil.copy2(src, dest / name)
            print(f"  archived {src} -> {dest / name}")
    cdir = OUTPUT_DIR / "citation_evidence"
    for src in (cdir / f"{pair_id}.json",):
        if src.exists():
            (dest / "citation_evidence").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / "citation_evidence" / src.name)


def _save(data, name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / name).write_text(json.dumps(data, indent=2))
    print(f"  saved outputs/{name}")


def _stage(name: str, fn, status: dict):
    t0 = time.time()
    print(f"[{name}] starting", flush=True)
    try:
        out = fn()
        status[name] = f"ok ({time.time() - t0:.0f}s)"
        return out
    except Exception as e:  # keep going; the failure is recorded in the run summary
        traceback.print_exc()
        status[name] = f"FAILED: {type(e).__name__}: {e}"
        return None


def run_pair(pair_id: str, args, stamp: str) -> dict:
    pair = get_pair(pair_id)
    d1, d2 = pair["disease1"], pair["disease2"]
    status: dict = {}
    print(f"\n===== {pair_id}: {d1} -> {d2} =====", flush=True)
    _archive(pair_id, stamp)

    saved_b = OUTPUT_DIR / f"pipeline_b_{pair_id}.json"
    reuse = args.reuse_saved_agent1 and saved_b.exists()
    if reuse:
        # Re-run only Agents 2-4 on the genes and citations of the previous Agent 1 run. Pipeline A and the
        # citation classifier are left alone, since they describe that same Agent 1 output.
        print(f"  reusing Agent 1 from {saved_b} (skipping Pipeline A and the classifier)")
        args = argparse.Namespace(**{**vars(args), "skip_classifier": True})
        status["pipeline_a"] = "kept previous"
        a1 = _stage("agent1", lambda: Agent1DiscoveryResult.from_dict(json.loads(saved_b.read_text())["agent1"]), status)
    else:
        a_res = _stage("pipeline_a", lambda: run_pipeline_a(d1, d2).to_dict(), status)
        if a_res:
            _save(a_res, f"pipeline_a_{pair_id}.json")
        a1 = _stage("agent1", lambda: run_agent1_discovery(d1, d2, min_score=args.min_score), status)
    a2 = a3 = None
    if a1:
        def _agent2():
            try:
                kw = {"max_hops": args.max_hops} if args.max_hops else {}
                # the full set of 2026-10-18 options: noise-control pass, edge strength + bridge-gene titles for the judge
                return run_agent2_chain(
                    a1, metalinks_client=MetalinksClient() if args.with_metabolites else None, stage1_control_pass=not args.no_stage1_control,
                    judge_evidence=not args.no_judge_evidence,
                    pubmed_client=None if args.no_judge_evidence else PubMedClient(), **kw)
            except NoChainFoundError as e:
                print(f"  Agent 2: {e}")
                return None
        a2 = _stage("agent2", _agent2, status)
        if a2:
            a3 = _stage("agent3", lambda: run_agent3_confirmation(a1, a2, null_baseline_size=0 if args.no_null_baseline else DEFAULT_NULL_SIZE), status)
    a4 = None if args.skip_agent4 else _stage("agent4", lambda: run_agent4_mimic(d1, d2), status)

    if a1:
        _save({
            "disease1": d1, "disease2": d2,
            "agent1": a1.to_dict(),
            "agent2": a2.to_dict() if a2 else None,
            "agent3": [c.to_dict() for c in (a3 or [])],
            "agent4": a4.to_dict() if a4 else None,
        }, f"pipeline_b_{pair_id}{args.out_suffix}.json")

        if not args.skip_classifier:
            t0 = time.time()
            print("[classifier] starting", flush=True)
            rc = subprocess.call([sys.executable, "-m", "scripts.classify_citations", "--pair", pair_id,
                                  "--sample-for-review", "50"])
            status["classifier"] = f"ok ({time.time() - t0:.0f}s)" if rc == 0 else f"FAILED (exit {rc})"
        if a2:
            rc = subprocess.call([sys.executable, "-m", "scripts.verify_edge_provenance", "--pair", pair_id,
                                  "--result", str(OUTPUT_DIR / f"pipeline_b_{pair_id}{args.out_suffix}.json"),
                                  "--out-dir", str(OUTPUT_DIR / f"provenance_audit{args.out_suffix}")])
            status["provenance_audit"] = "ok" if rc == 0 else f"FAILED (exit {rc})"
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", nargs="+", default=[p["id"] for p in DISEASE_PAIRS],
                    choices=[p["id"] for p in DISEASE_PAIRS])
    ap.add_argument("--min-score", type=float, default=0.1, help="Open Targets minimum association score (Agent 1)")
    ap.add_argument("--max-hops", type=int, default=None, help="Agent 2 hop limit (default: Agent 2's own default)")
    ap.add_argument("--reuse-saved-agent1", action="store_true",
                    help="re-run only Agents 2-4, reusing Agent 1 from outputs/pipeline_b_<pair>.json "
                         "(also skips Pipeline A and the classifier)")
    ap.add_argument("--with-metabolites", action="store_true",
                    help="Opt in to MetalinksDB gene<->metabolite hops in Agent 2 (the default search is protein-only; "
                         "the metabolite graph was evaluated and removed)")
    ap.add_argument("--no-metabolites", action="store_true", help="Deprecated no-op: protein-only is now the default")
    ap.add_argument("--no-judge-evidence", action="store_true",
                    help="ablation: do not show Agent 2's judge edge strength or bridge-gene PubMed titles")
    ap.add_argument("--no-null-baseline", action="store_true",
                    help="skip Agent 3's random-gene baseline (saves roughly 6-7 minutes of GWAS/GTEx requests per pair)")
    ap.add_argument("--no-stage1-control", action="store_true", help="skip the extra noise-control stage-1 pass")
    ap.add_argument("--out-suffix", default="",
                    help="append to the pipeline_b output file name, e.g. _noevidence, so an ablation run does not overwrite "
                         "the main result (Agent 1 is still read from the unsuffixed file with --reuse-saved-agent1)")
    ap.add_argument("--skip-agent4", action="store_true", help="skip the MIMIC-IV check")
    ap.add_argument("--skip-classifier", action="store_true", help="skip the citation-evidence classifier")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {pid: run_pair(pid, args, stamp) for pid in args.pairs}
    print("\n===== run summary =====")
    for pid, st in summary.items():
        for stage, s in st.items():
            print(f"{pid:9s} {stage:17s} {s}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / f"run_summary_{stamp}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
