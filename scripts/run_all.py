"""Run the whole study end to end for one or both disease pairs, saving the same files main.py saves.

    python -m scripts.run_all                      # both pairs, everything
    python -m scripts.run_all --pairs t2dm_ad      # one pair
    python -m scripts.run_all --skip-stage4        # no MIMIC-IV on this machine
    python -m scripts.run_all --skip-classifier    # skip the citation-evidence classifier
    python -m scripts.run_all --reuse-saved-stage1 # re-run only Stages 2-4 on the previous Stage 1 output
    python -m scripts.run_all --reuse-saved-stage1 --no-judge-evidence --out-suffix _noevidence   # ablation
    python -m scripts.run_all --reuse-saved-stage1 --with-metabolites --out-suffix _metab        # opt-in metabolite graph (default is protein-only)

Order per pair: Pipeline A -> Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> save pipeline_b_<pair>.json -> citation-evidence classifier
-> edge-provenance audit -> evidence-floor re-rank. Existing outputs are first copied to outputs/archive_<timestamp>/. A failure in one
step is printed and recorded, and the run carries on with what it can still do.
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
from PipelineB.stage1_discovery import Stage1DiscoveryResult, run_stage1_discovery
from PipelineB.stage2_judging import NoChainFoundError, run_stage2_judging
from PipelineB.stage3_confirmation import DEFAULT_NULL_SIZE, run_stage3_confirmation
from PipelineB.stage4_cohort_check import run_stage4_cohort_check
from PipelineB.metalinks_client import MetalinksClient
from PipelineB.pubmed_client import PubMedClient

OUTPUT_DIR = Path("outputs")


# Copy (never move or delete) a pair's previous result files into outputs/archive_<stamp>/ before they are overwritten.
# Input: pair id and timestamp. Output: none.
def _archive(pair_id: str, stamp: str) -> None:
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


# Write a dict as indented JSON under outputs/. Input: the data and a file name. Output: none.
def _save(data, name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / name).write_text(json.dumps(data, indent=2))
    print(f"  saved outputs/{name}")


# Run one step, timing it and recording "ok" or the failure in `status` instead of raising. Input: step name, a
# zero-argument function and the status dict. Output: the function's result, or None on failure.
def _run_step(name: str, fn, status: dict):
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


# Run every step for one disease pair and save the results. Input: pair id, the parsed arguments and the archive
# timestamp. Output: the pair's status dict.
def run_pair(pair_id: str, args, stamp: str) -> dict:
    pair = get_pair(pair_id)
    d1, d2 = pair["disease1"], pair["disease2"]
    status: dict = {}
    print(f"\n===== {pair_id}: {d1} -> {d2} =====", flush=True)
    _archive(pair_id, stamp)

    saved_b = OUTPUT_DIR / f"pipeline_b_{pair_id}.json"
    reuse = args.reuse_saved_stage1 and saved_b.exists()
    if reuse:
        # Re-run only Stages 2-4 on the genes and citations of the previous Stage 1 run; Pipeline A and the citation
        # classifier describe that same Stage 1 output and are left alone.
        print(f"  reusing Stage 1 from {saved_b} (skipping Pipeline A and the classifier)")
        args = argparse.Namespace(**{**vars(args), "skip_classifier": True})
        status["pipeline_a"] = "kept previous"
        a1 = _run_step("stage1", lambda: Stage1DiscoveryResult.from_dict(json.loads(saved_b.read_text())["stage1"]), status)
    else:
        a_res = _run_step("pipeline_a", lambda: run_pipeline_a(d1, d2).to_dict(), status)
        if a_res:
            _save(a_res, f"pipeline_a_{pair_id}.json")
        a1 = _run_step("stage1", lambda: run_stage1_discovery(d1, d2, min_score=args.min_score), status)
    a2 = a3 = None
    if a1:
        # Run Stage 2 with the full hardened options. Output: the Stage 2 result.
        def _stage2():
            try:
                kw = {"max_hops": args.max_hops} if args.max_hops else {}
                # Full hardened options: repeat judge pass as a noise control, and edge strength plus bridge-gene titles
                # shown to the judge.
                return run_stage2_judging(
                    a1, metalinks_client=MetalinksClient() if args.with_metabolites else None, judge_control_pass=not args.no_judge_control,
                    judge_evidence=not args.no_judge_evidence,
                    pubmed_client=None if args.no_judge_evidence else PubMedClient(), **kw)
            except NoChainFoundError as e:
                print(f"  Stage 2: {e}")
                return None
        a2 = _run_step("stage2", _stage2, status)
        if a2:
            a3 = _run_step("stage3", lambda: run_stage3_confirmation(a1, a2, null_baseline_size=0 if args.no_null_baseline else DEFAULT_NULL_SIZE), status)
    a4 = None if args.skip_stage4 else _run_step("stage4", lambda: run_stage4_cohort_check(d1, d2), status)

    if a1:
        _save({
            "disease1": d1, "disease2": d2,
            "stage1": a1.to_dict(),
            "stage2": a2.to_dict() if a2 else None,
            "stage3": [c.to_dict() for c in (a3 or [])],
            "stage4": a4.to_dict() if a4 else None,
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
        if a2 and a3:
            # Evidence floor: re-order Stage 2's chains by Stage 3's tissue result (code only, no LLM call).
            rc = subprocess.call([sys.executable, "-m", "PipelineB.evidence_rerank",
                                  "--result", str(OUTPUT_DIR / f"pipeline_b_{pair_id}{args.out_suffix}.json")])
            status["evidence_rerank"] = "ok" if rc == 0 else f"FAILED (exit {rc})"
    return status


# Command-line entry point: run each requested pair and print a per-step status summary.
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", nargs="+", default=[p["id"] for p in DISEASE_PAIRS],
                    choices=[p["id"] for p in DISEASE_PAIRS])
    ap.add_argument("--min-score", type=float, default=0.1, help="Open Targets minimum association score (Stage 1)")
    ap.add_argument("--max-hops", type=int, default=None, help="Stage 2 hop limit (default: Stage 2's own default)")
    ap.add_argument("--reuse-saved-stage1", action="store_true",
                    help="re-run only Stages 2-4, reusing Stage 1 from outputs/pipeline_b_<pair>.json "
                         "(also skips Pipeline A and the classifier)")
    ap.add_argument("--with-metabolites", action="store_true",
                    help="Opt in to MetalinksDB gene<->metabolite hops in Stage 2 (the default search is protein-only; "
                         "the metabolite graph was evaluated and removed)")
    ap.add_argument("--no-metabolites", action="store_true", help="Deprecated no-op: protein-only is now the default")
    ap.add_argument("--no-judge-evidence", action="store_true",
                    help="ablation: do not show Stage 2's judge edge strength or bridge-gene PubMed titles")
    ap.add_argument("--no-null-baseline", action="store_true",
                    help="skip Stage 3's random-gene baseline (saves roughly 6-7 minutes of GWAS/GTEx requests per pair)")
    ap.add_argument("--no-judge-control", action="store_true", help="skip the extra noise-control judge pass")
    ap.add_argument("--out-suffix", default="",
                    help="append to the pipeline_b output file name, e.g. _noevidence, so an ablation run does not overwrite "
                         "the main result (Stage 1 is still read from the unsuffixed file with --reuse-saved-stage1)")
    ap.add_argument("--skip-stage4", action="store_true", help="skip the MIMIC-IV check")
    ap.add_argument("--skip-classifier", action="store_true", help="skip the citation-evidence classifier")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {pid: run_pair(pid, args, stamp) for pid in args.pairs}
    print("\n===== run summary =====")
    for pid, st in summary.items():
        for step, s in st.items():
            print(f"{pid:9s} {step:17s} {s}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / f"run_summary_{stamp}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
