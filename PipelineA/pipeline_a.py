"""
Pipeline A -- Orphaned LLM (ungrounded baseline).

Service: LLMClient (common/llm_client.py) -- one call, no retrieval, no tools.
Data: two disease names in; free-text narrative out.
Used by: standalone CLI, and as the control arm compared against Pipeline B.

Usage:
    python -m PipelineA.pipeline_a --pair t2dm_ad
    python -m PipelineA.pipeline_a --disease1 "Type 2 Diabetes Mellitus" --disease2 "Alzheimer's Disease"

Metrics (BioPathBench: gene/protein extraction, protein recall,
hallucination rate) were purged 2026-09-14 to get the bare pipelines
running cleanly first -- see claude/metrics_purge.md. Pipeline A returns
just the raw narrative for now; extraction/scoring can be layered back on
once the metric design is settled.
"""
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional

from common.config import DISEASE_PAIRS, get_pair
from common.llm_client import LLMClient

PROMPT_TEMPLATE = """You are a biomedical researcher. Explain the most likely molecular \
mechanism connecting these two diseases:

Disease 1: {disease1}
Disease 2: {disease2}

Describe a plausible chain of genes and/or proteins linking Disease 1 to \
Disease 2, in the form Disease1 -> Protein1 -> Protein2 -> ... -> Disease2. \
For each step, name the specific gene or protein involved and state whether \
the interaction activates or inhibits the next step. Then give a short \
natural-language summary of the overall mechanism.

Use your own knowledge only. Do not say you need to look anything up."""


@dataclass
class PipelineAResult:
    disease1: str
    disease2: str
    narrative: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_pipeline_a(disease1: str, disease2: str, client: Optional[object] = None) -> PipelineAResult:
    # Input: two disease names, optional LLM client (defaults to real LLMClient(); pass StubLLMClient in tests).
    # Output: PipelineAResult with the raw narrative.
    client = client or LLMClient()
    prompt = PROMPT_TEMPLATE.format(disease1=disease1, disease2=disease2)
    narrative = client.generate(prompt, temperature=0.3)
    return PipelineAResult(disease1=disease1, disease2=disease2, narrative=narrative)


def _cli():
    parser = argparse.ArgumentParser(description="Run Pipeline A (Orphaned LLM) for one disease pair.")
    parser.add_argument("--pair", choices=[p["id"] for p in DISEASE_PAIRS], help="Disease pair id from common/config.py")
    parser.add_argument("--disease1", help="Override: disease 1 name")
    parser.add_argument("--disease2", help="Override: disease 2 name")
    args = parser.parse_args()

    if args.pair:
        pair = get_pair(args.pair)
        disease1, disease2 = pair["disease1"], pair["disease2"]
    elif args.disease1 and args.disease2:
        disease1, disease2 = args.disease1, args.disease2
    else:
        parser.error("Pass --pair, or both --disease1 and --disease2.")

    result = run_pipeline_a(disease1, disease2)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()
