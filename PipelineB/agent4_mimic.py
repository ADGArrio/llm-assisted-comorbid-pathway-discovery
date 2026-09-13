"""
Agent 4 -- MIMIC-IV Cohort Check.

Service: none directly -- orchestrates mimic_client.py (local file reads).
Data: MimicClient.compute_cohort_sets() output (per-disease subject_id
sets); this module turns that into rate/relative-risk numbers.
Used by: the top-level pipeline, once per disease pair (not per chain --
MIMIC-IV diagnosis codes can't confirm a specific protein-protein edge,
only whether the two diagnoses co-occur in real patients).

This is a crude, unadjusted co-occurrence ratio: no age/sex matching, no
adjustment for shared risk factors, no causal claim -- descriptive
evidence for or against the comorbidity premise only.

Getting MIMIC-IV data locally: only two files are needed
(hosp/diagnoses_icd.csv.gz, hosp/d_icd_diagnoses.csv.gz). A recursive
`wget -r` mirror of the full PhysioNet project is known to take many
hours regardless of bandwidth, because it has to crawl directory index
pages for hundreds of thousands of files first (see
https://github.com/MIT-LCP/mimic-code/issues/600). Download just the two
files directly by URL instead (non-recursive), which takes minutes.
"""
import argparse
import json
from dataclasses import dataclass
from typing import Optional

from common.config import DISEASE_ICD10_PREFIXES, DISEASE_PAIRS, get_pair
from PipelineB.mimic_client import MimicClient


@dataclass
class MimicCohortResult:
    disease1: str
    disease2: str
    n_total_patients: int
    n_disease1_patients: int
    n_disease2_patients: int
    n_both_patients: int
    disease2_rate_in_disease1_cohort: float
    disease2_base_rate: float
    relative_risk: Optional[float]  # None if disease2_base_rate is 0 (undefined, not zero)

    def to_dict(self) -> dict:
        return {
            "disease1": self.disease1,
            "disease2": self.disease2,
            "n_total_patients": self.n_total_patients,
            "n_disease1_patients": self.n_disease1_patients,
            "n_disease2_patients": self.n_disease2_patients,
            "n_both_patients": self.n_both_patients,
            "disease2_rate_in_disease1_cohort": self.disease2_rate_in_disease1_cohort,
            "disease2_base_rate": self.disease2_base_rate,
            "relative_risk": self.relative_risk,
        }


def run_agent4_mimic(disease1: str, disease2: str, mimic_client: Optional[object] = None) -> MimicCohortResult:
    # Input: two disease names, optional client (defaults to real MimicClient(), needs
    # MIMIC_DATA_DIR in .env; pass StubMimicClient in tests). Output: MimicCohortResult.
    prefixes1 = DISEASE_ICD10_PREFIXES.get(disease1)
    prefixes2 = DISEASE_ICD10_PREFIXES.get(disease2)
    if not prefixes1:
        raise ValueError(f"No ICD-10 prefix mapping for {disease1!r} in common/config.py's DISEASE_ICD10_PREFIXES.")
    if not prefixes2:
        raise ValueError(f"No ICD-10 prefix mapping for {disease2!r} in common/config.py's DISEASE_ICD10_PREFIXES.")

    mimic_client = mimic_client or MimicClient()
    sets = mimic_client.compute_cohort_sets({disease1: prefixes1, disease2: prefixes2})

    all_ids = sets["__all__"]
    disease1_ids = sets[disease1]
    disease2_ids = sets[disease2]
    both_ids = disease1_ids & disease2_ids

    n_total = len(all_ids)
    n_disease1 = len(disease1_ids)
    n_disease2 = len(disease2_ids)
    n_both = len(both_ids)

    disease2_rate_in_disease1_cohort = (n_both / n_disease1) if n_disease1 else 0.0
    disease2_base_rate = (n_disease2 / n_total) if n_total else 0.0
    relative_risk = (disease2_rate_in_disease1_cohort / disease2_base_rate) if disease2_base_rate else None

    return MimicCohortResult(
        disease1=disease1,
        disease2=disease2,
        n_total_patients=n_total,
        n_disease1_patients=n_disease1,
        n_disease2_patients=n_disease2,
        n_both_patients=n_both,
        disease2_rate_in_disease1_cohort=disease2_rate_in_disease1_cohort,
        disease2_base_rate=disease2_base_rate,
        relative_risk=relative_risk,
    )


def _cli():
    parser = argparse.ArgumentParser(description="Run Agent 4 (MIMIC-IV Cohort Check) for one disease pair.")
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

    result = run_agent4_mimic(disease1, disease2)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()
