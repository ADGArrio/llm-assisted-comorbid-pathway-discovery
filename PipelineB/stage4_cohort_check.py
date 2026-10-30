"""Stage 4 Cohort Check (MIMIC-IV): checks whether the two diseases co-occur in real patients, once per disease pair.
Turns MimicClient's per-disease patient sets into the risk ratio P(d2|d1) / P(d2|not d1) with a 95% CI and chi-square
p-value, plus an odds ratio and the observed/expected lift. The ratios are unadjusted (no age, sex or risk-factor matching)
and descriptive only.

Only two files are needed locally (hosp/diagnoses_icd.csv.gz and hosp/d_icd_diagnoses.csv.gz); download them directly by
URL rather than mirroring the whole PhysioNet project.
"""
import argparse
import json
import math
from dataclasses import dataclass
from typing import Optional

from common.config import DISEASE_ICD_PREFIXES, DISEASE_PAIRS, get_pair
from PipelineB.mimic_client import MimicClient


# Stage 4 output: cohort sizes, the disease-2 rate in the disease-1 cohort, and the association statistics.
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
    relative_risk: Optional[float]  # TRUE risk ratio: P(d2|d1) / P(d2|not d1); None if undefined
    relative_risk_ci95: Optional[tuple] = None
    odds_ratio: Optional[float] = None
    odds_ratio_ci95: Optional[tuple] = None
    p_value: Optional[float] = None  # chi-square test (1 df, no continuity correction) of d1 x d2 independence
    observed_over_expected: Optional[float] = None  # P(d2|d1) / P(d2 overall) -- the old "relative_risk" (a lift)

    # Output: the result as a JSON-serializable dict.
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
            "relative_risk_ci95": list(self.relative_risk_ci95) if self.relative_risk_ci95 else None,
            "odds_ratio": self.odds_ratio,
            "odds_ratio_ci95": list(self.odds_ratio_ci95) if self.odds_ratio_ci95 else None,
            "p_value": self.p_value,
            "observed_over_expected": self.observed_over_expected,
        }


# Compute association statistics from a 2x2 table. Input: counts (everyone, disease 1, disease 2, both). Output: risk
# ratio and odds ratio with 95% CIs and a chi-square p-value; each is None when its cells make it undefined.
def _two_by_two_stats(n_total: int, n1: int, n2: int, n_both: int) -> dict:
    a = n_both                    # d1 and d2
    b = n1 - n_both               # d1 only
    c = n2 - n_both               # d2 only
    d = n_total - n1 - n2 + n_both  # neither
    out = {"relative_risk": None, "relative_risk_ci95": None, "odds_ratio": None, "odds_ratio_ci95": None, "p_value": None}
    if min(a, b, c, d) < 0:
        return out  # inconsistent counts -- do not invent numbers
    exposed, unexposed = a + b, c + d
    if a > 0 and c > 0 and exposed > 0 and unexposed > 0:
        rr = (a / exposed) / (c / unexposed)
        se = math.sqrt(1 / a - 1 / exposed + 1 / c - 1 / unexposed)
        out["relative_risk"] = rr
        out["relative_risk_ci95"] = (rr * math.exp(-1.96 * se), rr * math.exp(1.96 * se))
    if min(a, b, c, d) > 0:
        orr = (a * d) / (b * c)
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        out["odds_ratio"] = orr
        out["odds_ratio_ci95"] = (orr * math.exp(-1.96 * se), orr * math.exp(1.96 * se))
    row1, row2, col1, col2 = a + b, c + d, a + c, b + d
    if min(row1, row2, col1, col2) > 0:
        chi2 = n_total * (a * d - b * c) ** 2 / (row1 * row2 * col1 * col2)
        out["p_value"] = math.erfc(math.sqrt(chi2 / 2))  # survival function of chi-square with 1 df
    return out


# Run Stage 4. Input: two disease names and an optional client (defaults to MimicClient(), which needs MIMIC_DATA_DIR in
# .env; tests pass a stub). Output: a MimicCohortResult.
def run_stage4_cohort_check(disease1: str, disease2: str, mimic_client: Optional[object] = None) -> MimicCohortResult:
    prefixes1 = DISEASE_ICD_PREFIXES.get(disease1)
    prefixes2 = DISEASE_ICD_PREFIXES.get(disease2)
    if not prefixes1:
        raise ValueError(f"No ICD prefix mapping for {disease1!r} in common/config.py's DISEASE_ICD_PREFIXES.")
    if not prefixes2:
        raise ValueError(f"No ICD prefix mapping for {disease2!r} in common/config.py's DISEASE_ICD_PREFIXES.")

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
    observed_over_expected = (disease2_rate_in_disease1_cohort / disease2_base_rate) if disease2_base_rate else None
    stats = _two_by_two_stats(n_total, n_disease1, n_disease2, n_both)

    return MimicCohortResult(
        disease1=disease1,
        disease2=disease2,
        n_total_patients=n_total,
        n_disease1_patients=n_disease1,
        n_disease2_patients=n_disease2,
        n_both_patients=n_both,
        disease2_rate_in_disease1_cohort=disease2_rate_in_disease1_cohort,
        disease2_base_rate=disease2_base_rate,
        observed_over_expected=observed_over_expected,
        **stats,
    )


# Command-line entry point: run the stage for a disease pair chosen by id or by name and print the result as JSON.
def _cli():
    parser = argparse.ArgumentParser(description="Run Stage 4 (Cohort Check) for one disease pair.")
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

    result = run_stage4_cohort_check(disease1, disease2)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()
