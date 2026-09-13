"""
Agent 4, MIMIC-IV client.

Service: none -- reads local PhysioNet-credentialed MIMIC-IV files (not a
live API). Expects the standard layout under MIMIC_DATA_DIR (.env):
`hosp/diagnoses_icd.csv.gz` and `hosp/d_icd_diagnoses.csv.gz` (only these
two files are needed, not the full dataset download).
Data: diagnoses_icd columns subject_id, icd_code, icd_version (str).
Read in 500k-row pandas chunks, never loaded whole.
Used by: PipelineB.agent4_mimic, to compute per-disease patient
(subject_id) sets for the pair-level co-occurrence check.
"""
import os
from typing import Dict, List, Set

import pandas as pd

DEFAULT_DIAGNOSES_RELATIVE_PATH = "hosp/diagnoses_icd.csv.gz"
DEFAULT_D_ICD_DIAGNOSES_RELATIVE_PATH = "hosp/d_icd_diagnoses.csv.gz"
_CHUNK_SIZE = 500_000


class MimicClient:
    def __init__(self, data_dir: str = None, chunk_size: int = _CHUNK_SIZE):
        self.data_dir = data_dir or os.environ.get("MIMIC_DATA_DIR")
        if not self.data_dir:
            raise RuntimeError(
                "MimicClient needs MIMIC_DATA_DIR set in .env -- the local folder containing "
                "MIMIC-IV's hosp/ directory (with diagnoses_icd.csv.gz and d_icd_diagnoses.csv.gz)."
            )
        self.diagnoses_path = os.path.join(self.data_dir, DEFAULT_DIAGNOSES_RELATIVE_PATH)
        self.chunk_size = chunk_size

    def compute_cohort_sets(self, disease_icd10_prefixes: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        # Input: disease name -> list of ICD-10 code prefixes (common/config.py's DISEASE_ICD10_PREFIXES).
        # Output: dict keyed by each disease name (subject_ids with >=1 matching icd_version==10
        # code), plus "__all__" (every subject_id with any diagnosis row at all, icd9 or icd10 --
        # the base population denominator).
        result_sets: Dict[str, Set[str]] = {name: set() for name in disease_icd10_prefixes}
        all_ids: Set[str] = set()

        for chunk in pd.read_csv(
            self.diagnoses_path,
            usecols=["subject_id", "icd_code", "icd_version"],
            dtype=str,
            chunksize=self.chunk_size,
        ):
            all_ids.update(chunk["subject_id"])
            chunk10 = chunk[chunk["icd_version"] == "10"]
            for name, prefixes in disease_icd10_prefixes.items():
                mask = chunk10["icd_code"].str.startswith(tuple(prefixes))
                result_sets[name].update(chunk10.loc[mask, "subject_id"])

        result_sets["__all__"] = all_ids
        return result_sets


class StubMimicClient:
    """Fixed, injectable cohort sets for offline tests -- no file I/O."""

    def __init__(self, cohort_sets: Dict[str, Set[str]]):
        self.cohort_sets = cohort_sets
        self.calls: List[dict] = []

    def compute_cohort_sets(self, disease_icd10_prefixes: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        self.calls.append(disease_icd10_prefixes)
        return self.cohort_sets
