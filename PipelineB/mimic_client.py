"""Stage 4 MIMIC-IV client. Reads MIMIC-IV's diagnoses_icd.csv.gz from local files (set MIMIC_DATA_DIR in .env to the folder
containing hosp/) in chunks and builds the sets of patients carrying each disease's ICD-9 or ICD-10 codes.
"""
import os
from typing import Dict, List, Set

import pandas as pd

DEFAULT_DIAGNOSES_RELATIVE_PATH = "hosp/diagnoses_icd.csv.gz"
DEFAULT_D_ICD_DIAGNOSES_RELATIVE_PATH = "hosp/d_icd_diagnoses.csv.gz"
_CHUNK_SIZE = 500_000


# Reads the local MIMIC-IV diagnosis table.
class MimicClient:
    # Input: data directory (default MIMIC_DATA_DIR from .env) and chunk size; raises RuntimeError if no directory is
    # set. Output: none.
    def __init__(self, data_dir: str = None, chunk_size: int = _CHUNK_SIZE):
        self.data_dir = data_dir or os.environ.get("MIMIC_DATA_DIR")
        if not self.data_dir:
            raise RuntimeError(
                "MimicClient needs MIMIC_DATA_DIR set in .env -- the local folder containing "
                "MIMIC-IV's hosp/ directory (with diagnoses_icd.csv.gz and d_icd_diagnoses.csv.gz)."
            )
        self.diagnoses_path = os.path.join(self.data_dir, DEFAULT_DIAGNOSES_RELATIVE_PATH)
        self.chunk_size = chunk_size

    # Scan diagnoses_icd.csv.gz once. Input: {disease: {"9": [prefixes], "10": [prefixes]}}. Output: {disease: set of
    # subject_ids with a matching code of that code's own ICD version}, plus "__all__" (every subject_id in the table,
    # the population denominator).
    def compute_cohort_sets(self, disease_icd_prefixes: Dict[str, Dict[str, List[str]]]) -> Dict[str, Set[str]]:
        result_sets: Dict[str, Set[str]] = {name: set() for name in disease_icd_prefixes}
        all_ids: Set[str] = set()

        for chunk in pd.read_csv(
            self.diagnoses_path,
            usecols=["subject_id", "icd_code", "icd_version"],
            dtype=str,
            chunksize=self.chunk_size,
        ):
            all_ids.update(chunk["subject_id"])
            for name, version_prefixes in disease_icd_prefixes.items():
                for version, prefixes in version_prefixes.items():
                    if not prefixes:
                        continue
                    version_chunk = chunk[chunk["icd_version"] == version]
                    mask = version_chunk["icd_code"].str.startswith(tuple(prefixes))
                    result_sets[name].update(version_chunk.loc[mask, "subject_id"])

        result_sets["__all__"] = all_ids
        return result_sets


# Test double returning fixed cohort sets with no file access.
class StubMimicClient:

    # Input: the cohort sets to return. Output: none.
    def __init__(self, cohort_sets: Dict[str, Set[str]]):
        self.cohort_sets = cohort_sets
        self.calls: List[dict] = []

    # Record the call and return the fixed sets. Input: the ICD prefix mapping. Output: the cohort sets.
    def compute_cohort_sets(self, disease_icd_prefixes: Dict[str, Dict[str, List[str]]]) -> Dict[str, Set[str]]:
        self.calls.append(disease_icd_prefixes)
        return self.cohort_sets
