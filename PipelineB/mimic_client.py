"""
Stage 4, MIMIC-IV client.

Per project_architecture.md, Stage 4 validates the disease *pair* (do
these two diagnoses actually co-occur more than chance in real patients),
not any individual chain or gene -- MIMIC-IV's diagnosis codes can't
confirm a specific protein-protein edge. Only two tables are needed:
`diagnoses_icd` (subject_id, icd_code, icd_version) and
`d_icd_diagnoses` (icd_code -> human-readable title, used for sanity
-checking the hand-mapped ICD prefixes in common/config.py, not for
the co-occurrence math itself).

This is read from local files, not a live API -- MIMIC-IV is
PhysioNet-credentialed data, queried locally per project_architecture.md.
Defaults to the standard PhysioNet MIMIC-IV directory layout:
`<MIMIC_DATA_DIR>/hosp/diagnoses_icd.csv.gz` and
`<MIMIC_DATA_DIR>/hosp/d_icd_diagnoses.csv.gz`. Set `MIMIC_DATA_DIR` in
`.env` to wherever these live on your machine.

Deliberately built to need only these two files, not the full MIMIC-IV
download: the full PhysioNet zip includes many modules (ICU chartevents,
labs, notes, waveform data, ...) this project doesn't touch, and a full
recursive `wget` mirror is known to be extremely slow for reasons
unrelated to bandwidth -- see the module docstring reasoning in
stage4_cohort_check.py for the practical download recommendation. This client
never asks for anything beyond the `hosp/` module's two diagnosis tables.

Reads `diagnoses_icd.csv.gz` in chunks (500k rows at a time) rather than
loading it whole -- it's several million rows across the full dataset,
and this project only needs three columns from it.

MIMIC-IV spans the US's October 2015 ICD-9-to-ICD-10 transition, so
`diagnoses_icd.csv.gz` genuinely mixes both coding systems in the same
table (distinguished by `icd_version`, "9" or "10" as strings). An
earlier version of `compute_cohort_sets` only ever matched icd_version
== "10", which silently dropped every ICD-9-coded patient -- confirmed
against the real MIMIC-IV data as a ~40-49% undercount per disease (see
common/config.py's DISEASE_ICD_PREFIXES comment and
the Stage 4 real-run write-up). `compute_cohort_sets` now takes a
version-aware prefix mapping (`{"9": [...], "10": [...]}` per disease)
and matches each version's rows against only that version's prefixes.
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

    def compute_cohort_sets(self, disease_icd_prefixes: Dict[str, Dict[str, List[str]]]) -> Dict[str, Set[str]]:
        """One pass over diagnoses_icd.csv.gz. `disease_icd_prefixes` is
        version-aware: {disease_name: {"9": [...], "10": [...]}}. Returns
        a dict keyed by each disease name (subject_ids with at least one
        code matching that disease's prefixes for that code's own
        icd_version), plus `"__all__"` (every subject_id appearing
        anywhere in the table, icd9 or icd10 -- the base population
        denominator: "has at least one recorded diagnosis of any kind,"
        not "has an icd10 diagnosis")."""
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


class StubMimicClient:
    """Fixed, injectable cohort sets for offline tests -- no file I/O."""

    def __init__(self, cohort_sets: Dict[str, Set[str]]):
        self.cohort_sets = cohort_sets
        self.calls: List[dict] = []

    def compute_cohort_sets(self, disease_icd_prefixes: Dict[str, Dict[str, List[str]]]) -> Dict[str, Set[str]]:
        self.calls.append(disease_icd_prefixes)
        return self.cohort_sets
