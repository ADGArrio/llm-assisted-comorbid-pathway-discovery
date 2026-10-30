"""Tests for the MIMIC-IV client (PipelineB/mimic_client.py) on a small gzipped diagnosis table."""

import gzip
import os

import pytest

from PipelineB.mimic_client import MimicClient, StubMimicClient

DIAGNOSES_CSV = """subject_id,hadm_id,seq_num,icd_code,icd_version
1,100,1,E119,10
1,100,2,I2510,10
2,101,1,E119,10
3,102,1,G309,10
4,103,1,25000,9
5,104,1,I2510,10
"""
# subject 1: T2DM (E119, ICD-10) + CAD (I2510, ICD-10) -- both
# subject 2: T2DM only (E119, ICD-10)
# subject 3: AD only (G309, ICD-10)
# subject 4: T2DM, but coded under the legacy ICD-9 system (25000, Type II unspecified) -- no
#            ICD-10 diagnosis at all; only counted for T2DM once ICD-9 prefixes are supplied
# subject 5: CAD only (I2510, ICD-10)


# Fixture: write a small diagnoses_icd.csv.gz under a temporary hosp/ folder and return the data directory.
@pytest.fixture
def mimic_data_dir(tmp_path):
    hosp_dir = tmp_path / "hosp"
    hosp_dir.mkdir()
    with gzip.open(hosp_dir / "diagnoses_icd.csv.gz", "wt") as f:
        f.write(DIAGNOSES_CSV)
    return str(tmp_path)


# Compute cohort sets from a real small gzipped csv.
def test_compute_cohort_sets_from_a_real_small_gzipped_csv(mimic_data_dir):
    client = MimicClient(data_dir=mimic_data_dir, chunk_size=2)  # small chunk size to exercise chunking

    sets = client.compute_cohort_sets(
        {
            "Type 2 Diabetes Mellitus": {"9": ["2500"], "10": ["E11"]},
            "Coronary Artery Disease": {"9": [], "10": ["I25"]},
            "Alzheimer's Disease": {"9": [], "10": ["G30"]},
        }
    )

    assert sets["Type 2 Diabetes Mellitus"] == {"1", "2", "4"}  # subject 4 now matched via ICD-9 "2500"
    assert sets["Coronary Artery Disease"] == {"1", "5"}
    assert sets["Alzheimer's Disease"] == {"3"}
    assert sets["__all__"] == {"1", "2", "3", "4", "5"}


# ICD-9 and ICD-10 prefixes are matched only against their own version.
def test_icd9_and_icd10_prefixes_are_matched_only_against_their_own_version(mimic_data_dir):
    # subject 4's code (25000, icd_version=9) must NOT match an icd_version=10 prefix list,
    # even if the prefix string would otherwise overlap -- versions are matched independently.
    client = MimicClient(data_dir=mimic_data_dir, chunk_size=2)

    sets = client.compute_cohort_sets({"Type 2 Diabetes Mellitus": {"9": [], "10": ["E11"]}})

    assert sets["Type 2 Diabetes Mellitus"] == {"1", "2"}  # subject 4 excluded: no icd9 prefixes supplied


# Missing MIMIC data dir raises clear error.
def test_missing_mimic_data_dir_raises_clear_error(monkeypatch):
    monkeypatch.delenv("MIMIC_DATA_DIR", raising=False)
    with pytest.raises(RuntimeError, match="MIMIC_DATA_DIR"):
        MimicClient(data_dir=None)


# Stub client returns configured sets without file io.
def test_stub_client_returns_configured_sets_without_file_io():
    stub = StubMimicClient({"__all__": {"1", "2"}, "Type 2 Diabetes Mellitus": {"1"}})

    result = stub.compute_cohort_sets({"Type 2 Diabetes Mellitus": {"9": [], "10": ["E11"]}})

    assert result == {"__all__": {"1", "2"}, "Type 2 Diabetes Mellitus": {"1"}}
    assert stub.calls == [{"Type 2 Diabetes Mellitus": {"9": [], "10": ["E11"]}}]
