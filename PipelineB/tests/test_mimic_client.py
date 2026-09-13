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
# subject 1: T2DM (E119) + CAD (I2510) -- both
# subject 2: T2DM only
# subject 3: AD only (G309)
# subject 4: icd9-only legacy code, no icd10 diagnosis at all -- still part of "__all__"
# subject 5: CAD only


@pytest.fixture
def mimic_data_dir(tmp_path):
    hosp_dir = tmp_path / "hosp"
    hosp_dir.mkdir()
    with gzip.open(hosp_dir / "diagnoses_icd.csv.gz", "wt") as f:
        f.write(DIAGNOSES_CSV)
    return str(tmp_path)


def test_compute_cohort_sets_from_a_real_small_gzipped_csv(mimic_data_dir):
    client = MimicClient(data_dir=mimic_data_dir, chunk_size=2)  # small chunk size to exercise chunking

    sets = client.compute_cohort_sets(
        {
            "Type 2 Diabetes Mellitus": ["E11"],
            "Coronary Artery Disease": ["I25"],
            "Alzheimer's Disease": ["G30"],
        }
    )

    assert sets["Type 2 Diabetes Mellitus"] == {"1", "2"}
    assert sets["Coronary Artery Disease"] == {"1", "5"}
    assert sets["Alzheimer's Disease"] == {"3"}
    assert sets["__all__"] == {"1", "2", "3", "4", "5"}  # subject 4's icd9-only row still counts


def test_missing_mimic_data_dir_raises_clear_error(monkeypatch):
    monkeypatch.delenv("MIMIC_DATA_DIR", raising=False)
    with pytest.raises(RuntimeError, match="MIMIC_DATA_DIR"):
        MimicClient(data_dir=None)


def test_stub_client_returns_configured_sets_without_file_io():
    stub = StubMimicClient({"__all__": {"1", "2"}, "Type 2 Diabetes Mellitus": {"1"}})

    result = stub.compute_cohort_sets({"Type 2 Diabetes Mellitus": ["E11"]})

    assert result == {"__all__": {"1", "2"}, "Type 2 Diabetes Mellitus": {"1"}}
    assert stub.calls == [{"Type 2 Diabetes Mellitus": ["E11"]}]
