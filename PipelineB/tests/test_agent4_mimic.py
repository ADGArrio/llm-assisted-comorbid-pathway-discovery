import pytest

from PipelineB.agent4_mimic import run_agent4_mimic
from PipelineB.mimic_client import StubMimicClient

DISEASE1 = "Type 2 Diabetes Mellitus"
DISEASE2 = "Alzheimer's Disease"


def test_relative_risk_computed_correctly_from_stub_cohort_sets():
    # 100 total patients; 40 have T2DM; 20 have AD; 10 have both.
    # base AD rate = 20/100 = 0.2; AD rate within T2DM cohort = 10/40 = 0.25
    # relative risk = 0.25 / 0.2 = 1.25
    all_ids = {str(i) for i in range(100)}
    disease1_ids = {str(i) for i in range(40)}
    disease2_ids = {str(i) for i in range(30, 50)}  # overlaps disease1_ids in [30,40) -> 10 shared
    stub = StubMimicClient({"__all__": all_ids, DISEASE1: disease1_ids, DISEASE2: disease2_ids})

    result = run_agent4_mimic(DISEASE1, DISEASE2, mimic_client=stub)

    assert result.n_total_patients == 100
    assert result.n_disease1_patients == 40
    assert result.n_disease2_patients == 20
    assert result.n_both_patients == 10
    assert result.disease2_rate_in_disease1_cohort == pytest.approx(0.25)
    assert result.disease2_base_rate == pytest.approx(0.2)
    assert result.relative_risk == pytest.approx(1.25)


def test_zero_disease2_base_rate_returns_none_relative_risk_not_a_crash():
    all_ids = {"1", "2", "3"}
    stub = StubMimicClient({"__all__": all_ids, DISEASE1: {"1"}, DISEASE2: set()})

    result = run_agent4_mimic(DISEASE1, DISEASE2, mimic_client=stub)

    assert result.disease2_base_rate == 0.0
    assert result.relative_risk is None
    assert result.disease2_rate_in_disease1_cohort == 0.0


def test_missing_icd_mapping_raises_value_error():
    stub = StubMimicClient({"__all__": set()})
    with pytest.raises(ValueError, match="No ICD-10 prefix mapping"):
        run_agent4_mimic("Some Unmapped Disease", DISEASE2, mimic_client=stub)


def test_to_dict_is_json_serializable():
    import json

    stub = StubMimicClient({"__all__": {"1"}, DISEASE1: {"1"}, DISEASE2: {"1"}})
    result = run_agent4_mimic(DISEASE1, DISEASE2, mimic_client=stub)
    json.dumps(result.to_dict())
