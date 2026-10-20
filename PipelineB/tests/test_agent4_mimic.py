import pytest

from PipelineB.agent4_mimic import run_agent4_mimic
from PipelineB.mimic_client import StubMimicClient

DISEASE1 = "Type 2 Diabetes Mellitus"
DISEASE2 = "Alzheimer's Disease"


def test_relative_risk_computed_correctly_from_stub_cohort_sets():
    # 100 total patients; 40 have T2DM; 20 have AD; 10 have both.
    # base AD rate = 20/100 = 0.2; AD rate within T2DM cohort = 10/40 = 0.25
    # observed/expected (the old "relative risk", a lift) = 0.25 / 0.2 = 1.25
    # TRUE relative risk: P(AD | T2DM) / P(AD | no T2DM) = 0.25 / (10 / 60) = 1.5
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
    assert result.observed_over_expected == pytest.approx(1.25)
    assert result.relative_risk == pytest.approx(1.5)


def test_zero_disease2_base_rate_returns_none_relative_risk_not_a_crash():
    all_ids = {"1", "2", "3"}
    stub = StubMimicClient({"__all__": all_ids, DISEASE1: {"1"}, DISEASE2: set()})

    result = run_agent4_mimic(DISEASE1, DISEASE2, mimic_client=stub)

    assert result.disease2_base_rate == 0.0
    assert result.relative_risk is None
    assert result.observed_over_expected is None
    assert result.odds_ratio is None and result.p_value is None
    assert result.disease2_rate_in_disease1_cohort == 0.0


def test_missing_icd_mapping_raises_value_error():
    stub = StubMimicClient({"__all__": set()})
    with pytest.raises(ValueError, match="No ICD prefix mapping"):
        run_agent4_mimic("Some Unmapped Disease", DISEASE2, mimic_client=stub)


def test_to_dict_is_json_serializable():
    import json

    stub = StubMimicClient({"__all__": {"1"}, DISEASE1: {"1"}, DISEASE2: {"1"}})
    result = run_agent4_mimic(DISEASE1, DISEASE2, mimic_client=stub)
    json.dumps(result.to_dict())


def test_real_mimic_counts_reproduce_the_hand_computed_statistics():
    # Counts from the 2026-10-18 real run (T2DM -> AD); values cross-checked against scipy's chi2_contingency.
    from PipelineB.agent4_mimic import _two_by_two_stats

    s = _two_by_two_stats(n_total=223291, n1=44005, n2=3234, n_both=841)
    assert s["relative_risk"] == pytest.approx(1.4317, abs=1e-3)
    lo, hi = s["relative_risk_ci95"]
    assert lo == pytest.approx(1.325, abs=2e-3) and hi == pytest.approx(1.548, abs=2e-3)
    assert s["odds_ratio"] == pytest.approx(1.440, abs=1e-3)
    assert s["p_value"] < 1e-15

    c = _two_by_two_stats(n_total=223291, n1=44005, n2=46062, n_both=18647)
    assert c["relative_risk"] == pytest.approx(2.771, abs=1e-3)
    assert c["odds_ratio"] == pytest.approx(4.074, abs=1e-3)


def test_null_association_gives_rr_one_and_p_one():
    from PipelineB.agent4_mimic import _two_by_two_stats

    s = _two_by_two_stats(n_total=100, n1=50, n2=20, n_both=10)  # 20% in both groups
    assert s["relative_risk"] == pytest.approx(1.0)
    assert s["odds_ratio"] == pytest.approx(1.0)
    assert s["p_value"] == pytest.approx(1.0)


def test_inconsistent_counts_yield_no_statistics():
    from PipelineB.agent4_mimic import _two_by_two_stats

    s = _two_by_two_stats(n_total=10, n1=8, n2=8, n_both=1)  # 'neither' cell would be negative
    assert all(v is None for v in s.values())
