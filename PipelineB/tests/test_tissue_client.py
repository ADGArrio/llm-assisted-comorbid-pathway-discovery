import json

from PipelineB.tissue_client import EXPRESSION_THRESHOLD, StubGTExClient, check_tissue_expression


def test_gtex_expression_above_threshold_marks_expressed():
    gtex = StubGTExClient({"INSR|Pancreas": EXPRESSION_THRESHOLD + 5.0})

    result = check_tissue_expression("INSR", "Pancreas", gtex)

    assert result.source == "GTEx"
    assert result.expressed is True
    assert result.value == EXPRESSION_THRESHOLD + 5.0
    assert gtex.calls == [("INSR", "Pancreas")]


def test_gtex_expression_below_threshold_marks_not_expressed():
    gtex = StubGTExClient({"APOE|Brain_Cortex": 0.05})

    result = check_tissue_expression("APOE", "Brain_Cortex", gtex)

    assert result.source == "GTEx"
    assert result.expressed is False
    assert result.value == 0.05


def test_no_gtex_data_reports_no_data():
    gtex = StubGTExClient({})

    result = check_tissue_expression("MID", "Liver", gtex)

    assert result.source == "none"
    assert result.expressed is None
    assert result.value is None
    assert "No GTEx record" in result.note


def test_to_dict_is_json_serializable():
    gtex = StubGTExClient({"INSR|Pancreas": 5.0})
    result = check_tissue_expression("INSR", "Pancreas", gtex)
    json.dumps(result.to_dict())
