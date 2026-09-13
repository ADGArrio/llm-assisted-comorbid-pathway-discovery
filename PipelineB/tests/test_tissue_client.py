from PipelineB.tissue_client import (
    EXPRESSION_THRESHOLD,
    StubGTExClient,
    StubHPAClient,
    check_tissue_expression,
)


def test_gtex_expression_above_threshold_marks_expressed():
    gtex = StubGTExClient({"INSR|Pancreas": EXPRESSION_THRESHOLD + 5.0})
    hpa = StubHPAClient({})

    result = check_tissue_expression("INSR", "Pancreas", gtex, hpa)

    assert result.source == "GTEx"
    assert result.expressed is True
    assert result.value == EXPRESSION_THRESHOLD + 5.0
    assert gtex.calls == [("INSR", "Pancreas")]
    assert hpa.calls == []  # GTEx answered; HPA should never be consulted


def test_gtex_expression_below_threshold_marks_not_expressed():
    gtex = StubGTExClient({"APOE|Brain_Cortex": 0.05})
    hpa = StubHPAClient({})

    result = check_tissue_expression("APOE", "Brain_Cortex", gtex, hpa)

    assert result.source == "GTEx"
    assert result.expressed is False
    assert result.value == 0.05


def test_gtex_none_falls_back_to_hpa_when_ensembl_id_known():
    gtex = StubGTExClient({})  # no record for this gene/tissue
    hpa = StubHPAClient({"ENSG00000141510|Liver": 3.2})

    result = check_tissue_expression("TP53", "Liver", gtex, hpa, ensembl_id="ENSG00000141510")

    assert result.source == "HPA"
    assert result.expressed is True
    assert result.value == 3.2
    assert hpa.calls == [("ENSG00000141510", "Liver")]


def test_no_gtex_no_ensembl_id_skips_hpa_and_reports_no_data():
    gtex = StubGTExClient({})
    hpa = StubHPAClient({"ENSG_SHOULD_NOT_BE_CALLED|Liver": 3.2})

    result = check_tissue_expression("MID", "Liver", gtex, hpa, ensembl_id=None)

    assert result.source == "none"
    assert result.expressed is None
    assert result.value is None
    assert hpa.calls == []  # no ensembl id available, HPA can't be tried
    assert "no known Ensembl id" in result.note


def test_no_gtex_and_no_hpa_data_reports_no_data():
    gtex = StubGTExClient({})
    hpa = StubHPAClient({})

    result = check_tissue_expression("MID", "Liver", gtex, hpa, ensembl_id="ENSG00000000000")

    assert result.source == "none"
    assert result.expressed is None
    assert result.value is None


def test_to_dict_is_json_serializable():
    import json

    gtex = StubGTExClient({"INSR|Pancreas": 5.0})
    hpa = StubHPAClient({})
    result = check_tissue_expression("INSR", "Pancreas", gtex, hpa)
    json.dumps(result.to_dict())
