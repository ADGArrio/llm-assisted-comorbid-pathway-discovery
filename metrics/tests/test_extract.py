from metrics.extract import extract_gene_protein_mentions


def test_extracts_plausible_gene_symbols():
    text = "TCF7L2 activates APP, which inhibits BACE1 via amyloid processing."
    symbols = extract_gene_protein_mentions(text)
    assert "TCF7L2" in symbols
    assert "APP" in symbols
    assert "BACE1" in symbols


def test_dedupes_preserving_first_occurrence_order():
    text = "APOE affects lipid transport. Later, APOE is mentioned again, then LDLR."
    symbols = extract_gene_protein_mentions(text)
    assert symbols == ["APOE", "LDLR"]


def test_filters_common_false_positives_by_default():
    text = "This mechanism was studied using GWAS and EHR data from a CAD cohort with AD."
    symbols = extract_gene_protein_mentions(text)
    assert symbols == []


def test_known_symbols_filter_narrows_results():
    text = "TCF7L2 and FAKEGENE both appear relevant, per this GWAS study."
    symbols = extract_gene_protein_mentions(text, known_symbols=["TCF7L2", "APOE"])
    assert symbols == ["TCF7L2"]


def test_empty_text_returns_empty_list():
    assert extract_gene_protein_mentions("") == []
    assert extract_gene_protein_mentions(None) == []


def test_extracts_hyphenated_subunit_symbols_whole():
    text = "IRS-1 mediates downstream signaling."
    symbols = extract_gene_protein_mentions(text)
    assert "IRS-1" in symbols
    assert "IRS" not in symbols


def test_filters_t2dm_disease_abbreviation_by_default():
    text = "T2DM leads to insulin resistance via INSR."
    symbols = extract_gene_protein_mentions(text)
    assert "T2DM" not in symbols
    assert "INSR" in symbols
