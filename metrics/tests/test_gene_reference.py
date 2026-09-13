import os
import tempfile

import pytest

from metrics.gene_reference import load_known_gene_symbols


def _write_tsv(rows):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, newline="")
    f.write("hgnc_id\tsymbol\tname\n")
    for row in rows:
        f.write(f"{row['hgnc_id']}\t{row['symbol']}\t{row['name']}\n")
    f.close()
    return f.name


def test_loads_symbols_uppercased():
    path = _write_tsv(
        [
            {"hgnc_id": "HGNC:1", "symbol": "tcf7l2", "name": "transcription factor 7 like 2"},
            {"hgnc_id": "HGNC:2", "symbol": "APOE", "name": "apolipoprotein E"},
        ]
    )
    try:
        symbols = load_known_gene_symbols(path)
        assert symbols == {"TCF7L2", "APOE"}
    finally:
        os.unlink(path)


def test_missing_path_raises_with_download_hint(monkeypatch):
    monkeypatch.delenv("HGNC_GENE_LIST_PATH", raising=False)
    with pytest.raises(FileNotFoundError, match="genenames.org"):
        load_known_gene_symbols(None)


def test_env_var_used_when_path_not_passed(monkeypatch):
    path = _write_tsv([{"hgnc_id": "HGNC:1", "symbol": "APP", "name": "amyloid beta precursor protein"}])
    try:
        monkeypatch.setenv("HGNC_GENE_LIST_PATH", path)
        symbols = load_known_gene_symbols()
        assert symbols == {"APP"}
    finally:
        os.unlink(path)
