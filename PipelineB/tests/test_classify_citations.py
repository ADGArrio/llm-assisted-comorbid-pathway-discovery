"""Tests for scripts/classify_citations.py. Stub LLM only: these check the plumbing (prompt, parsing,
quote-grounding, retries, bookkeeping), NOT whether a real model labels papers correctly -- that is
what the hand-labeled validation sample is for."""
import csv
import json

import pytest

from common.llm_client import StubLLMClient
from scripts.classify_citations import (
    ALL_LABELS,
    PROMPT_VERSION,
    EVIDENCE_TYPES,
    build_prompt,
    classify_paper,
    gene_in_text,
    is_gene_focused_mechanism,
    load_citation_records,
    parse_response,
    quote_is_grounded,
    regrade_saved,
    run,
    summarize,
    supports_pair,
    write_review_sample,
)

TITLE = "INSR knockdown impairs insulin signalling in neurons"
ABSTRACT = (
    "BACKGROUND: Insulin resistance is common in Alzheimer's disease. "
    "RESULTS: INSR knockdown in primary neurons reduced AKT phosphorylation by 60%. "
    "CONCLUSION: Neuronal INSR signalling is required for synaptic maintenance."
)
RECORD = {"gene_symbol": "INSR", "disease_name": "Alzheimer's Disease", "pmid": "111", "title": TITLE}


def _reply(label="mechanism", quote="INSR knockdown in primary neurons reduced AKT phosphorylation by 60%.",
           focus=True, disease_focus=True):
    return json.dumps({"evidence_type": label, "quote": quote, "gene_is_focus": focus,
                       "disease_is_focus": disease_focus})


# --- prompt ---

def test_prompt_contains_gene_disease_text_and_every_label():
    p = build_prompt("INSR", "Alzheimer's Disease", TITLE, ABSTRACT)
    assert "Gene: INSR" in p and "Disease: Alzheimer's Disease" in p
    assert TITLE in p and "AKT phosphorylation" in p
    for label in EVIDENCE_TYPES:
        assert f'"{label}"' in p


# --- parse_response ---

def test_parse_valid_json():
    assert parse_response(_reply()) == {
        "evidence_type": "mechanism",
        "quote": "INSR knockdown in primary neurons reduced AKT phosphorylation by 60%.",
        "gene_is_focus": True,
        "disease_is_focus": True,
    }


def test_parse_tolerates_code_fence_and_preamble():
    assert parse_response("Sure!\n```json\n" + _reply("review", "x", False) + "\n```")["evidence_type"] == "review"


@pytest.mark.parametrize("bad", [
    "", "no json here", "{not json}", "[1, 2]",
    json.dumps({"evidence_type": "causal", "quote": "q", "gene_is_focus": True}),   # label not in the set
    json.dumps({"evidence_type": "mechanism", "quote": 5, "gene_is_focus": True}),  # quote not a string
    json.dumps({"evidence_type": "mechanism", "quote": "q", "gene_is_focus": "yes"}),  # focus not a bool
    json.dumps({"evidence_type": "mechanism", "quote": "q", "gene_is_focus": True}),  # disease_is_focus missing
])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_response(bad)


# --- quote grounding / gene presence ---

def test_quote_grounded_ignores_case_whitespace_and_trailing_period():
    assert quote_is_grounded("insr knockdown  in primary neurons\nreduced AKT phosphorylation by 60%", TITLE, ABSTRACT)


def test_quote_grounded_handles_curly_quotes_and_dashes():
    assert quote_is_grounded("type-2 \"risk\"", "", "A type–2 “risk” factor")


def test_quote_not_grounded_when_invented_or_empty():
    assert not quote_is_grounded("INSR activates mTOR in microglia", TITLE, ABSTRACT)
    assert not quote_is_grounded("", TITLE, ABSTRACT)


def test_quote_can_come_from_title():
    assert quote_is_grounded("INSR knockdown impairs insulin signalling", TITLE, ABSTRACT)


def test_gene_in_text_is_whole_token():
    assert gene_in_text("INSR", TITLE, ABSTRACT)
    assert not gene_in_text("INS", TITLE, ABSTRACT)  # INS must not match inside INSR
    assert not gene_in_text("APP", TITLE, ABSTRACT)


# --- classify_paper ---

def test_classify_good_response_is_verified_and_not_flagged():
    out = classify_paper(StubLLMClient(fixed_response=_reply()), RECORD, ABSTRACT)
    assert out["evidence_type"] == "mechanism" and out["quote_verified"] is True
    assert out["disease_is_focus"] is True and out["needs_review"] is False and out["n_attempts"] == 1 and out["gene_in_text"] is True
    assert out["pmid"] == "111" and out["disease_name"] == "Alzheimer's Disease"  # record fields kept


def test_classify_flags_hallucinated_quote_but_keeps_label():
    out = classify_paper(StubLLMClient(fixed_response=_reply(quote="INSR was knocked out in mice")), RECORD, ABSTRACT)
    assert out["evidence_type"] == "mechanism"
    assert out["quote_verified"] is False and out["needs_review"] is True


def test_classify_retries_once_after_unparseable_reply():
    stub = StubLLMClient(responses=["I think it is a mechanism paper.", _reply()])
    out = classify_paper(stub, RECORD, ABSTRACT)
    assert out["evidence_type"] == "mechanism" and out["n_attempts"] == 2 and len(stub.prompts) == 2


def test_classify_gives_parse_failed_after_all_attempts():
    stub = StubLLMClient(fixed_response="nope")
    out = classify_paper(stub, RECORD, ABSTRACT)
    assert out["evidence_type"] == "parse_failed" and out["needs_review"] is True
    assert out["n_attempts"] == 2 and out["error"]


def test_classify_without_abstract_skips_the_llm():
    stub = StubLLMClient(fixed_response=_reply())
    out = classify_paper(stub, RECORD, None)
    assert out["evidence_type"] == "no_abstract" and out["needs_review"] is True
    assert stub.prompts == []


def test_classify_uses_temperature_zero():
    seen = []

    class Spy:
        def generate(self, prompt, temperature=0.3):
            seen.append(temperature)
            return _reply()

    classify_paper(Spy(), RECORD, ABSTRACT)
    assert seen == [0.0]


# --- load_citation_records ---

def _saved(citations, d1_genes=("INSR", "TCF7L2"), d2_genes=("APP", "APOE")):
    def targets(genes):
        return [{"gene_symbol": g, "disease_assoc_score": 0.9 - i * 0.1} for i, g in enumerate(genes)]
    return {"agent1": {"disease1": "Type 2 Diabetes Mellitus", "disease2": "Alzheimer's Disease",
                       "disease1_targets": targets(d1_genes), "disease2_targets": targets(d2_genes),
                       "citations": citations}}


def test_old_format_citations_get_disease_from_top_genes():
    recs = load_citation_records(_saved({"INSR": [{"pmid": "1", "title": "t"}], "APP": [{"pmid": "2", "title": "u"}]}))
    assert {(r["gene_symbol"], r["disease_name"]) for r in recs} == {
        ("INSR", "Type 2 Diabetes Mellitus"), ("APP", "Alzheimer's Disease")}


def test_old_format_shared_gene_is_marked_ambiguous_not_guessed():
    recs = load_citation_records(_saved({"INSR": [{"pmid": "1", "title": "t"}]}, d2_genes=("INSR", "APP")))
    assert recs[0]["disease_name"].startswith("ambiguous")


def test_new_format_citation_keeps_its_own_disease():
    recs = load_citation_records(_saved({"INSR": [
        {"pmid": "1", "title": "t", "disease_name": "Alzheimer's Disease", "search_rank": 1, "pubtype": ["Review"]}]}))
    assert recs[0]["disease_name"] == "Alzheimer's Disease" and recs[0]["search_rank"] == 1


# --- summarize / review sample / run ---

def _classified():
    mech = classify_paper(StubLLMClient(fixed_response=_reply()), RECORD, ABSTRACT)
    inc = classify_paper(StubLLMClient(fixed_response=_reply("incidental_mention", "Insulin resistance is common in Alzheimer's disease.", False)),
                         {**RECORD, "pmid": "222", "disease_name": "Type 2 Diabetes Mellitus"}, ABSTRACT)
    none = classify_paper(StubLLMClient(fixed_response=_reply()), {**RECORD, "pmid": "333"}, None)
    return [mech, inc, none]


def test_summarize_counts_by_label_and_disease():
    s = summarize(_classified())
    assert s["n_papers"] == 3 and s["overall"]["mechanism"] == 1 and s["overall"]["incidental_mention"] == 1
    assert s["overall"]["no_abstract"] == 1 and set(s["overall"]) == set(ALL_LABELS)
    assert s["by_disease"]["Type 2 Diabetes Mellitus"]["incidental_mention"] == 1
    assert s["n_needs_review"] == 1 and s["fraction_mechanism"] == round(1 / 3, 3)


def test_summarize_empty_does_not_divide_by_zero():
    assert summarize([])["fraction_mechanism"] is None


def test_review_sample_csv_has_no_model_label_and_is_reproducible(tmp_path):
    papers = _classified()
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_review_sample(papers, str(a), n=2, seed=1)
    write_review_sample(papers, str(b), n=2, seed=1)
    assert a.read_text() == b.read_text()
    rows = list(csv.DictReader(open(a)))
    assert len(rows) == 2 and "evidence_type" not in rows[0] and rows[0]["your_label"] == ""


class _FakePubMed:
    def __init__(self, abstracts):
        self.abstracts, self.asked = abstracts, None

    def fetch_abstracts(self, pmids):
        self.asked = list(pmids)
        return self.abstracts


def test_run_end_to_end_writes_json(tmp_path):
    src = tmp_path / "pipeline_b_x.json"
    src.write_text(json.dumps(_saved({"INSR": [{"pmid": "111", "title": TITLE}, {"pmid": "999", "title": "no abs"}]})))
    out = tmp_path / "out" / "x.json"
    payload = run(str(src), str(out), StubLLMClient(fixed_response=_reply()), _FakePubMed({"111": ABSTRACT}))
    saved = json.loads(out.read_text())
    assert saved["summary"]["overall"]["mechanism"] == 1 and saved["summary"]["overall"]["no_abstract"] == 1
    assert payload["papers"][0]["abstract"] == ABSTRACT


def test_run_limit_caps_papers(tmp_path):
    src = tmp_path / "s.json"
    src.write_text(json.dumps(_saved({"INSR": [{"pmid": str(i), "title": "t"} for i in range(5)]})))
    pm = _FakePubMed({})
    payload = run(str(src), str(tmp_path / "o.json"), StubLLMClient(fixed_response=_reply()), pm, limit=2)
    assert len(payload["papers"]) == 2 and pm.asked == ["0", "1"]


def test_summarize_counts_papers_where_the_disease_is_not_the_focus():
    off = classify_paper(StubLLMClient(fixed_response=_reply("genetic_association", disease_focus=False)), RECORD, ABSTRACT)
    on = classify_paper(StubLLMClient(fixed_response=_reply()), RECORD, ABSTRACT)
    skipped = classify_paper(StubLLMClient(fixed_response=_reply()), RECORD, None)  # None, not False: never read
    assert summarize([off, on, skipped])["n_disease_not_the_focus"] == 1


def test_clinical_study_is_an_accepted_label():
    assert parse_response(_reply("clinical_study"))["evidence_type"] == "clinical_study"


# --- progress / checkpointing / resume (added 2026-10-15 after a slow live run) ---

def _two_paper_source(tmp_path):
    src = tmp_path / "s.json"
    src.write_text(json.dumps(_saved({"INSR": [{"pmid": "1", "title": "t"}, {"pmid": "2", "title": "u"}]})))
    return src


def test_run_calls_progress_once_per_paper(tmp_path):
    seen = []
    run(str(_two_paper_source(tmp_path)), str(tmp_path / "o.json"), StubLLMClient(fixed_response=_reply()),
        _FakePubMed({"1": ABSTRACT, "2": ABSTRACT}), progress=lambda i, n, rec, secs, skipped: seen.append((i, n, skipped)))
    assert seen == [(1, 2, False), (2, 2, False)]


def test_run_saves_after_every_paper_so_an_interrupted_run_keeps_its_work(tmp_path):
    out = tmp_path / "o.json"
    stub = StubLLMClient(responses=[_reply()])  # 2nd generate() call raises (queue exhausted), like a crash/Ctrl-C
    with pytest.raises(Exception):
        run(str(_two_paper_source(tmp_path)), str(out), stub, _FakePubMed({"1": ABSTRACT, "2": ABSTRACT}))
    saved = json.loads(out.read_text())
    assert [p["pmid"] for p in saved["papers"]] == ["1"] and saved["complete"] is False


def test_run_marks_complete_when_finished(tmp_path):
    out = tmp_path / "o.json"
    run(str(_two_paper_source(tmp_path)), str(out), StubLLMClient(fixed_response=_reply()),
        _FakePubMed({"1": ABSTRACT, "2": ABSTRACT}))
    assert json.loads(out.read_text())["complete"] is True


def test_resume_skips_finished_papers_and_does_not_call_the_llm_for_them(tmp_path):
    src, out = _two_paper_source(tmp_path), tmp_path / "o.json"
    with pytest.raises(Exception):  # first run dies after paper 1
        run(str(src), str(out), StubLLMClient(responses=[_reply()]), _FakePubMed({"1": ABSTRACT, "2": ABSTRACT}))
    stub, pm = StubLLMClient(fixed_response=_reply("review", "Insulin resistance is common in Alzheimer's disease.")), _FakePubMed({"2": ABSTRACT})
    payload = run(str(src), str(out), stub, pm, resume=True)
    assert len(stub.prompts) == 1 and pm.asked == ["2"]  # only the missing paper was fetched and classified
    assert [p["evidence_type"] for p in payload["papers"]] == ["mechanism", "review"]
    assert payload["complete"] is True


def test_resume_retries_a_parse_failed_paper(tmp_path):
    src, out = _two_paper_source(tmp_path), tmp_path / "o.json"
    run(str(src), str(out), StubLLMClient(fixed_response="garbage"), _FakePubMed({"1": ABSTRACT, "2": ABSTRACT}))
    assert json.loads(out.read_text())["summary"]["overall"]["parse_failed"] == 2
    payload = run(str(src), str(out), StubLLMClient(fixed_response=_reply()), _FakePubMed({"1": ABSTRACT, "2": ABSTRACT}), resume=True)
    assert payload["summary"]["overall"]["mechanism"] == 2


def test_without_resume_an_existing_file_is_overwritten(tmp_path):
    src, out = _two_paper_source(tmp_path), tmp_path / "o.json"
    run(str(src), str(out), StubLLMClient(fixed_response=_reply()), _FakePubMed({"1": ABSTRACT, "2": ABSTRACT}))
    stub = StubLLMClient(fixed_response=_reply("review", "Insulin resistance is common in Alzheimer's disease."))
    payload = run(str(src), str(out), stub, _FakePubMed({"1": ABSTRACT, "2": ABSTRACT}))
    assert len(stub.prompts) == 2 and payload["summary"]["overall"]["review"] == 2


# --- ellipsis-tolerant quote check, pair-support rule, regrade (added 2026-10-15 after the first full run) ---

def test_quote_with_ellipsis_is_grounded_when_every_fragment_is_found_in_order():
    q = "BACKGROUND: Insulin resistance is common ... INSR knockdown in primary neurons reduced AKT phosphorylation"
    assert quote_is_grounded(q, TITLE, ABSTRACT)
    assert quote_is_grounded(q.replace("...", "\u2026"), TITLE, ABSTRACT)  # the single-character ellipsis too


def test_quote_with_ellipsis_fails_when_a_fragment_is_invented_or_out_of_order():
    assert not quote_is_grounded("Insulin resistance is common ... INSR activates mTOR in microglia", TITLE, ABSTRACT)
    assert not quote_is_grounded("INSR knockdown in primary neurons ... Insulin resistance is common", TITLE, ABSTRACT)


def test_quote_with_ellipsis_rejects_tiny_fragments_that_match_anything():
    assert not quote_is_grounded("INSR knockdown in primary neurons ... a ... phosphorylation by 60%", TITLE, ABSTRACT)


def _paper(label, gene_focus=True, disease_focus=True, gene="INSR", disease="Alzheimer's Disease"):
    return {"gene_symbol": gene, "disease_name": disease, "evidence_type": label, "gene_is_focus": gene_focus,
            "disease_is_focus": disease_focus, "needs_review": False, "gene_in_text": True}


def test_supports_pair_requires_disease_focus_and_a_non_incidental_label():
    assert supports_pair(_paper("genetic_association", gene_focus=False))
    assert not supports_pair(_paper("genetic_association", disease_focus=False))
    assert not supports_pair(_paper("incidental_mention"))
    assert not supports_pair(_paper("no_abstract", gene_focus=None, disease_focus=None))
    assert not supports_pair(_paper("parse_failed", gene_focus=None, disease_focus=None))


def test_supports_pair_strict_version_also_needs_the_gene_to_be_the_focus():
    assert supports_pair(_paper("mechanism"), require_gene_focus=True)
    assert not supports_pair(_paper("mechanism", gene_focus=False), require_gene_focus=True)


def test_gene_focused_mechanism_needs_all_three_conditions():
    assert is_gene_focused_mechanism(_paper("mechanism"))
    assert not is_gene_focused_mechanism(_paper("mechanism", gene_focus=False))
    assert not is_gene_focused_mechanism(_paper("mechanism", disease_focus=False))
    assert not is_gene_focused_mechanism(_paper("genetic_association"))


def test_summarize_pair_level_and_gene_level_counts():
    papers = [
        _paper("mechanism", gene="APP"),                          # APP: gene-focused mechanism
        _paper("mechanism", gene_focus=False, gene="APP"),        # network-pharmacology style
        _paper("mechanism", gene_focus=False, gene="CDK5"),       # CDK5: only a list-style mechanism paper
        _paper("incidental_mention", gene_focus=False, gene="APOE"),
        _paper("genetic_association", gene_focus=False, gene="GRIN3B"),
    ]
    s = summarize(papers)
    p = s["pair_level"]["overall"]
    assert p["n_papers"] == 5 and p["n_gene_is_focus"] == 1 and p["n_supports_pair"] == 4
    assert p["n_supports_pair_gene_focus"] == 1 and p["n_mechanism_gene_focus"] == 1
    g = s["gene_level"]["Alzheimer's Disease"]
    assert g["n_genes"] == 4 and g["n_genes_with_supporting_paper"] == 3          # APP, CDK5, GRIN3B (APOE is incidental)
    assert g["n_genes_with_gene_focused_supporting_paper"] == 1 and g["n_genes_with_gene_focused_mechanism"] == 1
    assert s["fraction_mechanism"] == 0.6 and s["fraction_mechanism_gene_focus"] == 0.2


def test_regrade_saved_rechecks_quotes_without_calling_the_llm(tmp_path):
    ellipsis_quote = "BACKGROUND: Insulin resistance is common ... INSR knockdown in primary neurons"
    old = classify_paper(StubLLMClient(fixed_response=_reply(quote=ellipsis_quote)), RECORD,
                         ABSTRACT)
    # Simulate a file written by the older, stricter checker: same model output, graded as unverified.
    old.update(quote_verified=False, needs_review=True)
    path = tmp_path / "saved.json"
    path.write_text(json.dumps({"complete": True, "summary": summarize([old]), "papers": [old]}))
    assert json.loads(path.read_text())["summary"]["n_needs_review"] == 1
    out = regrade_saved(str(path))
    assert out["papers"][0]["quote_verified"] is True and out["summary"]["n_needs_review"] == 0
    assert out["papers"][0]["evidence_type"] == "mechanism"        # the model's label is untouched
    assert json.loads(path.read_text())["regraded"] is True


# --- compound_study label + prompt versioning (added 2026-10-15) ---

def test_compound_study_is_a_label_and_the_prompt_explains_it():
    assert parse_response(_reply("compound_study"))["evidence_type"] == "compound_study"
    p = build_prompt("ACHE", "Alzheimer's Disease", "t", "a")
    assert '"compound_study"' in p and "COMPOUND does, not what the gene does" in p
    assert "choose compound_study" in p
    assert "compound_study" in ALL_LABELS


def test_mechanism_definition_no_longer_includes_drugs():
    assert "drug acting through the gene" not in EVIDENCE_TYPES["mechanism"]


def test_compound_study_counts_as_supporting_the_pair_only_when_the_disease_is_the_focus():
    assert supports_pair(_paper("compound_study"))
    assert not supports_pair(_paper("compound_study", disease_focus=False))
    assert not is_gene_focused_mechanism(_paper("compound_study"))


def test_output_records_the_prompt_version(tmp_path):
    out = tmp_path / "o.json"
    run(str(_two_paper_source(tmp_path)), str(out), StubLLMClient(fixed_response=_reply()),
        _FakePubMed({"1": ABSTRACT, "2": ABSTRACT}))
    assert json.loads(out.read_text())["prompt_version"] == PROMPT_VERSION


def test_resume_refuses_to_mix_labels_from_a_different_prompt_version(tmp_path):
    src, out = _two_paper_source(tmp_path), tmp_path / "o.json"
    run(str(src), str(out), StubLLMClient(fixed_response=_reply()), _FakePubMed({"1": ABSTRACT, "2": ABSTRACT}))
    saved = json.loads(out.read_text())
    saved["prompt_version"] = "v1"
    out.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="prompt"):
        run(str(src), str(out), StubLLMClient(fixed_response=_reply()), _FakePubMed({}), resume=True)


def test_resume_treats_a_file_without_a_version_as_v1(tmp_path):
    src, out = _two_paper_source(tmp_path), tmp_path / "o.json"
    run(str(src), str(out), StubLLMClient(fixed_response=_reply()), _FakePubMed({"1": ABSTRACT, "2": ABSTRACT}))
    saved = json.loads(out.read_text())
    del saved["prompt_version"]
    out.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="v1"):
        run(str(src), str(out), StubLLMClient(fixed_response=_reply()), _FakePubMed({}), resume=True)


# --- v3: review beats compound_study (added 2026-10-18) ---

def test_prompt_v3_tells_the_model_a_drug_review_is_a_review():
    p = build_prompt("ADAM10", "Alzheimer's Disease", "t", "a")
    assert PROMPT_VERSION == "v3-review-beats-compound"
    assert "review is a review even when a drug or compound is its subject" in p
    # the review check must come before the compound_study rule in the Rules list
    assert p.index("choose review, even when a drug") < p.index("choose compound_study, even when it also reports")
