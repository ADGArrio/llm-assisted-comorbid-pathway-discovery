"""Citation evidence classifier: for each paper Stage 1 attached to a top gene, the LLM reads the abstract and decides whether it shows a
mechanism or only mentions the gene and the disease. Standalone; nothing in the pipeline imports it.

Each paper gets one label from a fixed set plus a verbatim quote from the abstract. A label whose quote cannot be found in the title or
abstract is flagged needs_review, and papers with no abstract are labelled "no_abstract" without calling the LLM. The labels are a
measurement to validate against hand labels (--sample-for-review writes a blank CSV for that), not ground truth.

Data: Stage 1 citations saved in outputs/pipeline_b_<pair>.json, abstracts from PubMed efetch, and the LLM from common/llm_client.py.

Run (needs Ollama up, same .env as the pipeline):
    python -m scripts.classify_citations --pair t2dm_ad
    python -m scripts.classify_citations --pair t2dm_cad --limit 5   # quick smoke test
    python -m scripts.classify_citations --pair t2dm_ad --resume     # continue an interrupted run
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

# label -> definition shown to the model, ordered strongest to weakest evidence of a mechanism; one dict so the prompt,
# the validator and the summary agree.
EVIDENCE_TYPES: Dict[str, str] = {
    "mechanism": (
        "The paper experimentally manipulates or directly measures what the gene DOES in the disease "
        "biology (knockout/knockdown/overexpression, cell or animal model, signalling or pathway "
        "shown, protein function). Testing a compound or extract is NOT mechanism, even if pathways "
        "change: that is compound_study."
    ),
    "genetic_association": (
        "The paper links variants or genetic risk of the gene to the disease (GWAS, candidate-gene "
        "or polymorphism study, Mendelian randomization, meta-analysis of such studies) without "
        "showing what the gene does."
    ),
    "expression_or_biomarker": (
        "The paper reports the gene's expression, protein level, methylation or serum level being "
        "different or correlated in patients/samples, with no manipulation and no genetic test."
    ),
    "clinical_study": (
        "A patient or trial study in which the gene is a genotype, a stratifier or a clinical drug "
        "target (e.g. genotype-guided therapy, carriers vs non-carriers, an approved drug tested in "
        "patients) but the gene's own function is not tested."
    ),
    "compound_study": (
        "The paper's main subject is a drug, natural product, herbal extract or other compound: "
        "designing, docking, network-pharmacology-predicting or testing it in cells/animals. The gene "
        "is a target or readout of the compound. It shows what the COMPOUND does, not what the gene does."
    ),
    "review": (
        "A narrative or systematic review, perspective, commentary, regulatory/framework analysis or "
        "overview that summarises or discusses OTHER work and reports no new experiment, cohort or "
        "analysis of its own. A review is a review even when a drug or compound is its subject."
    ),
    "incidental_mention": (
        "The gene and the disease both appear, but the gene is not what the paper is about "
        "(e.g. it is one gene in a long list, or part of a panel/background sentence)."
    ),
}
NO_ABSTRACT = "no_abstract"
PARSE_FAILED = "parse_failed"
ALL_LABELS = list(EVIDENCE_TYPES) + [NO_ABSTRACT, PARSE_FAILED]

# Saved in every output file. Bump it whenever the prompt or label set changes: --resume refuses to mix papers labelled
# under different prompts (files without the field are "v1").
# v1: first label set. v2: adds compound_study. v3: a review beats compound_study.
PROMPT_VERSION = "v3-review-beats-compound"

DEFAULT_PAPERS_PER_GENE_LIMIT = None  # classify everything in the saved file unless --limit is given
DEFAULT_OUT_DIR = os.path.join("outputs", "citation_evidence")


# Input: the gene and disease the search was run for, and the paper's title and abstract. Output: the prompt text; the
# label set comes from EVIDENCE_TYPES.
def build_prompt(gene: str, disease: str, title: str, abstract: str) -> str:
    definitions = "\n".join(f'- "{label}": {text}' for label, text in EVIDENCE_TYPES.items())
    return f"""You are classifying what kind of evidence a single PubMed paper provides about one gene in one disease.

Gene: {gene}
Disease: {disease}

Title: {title}
Abstract: {abstract}

Choose exactly ONE evidence_type for how this paper relates {gene} to {disease}:
{definitions}

Rules:
- Judge only from the title and abstract above. Do not use outside knowledge about the gene.
- If several apply, choose the strongest one the abstract actually supports (mechanism > genetic_association > expression_or_biomarker). Do not infer a mechanism that is not stated.
- First ask: does the paper report its OWN new experiment, cohort or analysis? If it instead summarises, discusses or proposes a framework from other work (the abstract says "this review", "we review", "perspective", "overview", "this article examines"), choose review, even when a drug, compound or drug class is its subject.
- Only if it does report its own work: if the main subject is a compound, drug candidate or extract (docking, network pharmacology, an extract tested in a model), choose compound_study, even when it also reports pathway or expression changes.
- "quote" must be copied EXACTLY, word for word, from the title or abstract (at most 40 words) and must be the sentence that best justifies your label. For "incidental_mention" it may be the sentence where the gene appears.
- "gene_is_focus" is true only if {gene} is one of the main subjects of the paper (not one gene in a long list).
- "disease_is_focus" is true only if the paper studies {disease} itself. It is false when the paper is really about a different outcome (for example a Mendelian-randomization study of this gene's drug target on pneumonia or heart failure).

Answer with ONLY a JSON object, no other text:
{{"evidence_type": "<one of: {', '.join(EVIDENCE_TYPES)}>", "quote": "<verbatim sentence>", "gene_is_focus": true or false, "disease_is_focus": true or false}}"""


# Input: the model's raw reply (```json fences and chatty preambles are tolerated). Output: the validated
# {"evidence_type", "quote", "gene_is_focus", "disease_is_focus"}; raises ValueError if there is no JSON object, the
# label is not in EVIDENCE_TYPES or fields are missing.
def parse_response(text: str) -> dict:
    if not text:
        raise ValueError("empty response")
    start, end = text.find("{"), text.rfind("}")  # tolerates ```json fences and chatty preambles
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:120]!r}")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    label = data.get("evidence_type")
    if label not in EVIDENCE_TYPES:
        raise ValueError(f"evidence_type {label!r} is not one of {list(EVIDENCE_TYPES)}")
    quote = data.get("quote", "")
    if not isinstance(quote, str):
        raise ValueError("quote is not a string")
    flags = {}
    for key in ("gene_is_focus", "disease_is_focus"):
        if not isinstance(data.get(key), bool):
            raise ValueError(f"{key} is not a boolean")
        flags[key] = data[key]
    return {"evidence_type": label, "quote": quote.strip(), **flags}


# Lower-case, unify quotes and dashes and collapse whitespace so harmless formatting differences do not fail the quote
# check. Input and output: text.
def _normalize(text: str) -> str:
    text = text.lower()
    for a, b in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-"), ("−", "-")):
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


_ELLIPSIS = re.compile(r"\.{3,}|\u2026")


# Input: the model's quote and the source title and abstract. Output: True if the quote appears verbatim (after
# normalisation) in the title or abstract; an empty quote is not grounded. A quote shortened with "..." is split at each
# ellipsis and counts only if every fragment is found, in order, in the same text, and each fragment is at least 3 words
# when there are several.
def quote_is_grounded(quote: str, title: str, abstract: str) -> bool:
    fragments = [f.strip(" .") for f in _ELLIPSIS.split(_normalize(quote or ""))]
    fragments = [f for f in fragments if f]
    if not fragments:
        return False
    if len(fragments) > 1 and any(len(f.split()) < 3 for f in fragments):
        return False
    for text in (_normalize(title or ""), _normalize(abstract or "")):
        position, found_all = 0, True
        for fragment in fragments:
            at = text.find(fragment, position)
            if at == -1:
                found_all = False
                break
            position = at + len(fragment)
        if found_all:
            return True
    return False


# Input: gene symbol, title and abstract. Output: True if the literal symbol appears as a whole token (PubMed's match
# can be satisfied by a synonym, so this records whether the symbol itself is there).
def gene_in_text(gene: str, title: str, abstract: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(gene)}(?![A-Za-z0-9])", f"{title} {abstract}", re.I) is not None


# Classify one paper. Input: an LLM client, a citation record (gene_symbol, disease_name, pmid, title, ...), its
# abstract (None if PubMed has none) and the attempt limit. Output: the record extended with evidence_type, quote,
# gene_is_focus, disease_is_focus, quote_verified, gene_in_text, needs_review and n_attempts; never raises on bad model
# output.
def classify_paper(client, record: dict, abstract: Optional[str], max_attempts: int = 2) -> dict:
    out = dict(record)
    out["abstract"] = abstract
    out["gene_in_text"] = gene_in_text(record["gene_symbol"], record.get("title", ""), abstract or "")
    if not abstract:
        out.update(evidence_type=NO_ABSTRACT, quote="", gene_is_focus=None, disease_is_focus=None,
                   quote_verified=False, needs_review=True, n_attempts=0, error="PubMed returned no abstract for this PMID")
        return out

    prompt = build_prompt(record["gene_symbol"], record["disease_name"], record.get("title", ""), abstract)
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            parsed = parse_response(client.generate(prompt, temperature=0.0))
        except ValueError as e:
            last_error = str(e)
            continue
        verified = quote_is_grounded(parsed["quote"], record.get("title", ""), abstract)
        out.update(parsed, quote_verified=verified, n_attempts=attempt, error=None,
                   needs_review=not verified)  # an unverifiable quote means the label rests on nothing
        return out
    out.update(evidence_type=PARSE_FAILED, quote="", gene_is_focus=None, disease_is_focus=None,
               quote_verified=False, needs_review=True, n_attempts=max_attempts, error=last_error)
    return out


# Input: a saved pipeline_b_<pair>.json dict and the top-N gene limit. Output: one record per (gene, paper) with the
# disease the search used. Old-format citations (saved before the search provenance fields existed) only have pmid and
# title, so the disease is inferred from which top-N list the gene is in, and a gene in both is marked ambiguous.
def load_citation_records(result: dict, top_n: int = 10) -> List[dict]:
    a1 = result["stage1"]
    top = {}
    for key, name_key in (("disease1_targets", "disease1"), ("disease2_targets", "disease2")):
        ranked = sorted(a1.get(key, []), key=lambda t: t["disease_assoc_score"], reverse=True)[:top_n]
        top[a1[name_key]] = {t["gene_symbol"] for t in ranked}

    records = []
    for gene, papers in a1["citations"].items():
        for paper in papers:
            disease = paper.get("disease_name")
            if not disease:
                owners = [d for d, genes in top.items() if gene in genes]
                disease = owners[0] if len(owners) == 1 else "ambiguous (shared gene, old format)"
            records.append({
                "gene_symbol": gene,
                "disease_name": disease,
                "pmid": paper["pmid"],
                "title": paper.get("title", ""),
                "pubtype": paper.get("pubtype"),
                "search_rank": paper.get("search_rank"),
            })
    return records


# Does this paper count as literature for the (gene, disease) pair that was searched? Yes if the paper is about the
# disease and the gene is more than a passing mention (label is not incidental_mention, no_abstract or parse_failed); a
# review counts. With require_gene_focus the gene must also be a main subject of the paper (the strict version, used for
# reporting). Input: a classified record and that flag. Output: bool.
def supports_pair(r: dict, require_gene_focus: bool = False) -> bool:
    if r.get("disease_is_focus") is not True:
        return False
    if r.get("evidence_type") not in EVIDENCE_TYPES or r["evidence_type"] == "incidental_mention":
        return False
    return not require_gene_focus or r.get("gene_is_focus") is True


# Input: a classified record. Output: True if the paper shows a mechanism and is about this gene and this disease (the
# strongest label).
def is_gene_focused_mechanism(r: dict) -> bool:
    return (r.get("evidence_type") == "mechanism" and r.get("gene_is_focus") is True
            and r.get("disease_is_focus") is True)


# Input: classified records. Output: label counts overall and per disease, how many need review, and the pair-level
# numbers: papers where the gene or disease is the focus, papers supporting the pair, and per disease how many searched
# genes have at least one supporting paper.
def summarize(classified: List[dict]) -> dict:
    overall = Counter(r["evidence_type"] for r in classified)
    by_disease: Dict[str, Counter] = {}
    genes_by_disease: Dict[str, Dict[str, List[dict]]] = {}
    for r in classified:
        by_disease.setdefault(r["disease_name"], Counter())[r["evidence_type"]] += 1
        genes_by_disease.setdefault(r["disease_name"], {}).setdefault(r["gene_symbol"], []).append(r)
    n = len(classified)

    # Input: classified records. Output: the pair-level numbers for those records.
    def pair_level(rows: List[dict]) -> dict:
        return {
            "n_papers": len(rows),
            "n_gene_is_focus": sum(1 for r in rows if r.get("gene_is_focus") is True),
            "n_supports_pair": sum(1 for r in rows if supports_pair(r)),
            "n_supports_pair_gene_focus": sum(1 for r in rows if supports_pair(r, require_gene_focus=True)),
            "n_mechanism_gene_focus": sum(1 for r in rows if is_gene_focused_mechanism(r)),
        }

    gene_level = {}
    for disease, genes in genes_by_disease.items():
        gene_level[disease] = {
            "n_genes": len(genes),
            "n_genes_with_supporting_paper": sum(1 for rows in genes.values() if any(supports_pair(r) for r in rows)),
            "n_genes_with_gene_focused_supporting_paper": sum(
                1 for rows in genes.values() if any(supports_pair(r, require_gene_focus=True) for r in rows)),
            "n_genes_with_gene_focused_mechanism": sum(
                1 for rows in genes.values() if any(is_gene_focused_mechanism(r) for r in rows)),
        }
    return {
        "n_papers": n,
        "overall": {label: overall.get(label, 0) for label in ALL_LABELS},
        "by_disease": {d: {label: c.get(label, 0) for label in ALL_LABELS} for d, c in by_disease.items()},
        "pair_level": {"overall": pair_level(classified),
                       "by_disease": {d: pair_level([r for r in classified if r["disease_name"] == d]) for d in by_disease}},
        "gene_level": gene_level,
        "n_needs_review": sum(1 for r in classified if r["needs_review"]),
        "n_gene_not_literally_in_text": sum(1 for r in classified if not r["gene_in_text"]),
        "n_disease_not_the_focus": sum(1 for r in classified if r.get("disease_is_focus") is False),
        "fraction_mechanism": round(overall.get("mechanism", 0) / n, 3) if n else None,
        "fraction_mechanism_gene_focus": round(sum(1 for r in classified if is_gene_focused_mechanism(r)) / n, 3) if n else None,
    }


# Recompute the code-side fields (quote_verified, needs_review, gene_in_text, summary) of an already-saved output
# without calling the LLM; the model's labels, quotes and flags are untouched. Input: the output file path. Output: the
# regraded payload.
def regrade_saved(out_path: str) -> dict:
    with open(out_path) as f:
        payload = json.load(f)
    for r in payload["papers"]:
        if r["evidence_type"] in (NO_ABSTRACT, PARSE_FAILED):
            continue
        r["quote_verified"] = quote_is_grounded(r.get("quote", ""), r.get("title", ""), r.get("abstract") or "")
        r["needs_review"] = not r["quote_verified"]
        r["gene_in_text"] = gene_in_text(r["gene_symbol"], r.get("title", ""), r.get("abstract") or "")
    payload["summary"] = summarize(payload["papers"])
    payload["regraded"] = True
    _write_payload(out_path, payload)
    return payload


# Write a CSV of up to n randomly chosen papers, without the model's label, for hand labelling; seeded so it is
# reproducible. Input: classified records, path, n and seed. Output: none.
def write_review_sample(classified: List[dict], path: str, n: int = 50, seed: int = 0):
    rng = random.Random(seed)
    rows = rng.sample(classified, min(n, len(classified)))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pmid", "gene_symbol", "disease_name", "title", "abstract", "your_label", "notes"])
        for r in rows:
            w.writerow([r["pmid"], r["gene_symbol"], r["disease_name"], r["title"], r["abstract"] or "", "", ""])


# Input: a record. Output: its (gene, disease, paper) key; the same paper retrieved for two genes is two records because
# the prompt and answer depend on the gene.
def _record_key(r: dict) -> tuple:
    return (r["gene_symbol"], r["disease_name"], str(r["pmid"]))


# Write via a temp file and rename so an interrupt cannot leave a half-written JSON file. Input: path and payload dict.
# Output: none.
def _write_payload(out_path: str, payload: dict):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)


# Classify every paper in a saved pipeline file. Input: saved pipeline JSON path, output path, an LLM client, a PubMed
# client (anything with fetch_abstracts), an optional paper limit, and resume and progress options. Output: the written
# dict {"summary", "papers"}. The output file is rewritten after every paper ("complete": false until the last) so an
# interrupted run keeps what it finished; with resume=True, papers already classified are skipped (a parse_failed one is
# retried).
def run(result_path: str, out_path: str, client, pubmed_client, limit: Optional[int] = None,
        top_n: int = 10, resume: bool = False, progress=None) -> dict:
    with open(result_path) as f:
        records = load_citation_records(json.load(f), top_n=top_n)
    if limit:
        records = records[:limit]

    done: Dict[tuple, dict] = {}
    if resume and os.path.exists(out_path):
        with open(out_path) as f:
            saved = json.load(f)
        saved_version = saved.get("prompt_version", "v1")
        if saved_version != PROMPT_VERSION:
            raise ValueError(
                f"{out_path} was labelled with prompt {saved_version!r}, current is {PROMPT_VERSION!r}. "
                "Resuming would mix labels from two prompts. Rename the old file to keep it "
                "(e.g. t2dm_ad_v1.json) and run again without --resume.")
        for prior in saved.get("papers", []):
            if prior.get("evidence_type") != PARSE_FAILED:
                done[_record_key(prior)] = prior

    todo = [r for r in records if _record_key(r) not in done]
    abstracts = pubmed_client.fetch_abstracts([r["pmid"] for r in todo]) if todo else {}

    # Input: the classified records so far and whether the run is complete. Output: the payload dict to write.
    def payload_for(classified, complete):
        return {"source": result_path, "model": getattr(client, "model", None), "prompt_version": PROMPT_VERSION,
                "complete": complete,
                "summary": summarize(classified), "papers": classified}

    classified: List[dict] = []
    for i, r in enumerate(records, start=1):
        prior = done.get(_record_key(r))
        if prior is not None:
            classified.append(prior)
            if progress:
                progress(i, len(records), prior, 0.0, True)
            continue
        started = time.monotonic()
        classified.append(classify_paper(client, r, abstracts.get(str(r["pmid"]))))
        _write_payload(out_path, payload_for(classified, complete=(i == len(records))))
        if progress:
            progress(i, len(records), classified[-1], time.monotonic() - started, False)
    payload = payload_for(classified, complete=True)
    _write_payload(out_path, payload)
    return payload


# Print a progress line to stderr so it never mixes with the JSON summary on stdout.
def _print_progress(i, n, rec, seconds, skipped):
    what = "skipped (already done)" if skipped else f"{seconds:.0f}s"
    print(f"[{i}/{n}] {rec['gene_symbol']} / PMID {rec['pmid']}: {rec['evidence_type']} ({what})",
          file=sys.stderr, flush=True)


# Command-line entry point. Imports are local so importing this module (e.g. in tests) needs no network or .env.
def main():
    from common.llm_client import LLMClient
    from PipelineB.pubmed_client import PubMedClient

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pair", required=True, help="t2dm_ad or t2dm_cad (reads outputs/pipeline_b_<pair>.json)")
    ap.add_argument("--result", help="override the input json path")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, help="classify only the first N papers (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="keep papers already classified in the existing output file; only do the rest")
    ap.add_argument("--regrade", action="store_true",
                    help="no LLM: recompute quote checks and the summary of the existing output file, then exit")
    ap.add_argument("--sample-for-review", type=int, default=0, metavar="N",
                    help="also write a blank-labeled CSV of N random papers for hand validation")
    args = ap.parse_args()

    result_path = args.result or os.path.join("outputs", f"pipeline_b_{args.pair}.json")
    out_path = os.path.join(args.out_dir, f"{args.pair}.json")
    if args.regrade:
        print(json.dumps(regrade_saved(out_path)["summary"], indent=2))
        return
    payload = run(result_path, out_path, LLMClient(), PubMedClient(), limit=args.limit,
                  resume=args.resume, progress=_print_progress)
    if args.sample_for_review:
        write_review_sample(payload["papers"], os.path.join(args.out_dir, f"{args.pair}_review_sample.csv"),
                            n=args.sample_for_review)
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
