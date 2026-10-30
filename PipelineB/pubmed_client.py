"""
Stage 1 -- PubMed E-utilities client.

Service: NCBI E-utilities (https://eutils.ncbi.nlm.nih.gov/entrez/eutils),
esearch (gene+disease terms -> PMIDs) then esummary (PMIDs -> titles).
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/. No key required;
set PUBMED_API_KEY/PUBMED_EMAIL in .env to raise the <=3 req/sec limit.
Data: JSON. esearchresult.idlist -> list of PMIDs; result.<pmid>.title.
Used by: PipelineB.stage1_discovery, for citation support on genes
Open Targets already surfaced (not a second discovery source).

Verified live 2026-09-28 against the real E-utilities and BioC-PMC
endpoints (run by the user from their own machine, not the dev
sandbox, which cannot reach these hosts). That live run surfaced one
real bug, since fixed: the BioC-PMC response's top-level JSON value is
a LIST of BioCCollection objects, not a bare dict -- see
_bioc_collections() and the comment in _get_pmc_full_text_paragraphs.

Edge-level literature validation (added 2026-09-27, per committee
feedback -- see claude/committee_feedback_round2_coscientist.md section
3a): `search_citations` above only ever asked "is this gene associated
with this disease" (a gene-level check). It never asked "is there
literature support for this SPECIFIC gene-A-to-gene-B interaction" --
the question needed to justify one edge in a Stage 2 chain (e.g.
`PTPN1 --inhibits--> CAPN1`). `search_edge_citations` below is the new,
edge-level equivalent: it searches PubMed for both gene symbols
co-occurring in the same paper, same as before, but then goes one step
further than a title-only citation.

Abstract-only checking was flagged as insufficient by the user: a real
supporting mention of a specific gene-gene interaction can sit in the
body of a paper as a detail too minor to make it into the paper's own
compressed abstract, so relying on the abstract alone would
systematically under-count real support. The fix implemented here:
for each citation, first try to fetch the paper's real full text from
PubMed Central (PMC) -- but only the open-access subset of PubMed has
a PMC full-text version at all, so this is a best-effort upgrade, not a
guarantee. Two real NCBI calls make this work: ELink
(dbfrom=pubmed&db=pmc) finds whether a PMID has a linked PMC id, and
the BioC-PMC API (https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/)
returns that paper's full text pre-split into sections/paragraphs --
far easier to search over than raw text, since "does any paragraph
mention both genes" is a direct paragraph-by-paragraph check rather
than parsing an unstructured blob. When no PMC full text exists (true
for most citations), this falls back to the abstract via one more
E-utilities call (`efetch.fcgi?rettype=abstract&retmode=text`) --
`search_citations` above never fetches abstract text at all, only
titles via esummary. `EdgeCitation.full_text_checked` distinguishes
these two cases explicitly, since a null result means something
different depending on how much text was actually searched: "full text
was available and searched, no co-mention found" is a much stronger
negative than "only the abstract was available." Collapsing the two
would overstate how thoroughly a null result was checked.

Scope decision (confirmed 2026-09-27, see stage3_confirmation.py's
`_edge_literature_checks`): this is only ever called for a chain's
"bridge edges" -- edges whose both ends are bridge/middle genes, not
the chain's overall disease-anchored endpoints. The two endpoint genes
already get independent GWAS Catalog evidence (see crossdb_client.py);
bridge-to-bridge interactions are the part of a chain with the
thinnest independent support today, so that's where a literature check
adds the most value per query spent.
"""
import json
import os
import re
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional
from xml.etree import ElementTree

import requests

DEFAULT_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# BioC-PMC: structured (paragraph-split) full text for PMC's open-access subset only.
# See https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/
DEFAULT_BIOC_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"
_NO_KEY_MIN_INTERVAL_SECONDS = 0.4  # throttle to stay under NCBI's no-key 3 req/sec limit
# Retry/backoff constants for the new text endpoints (ELink, BioC, efetch abstract) below --
# same shape as crossdb_client.py's GWASCatalogClient._get, reused here since NCBI enforces a
# similar unauthenticated rate limit. Kept separate/smaller than crossdb_client's constants
# since `_get` above already throttles every call to _NO_KEY_MIN_INTERVAL_SECONDS; this only
# covers the rarer case of an actual 429 or connection failure on top of that pacing.
_MAX_RATE_LIMIT_RETRIES = 3
_RATE_LIMIT_INITIAL_BACKOFF_SECONDS = 2.0


def citation_query_term(gene_symbol: str, disease_name: str) -> str:
    # Input: gene symbol + disease name. Output: the exact PubMed query string search_citations
    # sends for that pair. A module-level function (not buried inside search_citations) so Stage 1
    # can record the same string for a search that returned zero hits, where there is no Citation
    # object to carry it.
    return f"{gene_symbol}[Title/Abstract] AND {disease_name}[Title/Abstract]"


@dataclass
class Citation:
    """One PubMed paper found by a gene-level search (Stage 1's `search_citations`).

    Provenance fields (added 2026-10-15) record WHICH search produced this citation, so a gene
    that was searched against both diseases (see stage1_discovery.cite_genes) never has an
    ambiguous record. Bibliographic fields come from the same esummary response the title was
    already read from -- no extra API call. Every field after `title` is optional so a citation
    built by hand or loaded from an older saved run (pmid + title only) still works."""

    pmid: str
    title: str
    # --- search provenance ---
    gene_symbol: Optional[str] = None  # the gene this search was for
    disease_name: Optional[str] = None  # the disease this search was for (disease 1 or 2 of the pair)
    query_term: Optional[str] = None  # exact PubMed query string that found this paper
    search_rank: Optional[int] = None  # 1 = first result in the order esearch returned. NOTE: that
    # is PubMed's default order, which a 2026-10-15 probe showed is recency-weighted, NOT relevance
    # (the same query with sort=relevance returned different, older papers). Rank 1 therefore means
    # 'among the most recently indexed matches', not 'best match'.
    search_total_hits: Optional[int] = None  # how many papers matched that search in total (esearch count)
    # --- bibliographic metadata (from esummary) ---
    pubdate: Optional[str] = None  # publication date as PubMed reports it (often just the year)
    epubdate: Optional[str] = None  # electronic publication date, e.g. "2026 Aug 20"
    journal: Optional[str] = None  # full journal name
    authors: Optional[List[str]] = None  # every author name, in PubMed's order
    doi: Optional[str] = None
    pmcid: Optional[str] = None  # set only if the paper has a PubMed Central (open-access) copy
    pubtype: Optional[List[str]] = None  # e.g. ["Journal Article", "Review"]
    url: Optional[str] = None  # https://pubmed.ncbi.nlm.nih.gov/<pmid>/

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "gene_symbol": self.gene_symbol,
            "disease_name": self.disease_name,
            "query_term": self.query_term,
            "search_rank": self.search_rank,
            "search_total_hits": self.search_total_hits,
            "pubdate": self.pubdate,
            "epubdate": self.epubdate,
            "journal": self.journal,
            "authors": self.authors,
            "doi": self.doi,
            "pmcid": self.pmcid,
            "pubtype": self.pubtype,
            "url": self.url,
        }


def _citation_from_summary(
    pmid: str,
    doc: dict,
    gene_symbol: str,
    disease_name: str,
    query_term: str,
    rank: int,
    total_hits: Optional[int],
) -> Citation:
    # Input: one esummary record (`doc`) plus the search it came from. Output: a fully-tagged Citation.
    article_ids = {a.get("idtype"): a.get("value") for a in (doc.get("articleids") or [])}
    authors = [a.get("name") for a in (doc.get("authors") or []) if a.get("name")]
    return Citation(
        pmid=pmid,
        title=doc.get("title", ""),
        gene_symbol=gene_symbol,
        disease_name=disease_name,
        query_term=query_term,
        search_rank=rank,
        search_total_hits=total_hits,
        pubdate=doc.get("pubdate") or None,
        epubdate=doc.get("epubdate") or None,
        journal=doc.get("fulljournalname") or doc.get("source") or None,
        authors=authors or None,
        doi=article_ids.get("doi"),
        pmcid=article_ids.get("pmc"),
        pubtype=doc.get("pubtype") or None,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    )


@dataclass
class EdgeCitation:
    """One real PubMed citation where two specific genes co-occur in the title/abstract search
    that found it, plus whatever text was actually checked for a passage supporting that specific
    gene-A-to-gene-B relationship -- see module docstring's "Edge-level literature validation"
    note. A citation existing does not by itself mean the interaction is supported -- co-mention
    in the same paper isn't automatically evidence of the specific claimed edge; `matched_text`
    together with `both_genes_confirmed` is what actually says whether a passage naming both genes
    together was found, as opposed to just a paper that happens to discuss both somewhere."""

    pmid: str
    title: str
    full_text_checked: bool  # True: a PMC open-access full-text version was found and searched
    # paragraph-by-paragraph. False: no PMC full text available for this paper -- only the
    # abstract was checked. A False here means a weaker check was done, not that nothing was found.
    matched_text: Optional[str]  # the specific paragraph (full-text case) or abstract (abstract-
    # only case) where both genes were found together, or the abstract text checked either way
    # when full_text_checked is False (kept even if it didn't confirm both genes, for review).
    both_genes_confirmed: bool  # True only if `matched_text` itself actually names both genes
    # together -- NOT merely that this citation matched the original gene_a AND gene_b search.

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "full_text_checked": self.full_text_checked,
            "matched_text": self.matched_text,
            "both_genes_confirmed": self.both_genes_confirmed,
        }


def _gene_pattern(gene_symbol: str) -> "re.Pattern":
    # Input: a gene symbol. Output: a compiled, case-insensitive, word-boundary regex for it --
    # avoids a bare substring match wrongly hitting an unrelated longer token (e.g. "APP" inside
    # "APPROXIMATELY").
    return re.compile(rf"\b{re.escape(gene_symbol)}\b", re.IGNORECASE)


def _contains_both_genes(text: str, gene_a: str, gene_b: str) -> bool:
    # Input: a text passage, two gene symbols. Output: whether both appear in it as whole words.
    return bool(_gene_pattern(gene_a).search(text) and _gene_pattern(gene_b).search(text))


def _first_paragraph_with_both_genes(paragraphs: List[str], gene_a: str, gene_b: str) -> Optional[str]:
    # Input: a paper's full-text paragraphs (BioC passages), two gene symbols. Output: the first
    # paragraph naming both, or None if no single paragraph does (they may still co-occur
    # somewhere across different paragraphs of the same paper -- that weaker signal is not
    # currently surfaced, since a same-paragraph co-mention is the more direct evidence for a
    # specific claimed interaction).
    for paragraph in paragraphs:
        if _contains_both_genes(paragraph, gene_a, gene_b):
            return paragraph
    return None


def _bioc_collections(data) -> List[dict]:
    # Input: parsed JSON from the BioC-PMC endpoint. Output: a list of BioCCollection dicts.
    # Verified live 2026-09-28: the real response is a LIST of collection objects (normally one),
    # not a bare dict -- see the long comment in _get_pmc_full_text_paragraphs for how this was
    # found and confirmed. A bare dict is also accepted as a defensive fallback (e.g. a future
    # API revision, or a different BioC-compatible mirror) rather than assuming only one shape.
    if isinstance(data, list):
        return [collection for collection in data if isinstance(collection, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _abstracts_from_efetch_xml(xml_text: str) -> Dict[str, str]:
    # Input: efetch (db=pubmed, retmode=xml) response text. Output: {pmid: abstract} for every
    # article that has one. A structured abstract's sections are joined as "LABEL: text" so the
    # reader (human or LLM) can tell Methods from Conclusions; an unlabeled one is just joined.
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return {}
    out: Dict[str, str] = {}
    for article in root.iter("PubmedArticle"):
        pmid_el = article.find("MedlineCitation/PMID")
        if pmid_el is None or not (pmid_el.text or "").strip():
            continue
        parts = []
        for el in article.findall("MedlineCitation/Article/Abstract/AbstractText"):
            text = "".join(el.itertext()).strip()  # itertext keeps text inside <i>/<sup> tags
            if not text:
                continue
            label = el.get("Label")
            parts.append(f"{label}: {text}" if label else text)
        if parts:
            out[pmid_el.text.strip()] = " ".join(parts)
    return out


class PubMedClient:
    def __init__(
        self,
        base_url: str = None,
        bioc_base_url: str = None,
        api_key: str = None,
        email: str = None,
        timeout: int = 30,
        sort: Optional[str] = None,
    ):
        self.base_url = (base_url or os.environ.get("PUBMED_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.bioc_base_url = (bioc_base_url or os.environ.get("PUBMED_BIOC_BASE_URL", DEFAULT_BIOC_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.environ.get("PUBMED_API_KEY")
        self.email = email or os.environ.get("PUBMED_EMAIL")
        self.timeout = timeout
        # esearch `sort` for gene-disease citation searches. None keeps PubMed's default (recency) order, which is what
        # every saved run and the citation classifier's hand-labelling set used; "relevance" asks for PubMed's
        # Best Match ranking. Left at None on purpose until the classifier has been validated -- changing the search
        # order changes which 77 papers exist. See claude/citation_evidence_classifier.md. Added 2026-10-18.
        self.sort = sort
        self._last_request_time = 0.0

    def _throttle(self):
        # Input/output: none -- sleeps if needed to respect the no-key rate limit.
        if self.api_key:
            return
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _NO_KEY_MIN_INTERVAL_SECONDS:
            time.sleep(_NO_KEY_MIN_INTERVAL_SECONDS - elapsed)

    def _get(self, path: str, params: dict) -> dict:
        # Input: E-utilities path + params. Output: parsed JSON response.
        params = {**params, "retmode": "json", "tool": "comorbidity-pathway-pipeline"}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email

        self._throttle()
        resp = requests.get(f"{self.base_url}/{path}", params=params, timeout=self.timeout)
        self._last_request_time = time.monotonic()
        resp.raise_for_status()
        return resp.json()

    def _get_text(self, url: str, params: dict = None) -> Optional[str]:
        # Input: a full request URL (BioC full-text, or an eutils text endpoint) + params.
        # Output: the raw text response, or None if it's ultimately unavailable -- a 404/empty
        # result here is a normal, expected outcome (e.g. "no PMC full text for this paper"), not
        # a pipeline error, so this degrades gracefully rather than raising. Retries a 429 or a
        # connection failure with backoff, same shape as crossdb_client.py's GWASCatalogClient._get.
        backoff = _RATE_LIMIT_INITIAL_BACKOFF_SECONDS
        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            self._throttle()
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                self._last_request_time = time.monotonic()
            except requests.exceptions.RequestException:
                if attempt == _MAX_RATE_LIMIT_RETRIES:
                    return None
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 429 and attempt < _MAX_RATE_LIMIT_RETRIES:
                retry_after = resp.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after and retry_after.strip().isdigit() else backoff
                time.sleep(wait_seconds)
                backoff *= 2
                continue
            return None  # 404 or any other non-200 -- treat as "not available", not an error
        return None

    def search_citations(self, gene_symbol: str, disease_name: str, retmax: int = 3) -> List[Citation]:
        # Input: gene symbol, disease name, max results. Output: up to `retmax` Citations.
        # Returns [] if nothing found -- a normal, expected outcome, not an error.
        term = citation_query_term(gene_symbol, disease_name)
        search_params = {"db": "pubmed", "term": term, "retmax": retmax}
        if self.sort:
            search_params["sort"] = self.sort
        search_data = self._get("esearch.fcgi", search_params)
        esearch_result = search_data.get("esearchresult", {})
        pmids = esearch_result.get("idlist", [])
        if not pmids:
            return []
        count = str(esearch_result.get("count", ""))
        total_hits = int(count) if count.isdigit() else None  # all papers matching, not just the retmax returned

        summary_data = self._get("esummary.fcgi", {"db": "pubmed", "id": ",".join(pmids)})
        result = summary_data.get("result", {})
        citations = []
        for rank, pmid in enumerate(pmids, start=1):
            doc = result.get(pmid)
            if doc:
                citations.append(_citation_from_summary(pmid, doc, gene_symbol, disease_name, term, rank, total_hits))
        return citations

    def search_edge_citations(self, gene_a: str, gene_b: str, retmax: int = 3) -> List[EdgeCitation]:
        # Input: two gene symbols (a chain edge's two endpoints), max results. Output: up to
        # `retmax` EdgeCitations -- real papers where both gene symbols co-occur in the title or
        # abstract, each checked for an actual co-mentioning passage (full text if available, else
        # abstract) rather than just returning a title. [] if nothing found. See module docstring.
        term = f"{gene_a}[Title/Abstract] AND {gene_b}[Title/Abstract]"
        search_data = self._get("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": retmax})
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        summary_data = self._get("esummary.fcgi", {"db": "pubmed", "id": ",".join(pmids)})
        result = summary_data.get("result", {})

        edge_citations: List[EdgeCitation] = []
        for pmid in pmids:
            doc = result.get(pmid)
            if not doc:
                continue
            edge_citations.append(self._check_edge_support(pmid, doc.get("title", ""), gene_a, gene_b))
        return edge_citations

    def _check_edge_support(self, pmid: str, title: str, gene_a: str, gene_b: str) -> EdgeCitation:
        # Input: one citation's pmid/title + the two gene symbols to look for together in it.
        # Output: its EdgeCitation, preferring PMC full text (searched paragraph-by-paragraph)
        # over the abstract when a full-text version exists -- see module docstring.
        paragraphs = self._get_pmc_full_text_paragraphs(pmid)
        if paragraphs is not None:
            match = _first_paragraph_with_both_genes(paragraphs, gene_a, gene_b)
            return EdgeCitation(
                pmid=pmid,
                title=title,
                full_text_checked=True,
                matched_text=match,
                both_genes_confirmed=match is not None,
            )

        abstract = self._get_abstract_text(pmid)
        both_confirmed = bool(abstract) and _contains_both_genes(abstract, gene_a, gene_b)
        return EdgeCitation(
            pmid=pmid,
            title=title,
            full_text_checked=False,
            matched_text=abstract,
            both_genes_confirmed=both_confirmed,
        )

    def _get_pmc_full_text_paragraphs(self, pmid: str) -> Optional[List[str]]:
        # Input: a PMID. Output: that paper's full-text paragraphs if it has an open-access PMC
        # version, else None (normal -- most PubMed records don't; not an error). Two real NCBI
        # calls: ELink (pubmed->pmc) to find the linked PMC id, then the BioC-PMC API for
        # structured, paragraph-split full text.
        #
        # Verified live 2026-09-28 against the real BioC-PMC endpoint: the top-level JSON value
        # is a LIST of BioCCollection objects (normally just one), each holding "documents" ->
        # "passages" -> "text" -- NOT a bare dict as originally assumed here. That assumption
        # (data.get("documents", ...)) raised AttributeError: 'list' object has no attribute
        # 'get' the first time this ran against a live paper. _bioc_collections() below
        # normalizes both the real list-wrapped shape and a bare-dict shape (kept as a fallback
        # in case a future API revision or mirror ever drops the wrapper).
        pmcid = self._pmid_to_pmcid(pmid)
        if not pmcid:
            return None
        text = self._get_text(f"{self.bioc_base_url}/BioC_json/{pmcid}/unicode")
        if not text:
            return None
        try:
            data = json.loads(text)
        except ValueError:
            return None
        paragraphs = []
        for collection in _bioc_collections(data):
            for document in collection.get("documents", []):
                for passage in document.get("passages", []):
                    passage_text = (passage.get("text") or "").strip()
                    if passage_text:
                        paragraphs.append(passage_text)
        return paragraphs or None

    def _pmid_to_pmcid(self, pmid: str) -> Optional[str]:
        # Input: a PMID. Output: its linked PMC id (e.g. "PMC1234567"), or None if no PMC link
        # exists -- i.e. no open-access full text, the normal case for most PubMed records.
        try:
            data = self._get("elink.fcgi", {"dbfrom": "pubmed", "db": "pmc", "id": pmid})
        except requests.HTTPError:
            return None
        for linkset in data.get("linksets", []):
            for linksetdb in linkset.get("linksetdbs", []):
                links = linksetdb.get("links", [])
                if links:
                    return f"PMC{links[0]}"
        return None

    def _get_abstract_text(self, pmid: str) -> Optional[str]:
        # Input: a PMID. Output: its plain abstract text via efetch, or None if unavailable.
        # search_citations()/esummary above only ever returns a title -- this is the one place
        # this client fetches real abstract text.
        params = {"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "text"}
        if self.api_key:
            params["api_key"] = self.api_key
        text = self._get_text(f"{self.base_url}/efetch.fcgi", params)
        return text.strip() if text else None

    def fetch_abstracts(self, pmids: List[str], batch_size: int = 50) -> Dict[str, str]:
        # Input: PMIDs. Output: {pmid: abstract text} for the ones that have an abstract; a PMID
        # with no abstract (editorials, some letters) or a failed request is simply absent, so
        # callers must treat "missing" as a normal outcome. One efetch call per `batch_size`
        # PMIDs, parsed from efetch's XML so structured abstracts keep their section labels
        # ("BACKGROUND: ...", "METHODS: ...") and the text has none of the author/affiliation
        # header that `_get_abstract_text`'s plain-text mode includes.
        abstracts: Dict[str, str] = {}
        unique = list(dict.fromkeys(str(p) for p in pmids))
        for start in range(0, len(unique), batch_size):
            batch = unique[start:start + batch_size]
            params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
            if self.api_key:
                params["api_key"] = self.api_key
            text = self._get_text(f"{self.base_url}/efetch.fcgi", params)
            if text:
                abstracts.update(_abstracts_from_efetch_xml(text))
        return abstracts


class StubPubMedClient:
    """Fixed, injectable responses for offline tests -- no network call."""

    def __init__(self, citations_by_gene: dict, edge_citations_by_pair: dict = None):
        self.citations_by_gene = citations_by_gene
        # Keyed by (gene_a, gene_b) tuples; looked up order-insensitively (see search_edge_citations
        # below), since the real query is symmetric in gene_a/gene_b.
        self.edge_citations_by_pair = edge_citations_by_pair or {}
        self.calls = []
        self.edge_calls = []

    def search_citations(self, gene_symbol: str, disease_name: str, retmax: int = 3) -> List[Citation]:
        self.calls.append((gene_symbol, disease_name, retmax))
        # Mirrors the real client's contract: every returned citation is tagged with the search
        # that produced it. Copies (dataclasses.replace), not the stored objects, so one gene
        # searched against two diseases yields two independently-tagged records.
        query_term = citation_query_term(gene_symbol, disease_name)
        found = self.citations_by_gene.get(gene_symbol, [])[:retmax]
        return [
            replace(c, gene_symbol=gene_symbol, disease_name=disease_name, query_term=query_term, search_rank=rank)
            for rank, c in enumerate(found, start=1)
        ]

    def search_edge_citations(self, gene_a: str, gene_b: str, retmax: int = 3) -> List[EdgeCitation]:
        self.edge_calls.append((gene_a, gene_b, retmax))
        key = (gene_a, gene_b) if (gene_a, gene_b) in self.edge_citations_by_pair else (gene_b, gene_a)
        return self.edge_citations_by_pair.get(key, [])[:retmax]
