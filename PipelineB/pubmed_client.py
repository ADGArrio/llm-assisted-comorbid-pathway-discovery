"""PubMed client over NCBI E-utilities (esearch, esummary, efetch, ELink) and the BioC-PMC full-text API. No key is required;
set PUBMED_API_KEY and PUBMED_EMAIL in .env to raise the 3 requests/sec limit.

Gene-level search (Stage 1): search_citations finds papers for a gene and disease. Edge-level search (Stage 3):
search_edge_citations finds papers where two genes co-occur and checks the paper's PMC open-access full text (paragraph by
paragraph) or, failing that, its abstract for a passage that names both genes. fetch_abstracts returns abstract text for
the citation classifier.
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
# BioC-PMC: paragraph-split full text for PMC's open-access subset only.
DEFAULT_BIOC_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"
_NO_KEY_MIN_INTERVAL_SECONDS = 0.4  # throttle to stay under NCBI's no-key limit of 3 requests/sec
# Retry/backoff constants for the text endpoints (ELink, BioC, efetch); smaller than the GWAS client's because _get
# already paces every call.
_MAX_RATE_LIMIT_RETRIES = 3
_RATE_LIMIT_INITIAL_BACKOFF_SECONDS = 2.0


# Build the PubMed query for a gene-disease search. Input: gene symbol and disease name. Output: the exact query string
# search_citations sends (also recorded for searches that return nothing).
def citation_query_term(gene_symbol: str, disease_name: str) -> str:
    return f"{gene_symbol}[Title/Abstract] AND {disease_name}[Title/Abstract]"


# One PubMed paper found by a gene-level search, tagged with the search that produced it.
@dataclass
class Citation:

    pmid: str
    title: str
    # Which search produced this citation.
    gene_symbol: Optional[str] = None  # the gene this search was for
    disease_name: Optional[str] = None  # the disease this search was for (disease 1 or 2 of the pair)
    query_term: Optional[str] = None  # exact PubMed query string that found this paper
    search_rank: Optional[int] = None  # 1 = first result in the order esearch returned
    # That is PubMed's default order, which is recency-weighted, so rank 1 means among the most recently indexed
    # matches, not the best match.
    search_total_hits: Optional[int] = None  # how many papers matched that search in total (esearch count)
    # Bibliographic metadata from esummary.
    pubdate: Optional[str] = None  # publication date as PubMed reports it (often just the year)
    epubdate: Optional[str] = None  # electronic publication date, e.g. "2026 Aug 20"
    journal: Optional[str] = None  # full journal name
    authors: Optional[List[str]] = None  # every author name, in PubMed's order
    doi: Optional[str] = None
    pmcid: Optional[str] = None  # set only if the paper has a PubMed Central (open-access) copy
    pubtype: Optional[List[str]] = None  # e.g. ["Journal Article", "Review"]
    url: Optional[str] = None  # https://pubmed.ncbi.nlm.nih.gov/<pmid>/

    # Output: the citation as a plain dict.
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


# Build a Citation from an esummary record. Input: the record and the search it came from. Output: a fully tagged
# Citation.
def _citation_from_summary(
    pmid: str,
    doc: dict,
    gene_symbol: str,
    disease_name: str,
    query_term: str,
    rank: int,
    total_hits: Optional[int],
) -> Citation:
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


# One PubMed paper where two genes co-occur, with the text checked for a passage naming both.
@dataclass
class EdgeCitation:

    pmid: str
    title: str
    full_text_checked: bool  # True: PMC full text was searched; False: only the abstract (a weaker check, not proof of absence)
    matched_text: Optional[str]  # paragraph or abstract where both genes appear together; the checked abstract if there is no full text
    both_genes_confirmed: bool  # True only if matched_text itself names both genes

    # Output: the citation as a plain dict.
    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "full_text_checked": self.full_text_checked,
            "matched_text": self.matched_text,
            "both_genes_confirmed": self.both_genes_confirmed,
        }


# Input: a gene symbol. Output: a case-insensitive, word-boundary regex for it (so "APP" does not match inside
# "APPROXIMATELY").
def _gene_pattern(gene_symbol: str) -> "re.Pattern":
    return re.compile(rf"\b{re.escape(gene_symbol)}\b", re.IGNORECASE)


# Input: a text passage and two gene symbols. Output: whether both appear in it as whole words.
def _contains_both_genes(text: str, gene_a: str, gene_b: str) -> bool:
    return bool(_gene_pattern(gene_a).search(text) and _gene_pattern(gene_b).search(text))


# Input: a paper's full-text paragraphs and two gene symbols. Output: the first paragraph naming both, or None.
def _first_paragraph_with_both_genes(paragraphs: List[str], gene_a: str, gene_b: str) -> Optional[str]:
    for paragraph in paragraphs:
        if _contains_both_genes(paragraph, gene_a, gene_b):
            return paragraph
    return None


# Normalise a BioC-PMC response. Input: parsed JSON, which is a list of BioCCollection objects (normally one); a bare
# dict is also accepted. Output: a list of collection dicts.
def _bioc_collections(data) -> List[dict]:
    if isinstance(data, list):
        return [collection for collection in data if isinstance(collection, dict)]
    if isinstance(data, dict):
        return [data]
    return []


# Parse abstracts from efetch XML. Input: the response text. Output: {pmid: abstract} for every article that has one;
# structured abstracts keep their section labels ("LABEL: text").
def _abstracts_from_efetch_xml(xml_text: str) -> Dict[str, str]:
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


# Queries PubMed and PubMed Central.
class PubMedClient:
    # Input: optional API key, email, base URLs, timeout and esearch sort order. Output: none.
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
        # esearch sort order for gene-disease citation searches. None keeps PubMed's default (recency) order, which
        # every saved run and the citation classifier's labelled set used; "relevance" asks for Best Match ranking.
        self.sort = sort
        self._last_request_time = 0.0

    # Sleep if needed to stay under the request-rate limit.
    def _throttle(self):
        if self.api_key:
            return
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _NO_KEY_MIN_INTERVAL_SECONDS:
            time.sleep(_NO_KEY_MIN_INTERVAL_SECONDS - elapsed)

    # GET an E-utilities endpoint. Input: path and params. Output: the parsed JSON response.
    def _get(self, path: str, params: dict) -> dict:
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

    # GET a text endpoint, retrying a 429 or connection failure with backoff. Input: a full URL and params. Output: the
    # response text, or None if unavailable (a 404 or empty result is a normal outcome, not an error).
    def _get_text(self, url: str, params: dict = None) -> Optional[str]:
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

    # Input: gene symbol, disease name and maximum results. Output: up to retmax Citations; an empty list if nothing is
    # found.
    def search_citations(self, gene_symbol: str, disease_name: str, retmax: int = 3) -> List[Citation]:
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

    # Input: two gene symbols (an edge's endpoints) and maximum results. Output: up to retmax EdgeCitations for papers
    # where both genes appear in the title or abstract, each checked for a passage naming both; an empty list if nothing
    # is found.
    def search_edge_citations(self, gene_a: str, gene_b: str, retmax: int = 3) -> List[EdgeCitation]:
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

    # Check one paper for an edge. Input: its pmid and title and the two gene symbols. Output: an EdgeCitation, using
    # PMC full text when available and the abstract otherwise.
    def _check_edge_support(self, pmid: str, title: str, gene_a: str, gene_b: str) -> EdgeCitation:
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

    # Input: a PMID. Output: the paper's full-text paragraphs if it has an open-access PMC copy (found through ELink,
    # then fetched from BioC-PMC), else None.
    def _get_pmc_full_text_paragraphs(self, pmid: str) -> Optional[List[str]]:
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

    # Input: a PMID. Output: its linked PMC id (e.g. "PMC1234567"), or None if there is no open-access copy.
    def _pmid_to_pmcid(self, pmid: str) -> Optional[str]:
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

    # Input: a PMID. Output: its plain abstract text via efetch, or None if unavailable.
    def _get_abstract_text(self, pmid: str) -> Optional[str]:
        params = {"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "text"}
        if self.api_key:
            params["api_key"] = self.api_key
        text = self._get_text(f"{self.base_url}/efetch.fcgi", params)
        return text.strip() if text else None

    # Input: PMIDs and a batch size. Output: {pmid: abstract text} for those that have an abstract (editorials and
    # failed requests are absent); one efetch call per batch.
    def fetch_abstracts(self, pmids: List[str], batch_size: int = 50) -> Dict[str, str]:
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


# Test double returning fixed citations with no network call.
class StubPubMedClient:

    # Input: citations by gene and, optionally, edge citations by gene pair. Output: none.
    def __init__(self, citations_by_gene: dict, edge_citations_by_pair: dict = None):
        self.citations_by_gene = citations_by_gene
        # Keyed by (gene_a, gene_b) and looked up in either order, since the real query is symmetric.
        self.edge_citations_by_pair = edge_citations_by_pair or {}
        self.calls = []
        self.edge_calls = []

    # Input: gene symbol, disease name and maximum results. Output: copies of the stored citations, tagged with this
    # search.
    def search_citations(self, gene_symbol: str, disease_name: str, retmax: int = 3) -> List[Citation]:
        self.calls.append((gene_symbol, disease_name, retmax))
        # Every returned citation is tagged with its search, as the real client does; copies are returned so one gene
        # searched against two diseases yields independently tagged records.
        query_term = citation_query_term(gene_symbol, disease_name)
        found = self.citations_by_gene.get(gene_symbol, [])[:retmax]
        return [
            replace(c, gene_symbol=gene_symbol, disease_name=disease_name, query_term=query_term, search_rank=rank)
            for rank, c in enumerate(found, start=1)
        ]

    # Input: two gene symbols and maximum results. Output: the stored edge citations for that pair, in either order.
    def search_edge_citations(self, gene_a: str, gene_b: str, retmax: int = 3) -> List[EdgeCitation]:
        self.edge_calls.append((gene_a, gene_b, retmax))
        key = (gene_a, gene_b) if (gene_a, gene_b) in self.edge_citations_by_pair else (gene_b, gene_a)
        return self.edge_citations_by_pair.get(key, [])[:retmax]
