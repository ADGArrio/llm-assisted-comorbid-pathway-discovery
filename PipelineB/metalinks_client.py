"""
Agent 2 -- MetalinksDB client (gene<->metabolite edges).

Service: MetalinksDB, accessed via the liana-py package (`pip install
liana`), which downloads a local SQLite snapshot on first use -- not a
live per-query API call. Confirmed reachable and working from this
environment (unlike Open Targets/OmniPath/GWAS Catalog/GTEx/DisGeNET,
which are network-blocked here).
Data: liana.resource.get_metalinks() returns a pandas DataFrame with
columns hmdb, uniprot, gene_symbol, metabolite, mor, transport_direction,
type, source. `type` is "lr" (ligand-receptor: the metabolite acts ON
the gene/receptor) or "pd" (production-degradation: the gene/enzyme acts
ON the metabolite). `mor` (mode of regulation) is 1/-1/0, mapped here to
activates/inhibits/unknown ("lr" rows) or produces/degrades/unknown
("pd" rows).
Used by: PipelineB.agent2_chain's BFS, as a second edge source alongside
OmniPath so metabolites (glucose, cholesterol, fatty acids, ...) can be
genuine intermediate hops in a chain: GeneA --produces--> Metabolite
--activates--> GeneB.
"""
from dataclasses import dataclass
from typing import List, Optional

_MOR_TO_LR_TYPE = {1: "activates", -1: "inhibits", 0: "unknown"}
_MOR_TO_PD_TYPE = {1: "produces", -1: "degrades", 0: "unknown"}


@dataclass
class MetaboliteEdge:
    source_kind: str  # "gene" | "metabolite"
    source_id: str
    target_kind: str  # "gene" | "metabolite"
    target_id: str
    interaction_type: str  # activates | inhibits | produces | degrades | unknown
    hmdb_id: Optional[str]  # set for whichever end is the metabolite
    uniprot_id: Optional[str]  # set for whichever end is the gene
    edge_id: str

    def to_dict(self) -> dict:
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "interaction_type": self.interaction_type,
            "hmdb_id": self.hmdb_id,
            "uniprot_id": self.uniprot_id,
            "edge_id": self.edge_id,
        }


class MetalinksClient:
    def __init__(self, db_path: str = None):
        # db_path: local SQLite cache path, passed straight to liana.resource.get_metalinks().
        # None means liana downloads to the current working directory on first use (set
        # METALINKS_DB_PATH-equivalent explicitly if you want a stable cache location instead).
        self.db_path = db_path
        self._df = None

    def _load(self):
        # Input: none. Output: the full lr+pd DataFrame, cached after the first (one-time
        # download) call.
        if self._df is None:
            import liana

            self._df = liana.resource.get_metalinks(db_path=self.db_path, types=["lr", "pd"])
        return self._df

    def get_gene_to_metabolite(self, gene_symbols: List[str]) -> List[MetaboliteEdge]:
        # Input: gene symbols (candidate enzymes/producers). Output: gene->metabolite edges
        # from "pd" rows (the gene produces/degrades the metabolite).
        if not gene_symbols:
            return []
        df = self._load()
        genes = {g.upper() for g in gene_symbols}
        rows = df[(df["type"] == "pd") & (df["gene_symbol"].str.upper().isin(genes))]
        return [
            MetaboliteEdge(
                source_kind="gene",
                source_id=row["gene_symbol"],
                target_kind="metabolite",
                target_id=row["metabolite"],
                interaction_type=_MOR_TO_PD_TYPE.get(row["mor"], "unknown"),
                hmdb_id=row["hmdb"],
                uniprot_id=row["uniprot"],
                edge_id=f"{row['gene_symbol']}->{row['metabolite']}",
            )
            for _, row in rows.iterrows()
        ]

    def get_metabolite_to_gene(self, metabolite_names: List[str]) -> List[MetaboliteEdge]:
        # Input: metabolite names (as they appear in MetalinksDB's `metabolite` column).
        # Output: metabolite->gene edges from "lr" rows (the metabolite acts on the gene/receptor).
        if not metabolite_names:
            return []
        df = self._load()
        names = {m.lower() for m in metabolite_names}
        rows = df[(df["type"] == "lr") & (df["metabolite"].str.lower().isin(names))]
        return [
            MetaboliteEdge(
                source_kind="metabolite",
                source_id=row["metabolite"],
                target_kind="gene",
                target_id=row["gene_symbol"],
                interaction_type=_MOR_TO_LR_TYPE.get(row["mor"], "unknown"),
                hmdb_id=row["hmdb"],
                uniprot_id=row["uniprot"],
                edge_id=f"{row['metabolite']}->{row['gene_symbol']}",
            )
            for _, row in rows.iterrows()
        ]


class StubMetalinksClient:
    """Fixed, injectable edges for offline tests -- no download/network call."""

    def __init__(self, gene_to_metabolite: List[MetaboliteEdge] = None, metabolite_to_gene: List[MetaboliteEdge] = None):
        self.gene_to_metabolite_edges = gene_to_metabolite or []
        self.metabolite_to_gene_edges = metabolite_to_gene or []
        self.calls: List[tuple] = []

    def get_gene_to_metabolite(self, gene_symbols: List[str]) -> List[MetaboliteEdge]:
        genes = {g.upper() for g in gene_symbols}
        self.calls.append(("gene_to_metabolite", sorted(genes)))
        return [e for e in self.gene_to_metabolite_edges if e.source_id.upper() in genes]

    def get_metabolite_to_gene(self, metabolite_names: List[str]) -> List[MetaboliteEdge]:
        names = {m.lower() for m in metabolite_names}
        self.calls.append(("metabolite_to_gene", sorted(names)))
        return [e for e in self.metabolite_to_gene_edges if e.source_id.lower() in names]
