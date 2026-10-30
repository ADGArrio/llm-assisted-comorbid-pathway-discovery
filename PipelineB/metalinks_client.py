"""MetalinksDB client for gene-metabolite edges, used by Stage 2 as a second edge source beside OmniPath so metabolites can be
intermediate hops (GeneA --produces--> Metabolite --activates--> GeneB). Reads a local SQLite snapshot through the liana package.
"lr" rows are ligand-receptor (metabolite acts on the gene); "pd" rows are production-degradation (gene acts on the metabolite).
"""
from dataclasses import dataclass
from typing import List, Optional

_MOR_TO_LR_TYPE = {1: "activates", -1: "inhibits", 0: "unknown"}
_MOR_TO_PD_TYPE = {1: "produces", -1: "degrades", 0: "unknown"}


# One directed gene-metabolite edge.
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

    # Output: the edge as a plain dict.
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


# Queries the MetalinksDB snapshot.
class MetalinksClient:
    # Input: optional local cache path for the database (default: liana downloads to the working directory on first
    # use). Output: none.
    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self._df = None

    # Load the full table once and cache it. Output: the DataFrame.
    def _load(self):
        if self._df is None:
            import liana

            self._df = liana.resource.get_metalinks(db_path=self.db_path, types=["lr", "pd"])
        return self._df

    # Input: gene symbols. Output: edges from genes to the metabolites they produce or degrade ("pd" rows).
    def get_gene_to_metabolite(self, gene_symbols: List[str]) -> List[MetaboliteEdge]:
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

    # Input: metabolite names as they appear in MetalinksDB. Output: edges from metabolites to the genes they act on
    # ("lr" rows).
    def get_metabolite_to_gene(self, metabolite_names: List[str]) -> List[MetaboliteEdge]:
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


# Test double returning fixed edges with no download.
class StubMetalinksClient:

    # Input: the gene-to-metabolite and metabolite-to-gene edge lists to serve. Output: none.
    def __init__(self, gene_to_metabolite: List[MetaboliteEdge] = None, metabolite_to_gene: List[MetaboliteEdge] = None):
        self.gene_to_metabolite_edges = gene_to_metabolite or []
        self.metabolite_to_gene_edges = metabolite_to_gene or []
        self.calls: List[tuple] = []

    # Input: gene symbols. Output: the fixed edges whose source is one of them.
    def get_gene_to_metabolite(self, gene_symbols: List[str]) -> List[MetaboliteEdge]:
        genes = {g.upper() for g in gene_symbols}
        self.calls.append(("gene_to_metabolite", sorted(genes)))
        return [e for e in self.gene_to_metabolite_edges if e.source_id.upper() in genes]

    # Input: metabolite names. Output: the fixed edges whose source is one of them.
    def get_metabolite_to_gene(self, metabolite_names: List[str]) -> List[MetaboliteEdge]:
        names = {m.lower() for m in metabolite_names}
        self.calls.append(("metabolite_to_gene", sorted(names)))
        return [e for e in self.metabolite_to_gene_edges if e.source_id.lower() in names]
