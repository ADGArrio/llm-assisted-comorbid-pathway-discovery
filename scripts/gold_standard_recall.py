"""Gold-standard structural coverage: compares each endorsed chain's genes with real KEGG gene sets, per disease pair.

coverage = |chain genes intersect gold set| / |gold set|, and chain_precision = |chain genes intersect gold set| / |chain genes|. These are
proxies: the gold sets are per-disease pathway or disease gene sets, not a gold standard for the specific T2DM-to-X bridge mechanism, so
chain_precision (the share of a chain's own genes independently known to matter for the disease) is the main figure for a short chain.
GOLD_CAD_BRIDGE (the AGE-RAGE pathway) is the one set that is itself a diabetes-to-vascular-disease bridge.

Gold sets, fetched from the KEGG REST API on 2026-10-11: GOLD_T2DM (hsa04930, full gene section); GOLD_AD (hsa05010, a curated 40-gene
subset of AD-specific modules, leaving out generic oxidative-phosphorylation and proteasome subunits); GOLD_CAD_BRIDGE (hsa04933, full);
GOLD_CAD_ATHERO (hsa05417, full); GOLD_CAD_STRICT (disease H01742, six genes linked directly to CAD).

Usage:
    python3 -m scripts.gold_standard_recall --pair t2dm_cad
    python3 -m scripts.gold_standard_recall --pair t2dm_ad
    python3 -m scripts.gold_standard_recall            # default: both pairs

Output: outputs/gold_standard_recall/{pair_id}.json (per-chain, per-gold-set detail and summary) plus a printed table. No network calls.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
RESULT_DIR = OUTPUTS_DIR / "gold_standard_recall"

# KEGG hsa04930 "Type II diabetes mellitus", full gene section.
GOLD_T2DM: Set[str] = {
    "SOCS4", "MTOR", "GCK", "HK1", "HK2", "HK3", "IKBKB", "INS", "INSR", "PDX1", "IRS1",
    "KCNJ11", "MAFA", "PIK3CA", "PIK3CB", "PIK3CD", "PIK3R1", "PIK3R2", "PKLR", "PKM",
    "PRKCD", "PRKCE", "PRKCZ", "MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK10", "SLC2A2",
    "SLC2A4", "ABCC8", "TNF", "CACNA1A", "CACNA1B", "CACNA1C", "CACNA1D", "CACNA1E",
    "HKDC1", "IRS4", "PIK3R3", "SOCS1", "IRS2", "SOCS2", "CACNA1G", "SOCS3", "ADIPOQ",
}

# KEGG hsa05010 "Alzheimer disease", curated subset of AD-specific modules.
GOLD_AD: Set[str] = {
    "APP", "PSEN1", "PSEN2", "APOE", "BACE1", "BACE2",
    "GRIN1", "GRIN2A", "GRIN2B", "GRM5", "CACNA1C", "CALM1", "CALM2", "CALM3",
    "ITPR1", "ITPR2", "ITPR3",
    "ATF6", "ERN1", "EIF2AK3", "XBP1", "ATF4", "DDIT3", "EIF2S1",
    "CASP3", "CASP7", "CASP8", "CASP9", "APAF1", "BID", "BAD",
    "MAPT", "CDK5", "GSK3B", "MAPK8", "MAP3K5",
    "AGER", "LRP1", "IDE", "MME", "ADAM10",
}

# KEGG hsa04933 "AGE-RAGE signaling pathway in diabetic complications", full gene section (101 entries).
GOLD_CAD_BRIDGE: Set[str] = {
    "AGER", "CYBB", "NOX1", "NOX4",
    "PRKCA", "PRKCB", "PRKCD", "PRKCE", "PRKCZ",
    "PLCB1", "PLCB2", "PLCB3", "PLCB4", "PLCD1", "PLCD3", "PLCD4", "PLCE1", "PLCG1", "PLCG2",
    "PIK3CA", "PIK3CB", "PIK3CD", "PIK3R1", "PIK3R2", "PIK3R3", "AKT1", "AKT2", "AKT3",
    "FOXO1", "PIM1",
    "HRAS", "KRAS", "NRAS", "RAC1", "CDC42", "DIAPH1",
    "MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK10", "MAPK11", "MAPK12", "MAPK13", "MAPK14",
    "JUN", "EGR1",
    "JAK2", "STAT1", "STAT3", "STAT5A", "STAT5B",
    "NFKB1", "RELA", "NFATC1", "IL1A", "IL1B", "IL6", "CXCL8", "TNF", "CCL2",
    "ICAM1", "VCAM1", "SELE",
    "F3", "THBD", "SERPINE1", "EDN1", "NOS3", "AGT", "AGTR1",
    "VEGFA", "VEGFB", "VEGFC", "VEGFD", "MMP2", "FN1",
    "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD4",
    "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL4A2", "COL4A3", "COL4A4", "COL4A5", "COL4A6",
    "CDK4", "CDKN1B", "CCND1", "BAX", "BCL2", "CASP3",
}

# KEGG hsa05417 "Lipid and atherosclerosis", full gene section.
GOLD_CAD_ATHERO: Set[str] = {
    "OLR1", "LDLR", "VLDLR", "APOA1", "APOA4", "APOB", "ABCA1", "ABCG1", "CD36", "AGER",
    "CYBA", "CYBB", "NCF1", "NCF2", "NCF4", "NOX1", "SOD2", "NOS3", "NFE2L2",
    "CYP1A1", "CYP2A6", "CYP2A7", "CYP2B6", "CYP2C8", "CYP2C9", "CYP2J2",
    "TLR2", "TLR4", "TLR6", "TIRAP", "TICAM1", "TICAM2", "MYD88", "IRAK1", "IRAK4",
    "TAB1", "TAB2", "MAP3K7", "TRAF2", "TRAF3", "TRAF6", "TBK1", "IKBKE", "IRF3", "IRF7",
    "LY96", "LBP", "CD14",
    "NLRP3", "PYCARD", "CASP1", "IL18", "IL1B",
    "CHUK", "IKBKB", "IKBKG", "NFKB1", "NFKBIA", "RELA", "TANK",
    "IFNA1", "IFNA2", "IFNA4", "IFNA5", "IFNA6", "IFNA7", "IFNA8", "IFNA10", "IFNA13",
    "IFNA14", "IFNA16", "IFNA17", "IFNA21", "IFNB1",
    "IL6", "IL12A", "IL12B", "TNF", "TNFRSF1A", "CXCL1", "CXCL2", "CXCL3", "CXCL8",
    "CCL2", "CCL3", "CCL3L1", "CCL3L3", "CCL5", "ICAM1", "VCAM1", "SELE", "SELP",
    "CD40", "CD40LG", "MMP1", "MMP3", "MMP9",
    "MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK10", "MAPK11", "MAPK12", "MAPK13", "MAPK14",
    "MAP2K3", "MAP2K4", "MAP2K6", "MAP2K7", "MAP3K5", "HRAS", "KRAS", "NRAS", "JUN",
    "RHOA", "RAC1", "CDC42", "RAP1A", "RAP1B", "VAV1", "VAV2", "VAV3", "ARHGEF1", "ROCK2",
    "JAK2", "LYN", "SRC", "PTK2", "STAT3", "PRKCA",
    "PIK3CA", "PIK3CB", "PIK3CD", "PIK3R1", "PIK3R2", "PIK3R3", "PDPK1", "AKT1", "AKT2",
    "AKT3", "GSK3B",
    "CALM1", "CALM2", "CALM3", "CALML3", "CALML4", "CALML5", "CALML6",
    "CAMK2A", "CAMK2B", "CAMK2D", "CAMK2G",
    "PLCB1", "PLCB2", "PLCB3", "PLCB4", "PLCG1", "ITPR1",
    "PPP3CA", "PPP3CB", "PPP3CC", "PPP3R1", "PPP3R2", "NFATC1", "NFATC2", "NFATC3",
    "ERN1", "ATF4", "ATF6", "XBP1", "EIF2AK3", "EIF2S1", "DDIT3", "ERO1A", "HSPA5", "HSP90B1",
    "HSPA1A", "HSPA1B", "HSPA1L", "HSPA2", "HSPA4", "HSPA6", "HSPA8", "HSP90AA1",
    "HSP90AB1", "HSPD1",
    "FAS", "FASLG", "TNFSF10", "TNFRSF10A", "TNFRSF10B", "CASP3", "CASP6", "CASP7",
    "CASP8", "CASP9", "APAF1", "CYCS", "BAX", "BAD", "BID", "BCL2", "BCL2L1", "TP53",
    "FOS", "POU2F1", "POU2F2", "POU2F3", "PPARG", "RXRA", "RXRB", "RXRG",
    "MIB1", "MIB2",
}

# KEGG DISEASE H01742 "Coronary artery disease", direct gene field.
GOLD_CAD_STRICT: Set[str] = {"MEF2A", "LRP6", "CX3CR1", "KALRN", "MMP3", "CD36"}

GOLD_SETS = {
    "t2dm_ad": {"disease1": ("T2DM", GOLD_T2DM), "disease2": [("AD", GOLD_AD)]},
    "t2dm_cad": {
        "disease1": ("T2DM", GOLD_T2DM),
        "disease2": [
            ("CAD_bridge_AGE_RAGE", GOLD_CAD_BRIDGE),
            ("CAD_atherosclerosis", GOLD_CAD_ATHERO),
            ("CAD_strict_disease_genes", GOLD_CAD_STRICT),
        ],
    },
}


# Input: a chain dict. Output: its genes in order.
def gene_list_for_chain(chain: dict) -> List[str]:
    edges = chain["edges"]
    return [edges[0]["source_gene"]] + [e["target_gene"] for e in edges]


# Input: a chain's genes and a gold gene set. Output: the matched genes, chain_precision, gold_coverage and the gold set
# size.
def score_against_gold(chain_genes: List[str], gold: Set[str]) -> dict:
    chain_set = set(chain_genes)
    matched = sorted(chain_set & gold)
    return {
        "matched_genes": matched,
        "chain_precision": (len(matched) / len(chain_set)) if chain_set else 0.0,
        "gold_coverage": (len(matched) / len(gold)) if gold else 0.0,
        "gold_set_size": len(gold),
    }


# Input: pair id. Output: per-chain scores against each gold set for that pair, with a summary of mean chain_precision
# per gold set.
def run_for_pair(pair_id: str) -> dict:
    path = OUTPUTS_DIR / f"pipeline_b_{pair_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run Pipeline B for {pair_id} first.")
    data = json.loads(path.read_text())
    chains = data["stage2"]["chains"]
    gold_cfg = GOLD_SETS[pair_id]
    d1_label, d1_gold = gold_cfg["disease1"]

    chain_results = []
    for i, chain in enumerate(chains):
        genes = gene_list_for_chain(chain)
        entry = {
            "chain_index": i,
            "genes": genes,
            "elo": chain.get("elo_rating"),
            "scores": {d1_label: score_against_gold(genes, d1_gold)},
        }
        for d2_label, d2_gold in gold_cfg["disease2"]:
            entry["scores"][d2_label] = score_against_gold(genes, d2_gold)
        chain_results.append(entry)

    # summary: mean chain_precision per gold set across all endorsed chains
    all_labels = [d1_label] + [l for l, _ in gold_cfg["disease2"]]
    summary = {}
    for label in all_labels:
        precisions = [c["scores"][label]["chain_precision"] for c in chain_results]
        summary[label] = {
            "mean_chain_precision": sum(precisions) / len(precisions) if precisions else None,
            "chains_with_any_match": sum(1 for p in precisions if p > 0),
            "total_chains": len(chain_results),
        }

    return {"pair_id": pair_id, "chains": chain_results, "summary": summary}


# Command-line entry point: score one pair or both and save the results.
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", choices=["t2dm_ad", "t2dm_cad", "both"], default="both")
    args = parser.parse_args()
    pair_ids = ["t2dm_ad", "t2dm_cad"] if args.pair == "both" else [args.pair]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for pair_id in pair_ids:
        print(f"\n=== {pair_id} ===")
        result = run_for_pair(pair_id)
        out_path = RESULT_DIR / f"{pair_id}.json"
        out_path.write_text(json.dumps(result, indent=2))
        for label, s in result["summary"].items():
            mp = s["mean_chain_precision"]
            mp_str = f"{mp*100:.1f}%" if mp is not None else "n/a"
            print(f"  vs {label:<24s} mean chain precision: {mp_str}  "
                  f"({s['chains_with_any_match']}/{s['total_chains']} chains had >=1 gold-set gene)")
        for c in result["chains"]:
            genes_str = " -> ".join(c["genes"])
            print(f"    chain {c['chain_index']} ({genes_str}):")
            for label, sc in c["scores"].items():
                if sc["matched_genes"]:
                    print(f"      {label}: {sc['matched_genes']} ({sc['chain_precision']*100:.0f}% of chain)")
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
