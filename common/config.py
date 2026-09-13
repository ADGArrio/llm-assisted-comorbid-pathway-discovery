"""
Disease pairs and reference mappings shared by both pipelines.

Data: hardcoded config, no external service.
Used by: every agent/pipeline in common/, PipelineA/, PipelineB/.
"""
from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ once, at first import -- every client below reads its config via os.environ.get(...)

DISEASE_PAIRS = [
    {
        "id": "t2dm_ad",
        "disease1": "Type 2 Diabetes Mellitus",
        "disease2": "Alzheimer's Disease",
    },
    {
        "id": "t2dm_cad",
        "disease1": "Type 2 Diabetes Mellitus",
        "disease2": "Coronary Artery Disease",
    },
]


def get_pair(pair_id: str) -> dict:
    # Input: pair id string ("t2dm_ad" / "t2dm_cad"). Output: the matching dict from DISEASE_PAIRS, or raises KeyError.
    for pair in DISEASE_PAIRS:
        if pair["id"] == pair_id:
            return pair
    raise KeyError(f"Unknown disease pair id: {pair_id!r}. Known ids: {[p['id'] for p in DISEASE_PAIRS]}")


# Agent 3: disease -> GTEx tissueSiteDetailId list (hand-picked, not exhaustive -- see PipelineB/README.md).
DISEASE_RELEVANT_TISSUES = {
    "Type 2 Diabetes Mellitus": ["Pancreas", "Liver", "Muscle_Skeletal", "Adipose_Subcutaneous"],
    "Alzheimer's Disease": ["Brain_Cortex", "Brain_Hippocampus"],
    "Coronary Artery Disease": ["Heart_Left_Ventricle", "Artery_Coronary"],
}

# Agent 4: disease -> ICD-10 code prefixes matched against MIMIC-IV's diagnoses_icd.icd_code (icd_version==10 rows only).
# CAD uses the full ischemic-heart-disease range I20-I25 (angina, acute/subsequent MI, other acute
# ischemic disease, chronic ischemic heart disease) -- I25 alone was checked against the real MIMIC-IV
# data and undercounts by ~2,755 patients (~10%), mostly from I21 (acute MI). See
# claude/agent4_mimic_real_run.md for the verification.
DISEASE_ICD10_PREFIXES = {
    "Type 2 Diabetes Mellitus": ["E11"],
    "Alzheimer's Disease": ["G30"],
    "Coronary Artery Disease": ["I20", "I21", "I22", "I23", "I24", "I25"],
}
