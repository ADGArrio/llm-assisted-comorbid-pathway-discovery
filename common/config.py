"""Disease pairs and the reference mappings (GTEx tissues, MIMIC-IV ICD codes) shared by both pipelines."""
from dotenv import load_dotenv

load_dotenv()  # Load .env into os.environ once, at first import; every client reads its settings from there.

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


# Look up a disease pair by id. Input: pair id ("t2dm_ad" or "t2dm_cad"). Output: the matching dict from DISEASE_PAIRS;
# raises KeyError if unknown.
def get_pair(pair_id: str) -> dict:
    for pair in DISEASE_PAIRS:
        if pair["id"] == pair_id:
            return pair
    raise KeyError(f"Unknown disease pair id: {pair_id!r}. Known ids: {[p['id'] for p in DISEASE_PAIRS]}")


# Stage 3: disease -> GTEx tissueSiteDetailId list (hand-picked, not exhaustive).
DISEASE_RELEVANT_TISSUES = {
    "Type 2 Diabetes Mellitus": ["Pancreas", "Liver", "Muscle_Skeletal", "Adipose_Subcutaneous"],
    "Alzheimer's Disease": ["Brain_Cortex", "Brain_Hippocampus"],
    "Coronary Artery Disease": ["Heart_Left_Ventricle", "Artery_Coronary"],
}

# Stage 4: disease -> ICD code prefixes matched against MIMIC-IV diagnoses_icd.icd_code, split by icd_version ("9" /
# "10") because MIMIC-IV mixes both coding systems.
# ICD-9 prefixes were derived from MIMIC-IV's own d_icd_diagnoses lookup table.
# T2DM: the "250" family's last digit encodes diabetes type; only 0 and 2 (type II / unspecified) count, so the full set
# of 20 codes is listed rather than a bare "250" prefix.
# AD: "3310" is exactly Alzheimer's disease, so it is an exact code, not a prefix.
# CAD: "410"-"414" is the ischemic-heart-disease family, mirroring the ICD-10 I20-I25 range.
_ICD9_T2DM_CODES = [f"250{tens}{last}" for tens in range(10) for last in (0, 2)]

DISEASE_ICD_PREFIXES = {
    "Type 2 Diabetes Mellitus": {"9": _ICD9_T2DM_CODES, "10": ["E11"]},
    "Alzheimer's Disease": {"9": ["3310"], "10": ["G30"]},
    "Coronary Artery Disease": {"9": ["410", "411", "412", "413", "414"], "10": ["I20", "I21", "I22", "I23", "I24", "I25"]},
}
