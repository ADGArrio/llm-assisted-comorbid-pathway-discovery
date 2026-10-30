"""
Disease pairs and reference mappings shared by both pipelines.

Data: hardcoded config, no external service.
Used by: every stage and pipeline in common/, PipelineA/, PipelineB/.
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


# Stage 3: disease -> GTEx tissueSiteDetailId list (hand-picked, not exhaustive -- see PipelineB/README.md).
DISEASE_RELEVANT_TISSUES = {
    "Type 2 Diabetes Mellitus": ["Pancreas", "Liver", "Muscle_Skeletal", "Adipose_Subcutaneous"],
    "Alzheimer's Disease": ["Brain_Cortex", "Brain_Hippocampus"],
    "Coronary Artery Disease": ["Heart_Left_Ventricle", "Artery_Coronary"],
}

# Stage 4: disease -> ICD code prefixes matched against MIMIC-IV's diagnoses_icd.icd_code, split by
# icd_version ("9" / "10") since MIMIC-IV spans the US's October 2015 ICD-9-to-ICD-10 transition and
# genuinely mixes both coding systems in the same table.
#
# Matching only icd_version=="10" (the original version of this mapping) was checked against the
# real, downloaded MIMIC-IV data on 2026-09-14 and silently drops every ICD-9-coded patient -- a
# ~40-49% undercount per disease here, not a rounding error:
#   T2DM: ICD-9=24305  ICD-10=26344  overlap=6644   UNION=44005  (ICD-10-only missed 17661, 40.1%)
#   AD:   ICD-9=1703   ICD-10=1639   overlap=108    UNION=3234   (ICD-10-only missed 1595, 49.3%)
#   CAD:  ICD-9=24694  ICD-10=27097  overlap=5729   UNION=46062  (ICD-10-only missed 18965, 41.2%)
# The relative-risk numbers computed from the ICD-10-only cohorts (2.30 for T2DM->AD, 3.447 for
# T2DM->CAD, see the Stage 4 real-run write-up) were consequently inflated by sampling bias --
# ICD-10-only means counting only a biased, more-recently-admitted subset of patients. Corrected
# (ICD-9+10 union) relative risks: T2DM->AD ~1.32, T2DM->CAD ~2.05. See
# the Stage 4 real-run write-up for the full writeup.
#
# ICD-9 prefixes below were derived from MIMIC-IV's own d_icd_diagnoses.csv.gz lookup table, not
# guessed from the ICD-10 mapping:
#   - T2DM: the "250" family's last digit encodes diabetes type -- 0/2 = Type II/unspecified,
#     1/3 = Type I (juvenile). Only 0/2 should count, so this is the full set of 20 codes
#     "250" + {0..9} + {0,2}, not a bare "250" prefix (which would incorrectly include Type I).
#   - AD: "3310" is exactly "Alzheimer's disease" in the lookup table; sibling codes (33111,
#     33119, 3312-3317, 3318x, 3319) are Pick's disease, frontotemporal dementia, hydrocephalus,
#     Lewy body dementia, etc. and must be excluded -- so this is an exact code, not a prefix.
#   - CAD: "410"-"414" is the full ICD-9 ischemic-heart-disease family (acute MI, other acute/
#     subacute ischemic forms, old MI, angina pectoris, chronic ischemic heart disease incl.
#     coronary atherosclerosis) -- mirrors the already-corrected ICD-10 I20-I25 range.
_ICD9_T2DM_CODES = [f"250{tens}{last}" for tens in range(10) for last in (0, 2)]

DISEASE_ICD_PREFIXES = {
    "Type 2 Diabetes Mellitus": {"9": _ICD9_T2DM_CODES, "10": ["E11"]},
    "Alzheimer's Disease": {"9": ["3310"], "10": ["G30"]},
    "Coronary Artery Disease": {"9": ["410", "411", "412", "413", "414"], "10": ["I20", "I21", "I22", "I23", "I24", "I25"]},
}
