"""Generate the thesis figures from real saved pipeline outputs.

Reads: outputs/pipeline_b_t2dm_ad{FIG_RUN_SUFFIX}.json, outputs/pipeline_b_t2dm_cad{FIG_RUN_SUFFIX}.json,
       outputs/model_comparison/*.json, outputs/edge_literature_recall/t2dm_cad.json
Writes: docs/figures/*.png (also copied to ../Thesis/figures/)

Every plotted value is read from the saved JSON files; a missing file fails loudly instead of being replaced by a placeholder.
"""
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "docs", "figures")
THESIS_FIG_DIR = os.path.join(HERE, "..", "Thesis", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Palette (fixed categorical order).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GOOD = "#0ca30c"
MUTED_INK = "#52514e"
AXIS = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": AXIS,
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# Input: a path relative to this script's folder. Output: the parsed JSON.
def load(path):
    with open(os.path.join(HERE, path)) as f:
        return json.load(f)


# Input: a chain dict. Output: its genes joined by " -> ".
def gene_chain_str(chain):
    edges = chain["edges"]
    genes = [edges[0]["source_gene"]] + [e["target_gene"] for e in edges]
    return " -> ".join(genes)


# Figure 1: multi-model robustness (endorsement rate and winning Elo).
MODEL_COLORS = {"gemma4:12b": BLUE, "qwen3:14b": ORANGE, "deepseek-r1:14b": AQUA}
MODEL_ORDER = ["gemma4:12b", "qwen3:14b", "deepseek-r1:14b"]
MODEL_LABELS = {"gemma4:12b": "gemma4:12b\n(default)", "qwen3:14b": "qwen3:14b", "deepseek-r1:14b": "deepseek-r1:14b"}

pairs = {
    "T2DM -> AD": load("outputs/model_comparison/t2dm_ad__summary.json"),
    "T2DM -> CAD": load("outputs/model_comparison/t2dm_cad__summary.json"),
}
per_model_files = {
    ("T2DM -> AD", "gemma4:12b"): "outputs/model_comparison/t2dm_ad__gemma4_12b.json",
    ("T2DM -> AD", "qwen3:14b"): "outputs/model_comparison/t2dm_ad__qwen3_14b.json",
    ("T2DM -> AD", "deepseek-r1:14b"): "outputs/model_comparison/t2dm_ad__deepseek-r1_14b.json",
    ("T2DM -> CAD", "gemma4:12b"): "outputs/model_comparison/t2dm_cad__gemma4_12b.json",
    ("T2DM -> CAD", "qwen3:14b"): "outputs/model_comparison/t2dm_cad__qwen3_14b.json",
    ("T2DM -> CAD", "deepseek-r1:14b"): "outputs/model_comparison/t2dm_cad__deepseek-r1_14b.json",
}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

# Panel A: endorsement rate
ax = axes[0]
pair_names = list(pairs.keys())
x = range(len(pair_names))
width = 0.25
for i, model in enumerate(MODEL_ORDER):
    vals = [pairs[p]["per_model"][model]["endorsed"] for p in pair_names]
    offs = [xi + (i - 1) * width for xi in x]
    bars = ax.bar(offs, vals, width=width, color=MODEL_COLORS[model], label=MODEL_LABELS[model].split("\n")[0])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v), ha="center", va="bottom", fontsize=9, color="#0b0b0b")
ax.set_xticks(list(x))
ax.set_xticklabels(pair_names)
ax.set_ylabel("Candidates endorsed (of 10)")
ax.set_title("A. Judge endorsement rate by model")
ax.set_ylim(0, 6)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y")
ax.grid(axis="x", visible=False)

# Panel B: winning chain Elo (where a model endorsed anything)
ax2 = axes[1]
elo_vals = []
elo_colors = []
elo_labels = []
for p in pair_names:
    for model in MODEL_ORDER:
        d = load(per_model_files[(p, model)])
        chains = d.get("stage2_chains") or []
        if not chains:
            continue
        top = max(chains, key=lambda c: c.get("elo_rating") or 0)
        elo_vals.append(top["elo_rating"])
        elo_colors.append(MODEL_COLORS[model])
        elo_labels.append(f"{p}\n{model.split(':')[0]}")
xpos = range(len(elo_vals))
bars = ax2.bar(xpos, elo_vals, color=elo_colors, width=0.6)
for b, v in zip(bars, elo_vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color="#0b0b0b")
ax2.set_xticks(list(xpos))
ax2.set_xticklabels(elo_labels, fontsize=8)
ax2.set_ylabel("Elo rating of winning chain")
ax2.set_title("B. Winning-chain Elo (models that endorsed ≥ 1)")
ax2.set_ylim(1100, 1300)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="y")
ax2.grid(axis="x", visible=False)

handles, labels = ax.get_legend_handles_labels()
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=9)
fig.savefig(os.path.join(OUT_DIR, "fig_multimodel_robustness.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote fig_multimodel_robustness.png")

# Figure 2: GTEx tissue-expression confirmation rate per endorsed chain.
# Which saved run to plot: FIG_RUN_SUFFIX="_nometab" reads outputs/pipeline_b_<pair>_nometab.json (the run used for the
# reported results); empty reads the latest main run.
RUN_SUFFIX = os.environ.get("FIG_RUN_SUFFIX", "")
ad = load(f"outputs/pipeline_b_t2dm_ad{RUN_SUFFIX}.json")
cad = load(f"outputs/pipeline_b_t2dm_cad{RUN_SUFFIX}.json")

rows = []
for label_prefix, data in [("T2DM→AD", ad), ("T2DM→CAD", cad)]:
    for i, chain in enumerate(data["stage2"]["chains"]):
        a3 = data["stage3"][i]
        expressed = a3["tissue_expressed_count"]
        total = a3["tissue_checked_count"]
        rows.append((f"{label_prefix}\nP{i+1}", expressed, total, gene_chain_str(chain)))

fig, ax = plt.subplots(figsize=(8.5, 4.2))
xlabels = [r[0] for r in rows]
fracs = [r[1] / r[2] for r in rows]
bars = ax.bar(range(len(rows)), fracs, color=BLUE, width=0.55)
for b, r in zip(bars, rows):
    ax.text(b.get_x() + b.get_width() / 2, r[1] / r[2] + 0.02, f"{r[1]}/{r[2]}", ha="center", va="bottom", fontsize=9, color="#0b0b0b")
ax.set_xticks(range(len(rows)))
ax.set_xticklabels(xlabels, fontsize=8.5)
ax.set_ylabel("Fraction of gene/tissue checks\nconfirmed expressed (GTEx)")
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title("GTEx tissue-expression confirmation, all endorsed chains (both disease pairs)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_gtex_coverage.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote fig_gtex_coverage.png")

# Figure 3: MIMIC-IV cohort rate vs. general-population base rate.
ad4 = ad["stage4"]
cad4 = cad["stage4"]
groups = ["T2DM → AD", "T2DM → CAD"]
from PipelineB.stage4_cohort_check import _two_by_two_stats  # recompute from counts so old saved results work too


# Input: a Stage 4 result dict. Output: the disease-2 rate among patients without disease 1.
def _unexposed_rate(a4):
    return (a4["n_disease2_patients"] - a4["n_both_patients"]) / (a4["n_total_patients"] - a4["n_disease1_patients"])


base_rates = [_unexposed_rate(ad4), _unexposed_rate(cad4)]  # rate among patients WITHOUT T2DM
cohort_rates = [ad4["disease2_rate_in_disease1_cohort"], cad4["disease2_rate_in_disease1_cohort"]]
rrs = [
    _two_by_two_stats(a["n_total_patients"], a["n_disease1_patients"], a["n_disease2_patients"], a["n_both_patients"])["relative_risk"]
    for a in (ad4, cad4)
]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = range(len(groups))
width = 0.32
b1 = ax.bar([xi - width / 2 for xi in x], base_rates, width=width, color=BLUE, label="Rate among patients without T2DM")
b2 = ax.bar([xi + width / 2 for xi in x], cohort_rates, width=width, color=ORANGE, label="Rate among T2DM patients")
for bars, vals in [(b1, base_rates), (b2, cohort_rates)]:
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9, color="#0b0b0b")
for xi, rr in zip(x, rrs):
    ax.text(xi, max(base_rates[xi], cohort_rates[xi]) + 0.05, f"RR={rr:.2f}", ha="center", va="bottom", fontsize=9.5, color=MUTED_INK, style="italic")
ax.set_xticks(list(x))
ax.set_xticklabels(groups)
ax.set_ylabel("Rate in MIMIC-IV cohort (n=223,291)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title("MIMIC-IV real-world comorbidity rates")
ax.set_ylim(0, max(cohort_rates) * 1.35)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_mimic_relative_risk.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote fig_mimic_relative_risk.png")

# copy to Thesis/figures as well
os.makedirs(THESIS_FIG_DIR, exist_ok=True)
for fn in ["fig_multimodel_robustness.png", "fig_gtex_coverage.png", "fig_mimic_relative_risk.png"]:
    shutil.copy(os.path.join(OUT_DIR, fn), os.path.join(THESIS_FIG_DIR, fn))
print("copied figures to Thesis/figures/")

# Figure 4: gold-standard structural coverage (real KEGG gene sets, per endorsed chain); see
# scripts/gold_standard_recall.py.
gs_ad = load("outputs/gold_standard_recall/t2dm_ad.json")
gs_cad = load("outputs/gold_standard_recall/t2dm_cad.json")

labels = []
t2dm_vals = []
d2_vals = []
d2_set_label = []
for prefix, gs, d2_key in [("T2DM→AD", gs_ad, "AD"), ("T2DM→CAD", gs_cad, "CAD_bridge_AGE_RAGE")]:
    for c in gs["chains"]:
        labels.append(f"{prefix}\nP{c['chain_index']+1}")
        t2dm_vals.append(c["scores"]["T2DM"]["chain_precision"])
        d2_vals.append(c["scores"][d2_key]["chain_precision"])
        d2_set_label.append("AD" if d2_key == "AD" else "CAD (AGE-RAGE\nbridge pathway)")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
x = range(len(labels))
ax = axes[0]
bars = ax.bar(x, t2dm_vals, color=BLUE, width=0.55)
for b, v in zip(bars, t2dm_vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0, 0.65)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_ylabel("Fraction of chain's genes\nin KEGG hsa04930 (T2DM)")
ax.set_title("A. T2DM-side coverage")
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y"); ax.grid(axis="x", visible=False)

ax2 = axes[1]
bars2 = ax2.bar(x, d2_vals, color=ORANGE, width=0.55)
for b, v in zip(bars2, d2_vals):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=9)
ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylim(0, 0.65)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax2.set_ylabel("Fraction of chain's genes in the\ndisease-2 KEGG gold set")
ax2.set_title("B. Disease-2-side coverage (AD pathway / CAD bridge pathway)")
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="y"); ax2.grid(axis="x", visible=False)

fig.suptitle("Gold-standard structural coverage vs. real KEGG gene sets (metric 8)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_gold_standard_coverage.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote fig_gold_standard_coverage.png")

for fn in ["fig_gold_standard_coverage.png"]:
    shutil.copy(os.path.join(OUT_DIR, fn), os.path.join(THESIS_FIG_DIR, fn))
print("copied gold-standard figure to Thesis/figures/")
