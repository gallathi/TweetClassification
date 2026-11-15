"""Figures of the report built from the 5-fold results (phase 2) and the weight-space programming experiment.
Built from results/resultats_modelling2.csv, results/resultats_modelling3.csv and wsp/final_results.json.
Run from the project root:  python rapport/make_figures_phase2.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("rapport/figures")
OUT.mkdir(parents=True, exist_ok=True)
RES = Path("results")

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9994"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


m2 = pd.read_csv(RES / "resultats_modelling2.csv", encoding="utf-8-sig").set_index("nom")
m3 = pd.read_csv(RES / "resultats_modelling3.csv", encoding="utf-8-sig").set_index("nom")
res = pd.concat([m2, m3[~m3.index.isin(m2.index)]])

# --- Figure 6: one-factor ablations of XLM-R large (soft labels), OOF and dev -------------------
abl = [("base_soft", "Reference: soft labels, 3 ep., lr 1e-5"), ("prep_v2", "Corrected preprocessing"),
       ("multitask", "Multi-task (Tasks 2 and 3)"), ("cls4", "4 classes (intentions)"),
       ("aeda", "AEDA augmentation"), ("cls7", "7 classes (number of votes)"), ("head_cnn", "MultiCNN head"),
       ("combo_seeds", "Average of 5 seeds (lr 2e-5)")]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
y = np.arange(len(abl))
for ax, col, title, xlim in [(axes[0], "OOF F1@seuil", "Cross-validation (selection)", (0.840, 0.862)),
                             (axes[1], "dev F1@seuil", "Dev set (measurement)", (0.855, 0.895))]:
    ref = res.loc["base_soft", col]
    ax.axvline(ref, color=GREY, lw=1, ls="--")
    for yi, (key, _) in zip(y, abl):
        v = res.loc[key, col]
        color = GREY if key == "base_soft" else (BLUE if v > ref else ORANGE)
        ax.plot(v, yi, "o", color=color, ms=7)
        ax.text(v + (xlim[1] - xlim[0]) * 0.02, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(*xlim)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
fig.supxlabel("F1 of the YES class (threshold chosen on the OOF predictions)", fontsize=9, x=0.62)
axes[0].set_yticks(y, [a[1] for a in abl])
axes[0].invert_yaxis()
axes[0].plot([], [], "o", color=BLUE, label="above the reference")
axes[0].plot([], [], "o", color=ORANGE, label="below the reference")
fig.legend(loc="lower center", bbox_to_anchor=(0.6, 1.0), ncol=2, fontsize=7.5)
fig.tight_layout()
save(fig, "fig_ablations")

# --- Figure 7: all strategies ordered by computing cost, dev F1 with 95% CI ---------------------------
# (label, key in the result tables or a fixed value for runs outside the 5-fold protocol, colour = compute level)
rows = [
    ("TF-IDF + logistic regression (P1)", 0.781, GREY),
    ("Improved TF-IDF, soft labels (P2)", "tfidf", GREY),
    ("Model 1: TF-IDF + features + priors (P2)", "modele1", GREY),
    ("Twitter-XLM-R base, hard + soft (P1)", 0.846, AQUA),
    ("BERTweet-large, translated, EDOS + features (P1)", 0.848, BLUE),
    ("XLM-R large, hard + soft (P2)", "baseline", BLUE),
    ("XLM-R large, soft, 5 seeds (P2)", "xlmr_light", BLUE),
    ("Qwen2.5-7B few-shot, no training (P2)", "llm_few_shot", ORANGE),
    ("Qwen2.5-7B LoRA, 4 bits (P2)", "llm_qwen25_7b_4bit", ORANGE),
    ("Qwen2.5-14B LoRA (P2)", "llm_qwen25_14b", ORANGE),
    ("Qwen3-14B LoRA (P2)", "llm_qwen3_14b", ORANGE),
    ("Ensemble Qwen3-14B + Qwen2.5-7B + XLM-R (P2)", "greedy_tout", ORANGE),
]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for i, (name, key, color) in enumerate(rows):
    if isinstance(key, float):
        v, lo, hi = key, None, None
    else:
        v, lo, hi = res.loc[key, ["dev F1@seuil", "IC95 bas", "IC95 haut"]]
        ax.plot([lo, hi], [i, i], color=color, lw=2, alpha=0.45, solid_capstyle="round")
    ax.plot(v, i, "o" if lo is not None else "D", color=color, ms=7 if lo is not None else 6)
    ax.text((hi if hi is not None else v) + 0.003, i, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.invert_yaxis()
ax.set_xlim(0.75, 0.94)
ax.set_xlabel("F1 of the YES class on the dev set (axis starts at 0.75)")
ax.grid(axis="y", visible=False)
for color, label in [(GREY, "Low compute"), (AQUA, "Medium"), (BLUE, "High"), (ORANGE, "Very high")]:
    ax.plot([], [], "o", color=color, label=label)
ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=4, fontsize=7.5)
save(fig, "fig_overview")

# --- Figure 8: weight of the LLM in the LLM/XLM-R mixture (OOF) ------------------------------------
w = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
f1 = [0.8619, 0.8625, 0.8633, 0.8647, 0.8660, 0.8668, 0.8699, 0.8691, 0.8690, 0.8700, 0.8707, 0.8699, 0.8704,
      0.8708, 0.8706, 0.8708, 0.8709, 0.8715, 0.8708, 0.8699]
fig, ax = plt.subplots(figsize=(4.6, 2.3))
ax.plot(w, f1, "-o", color=BLUE, lw=2, ms=4)
ax.annotate("chosen: w = 0.9\n0.872", (0.9, 0.8715), xytext=(0.55, 0.8625), fontsize=7.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_xlabel("Weight of the LLM ensemble in the mixture")
ax.set_ylabel("OOF F1 (YES)")
save(fig, "fig_llm_weight")

# --- Figure 9: weight-space programming on Qwen3.5-0.8B (test = dev set) ---------------------------
wsp = json.load(open("wsp/final_results.json", encoding="utf-8"))
metrics = [("f1_macro", "Macro-F1"), ("f1_yes", "F1 (YES)"), ("precision_yes", "Precision (YES)"),
           ("auc", "AUC"), ("pct_pred_yes", "Share of 'Yes' answers")]
base = [wsp["base"]["test"]["all"][k] for k, _ in metrics]
edit = [wsp["edite"]["test"]["all"][k] for k, _ in metrics]
fig, ax = plt.subplots(figsize=(7.2, 2.4))
x = np.arange(len(metrics))
for dx, vals, color, label in [(-0.2, base, GREY, "Original model"), (0.2, edit, BLUE, "Edited weights (0.0022%)")]:
    ax.bar(x + dx, vals, 0.38, color=color, label=label)
    for xi, v in zip(x, vals):
        ax.text(xi + dx, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(x, [m[1] for m in metrics])
ax.set_ylim(0, 1.0)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", ncol=2, fontsize=8)
save(fig, "fig_wsp")
print("figures written to", OUT)
