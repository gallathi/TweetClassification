"""Figures of the report (English labels), built from the dataset and the results of Modelling.ipynb.
Run from the project root:  python rapport/make_figures.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("rapport/figures")
OUT.mkdir(parents=True, exist_ok=True)

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


# --- Figure 1: votes and labels (training set) -------------------------------------------
d = pd.DataFrame.from_dict(json.load(open("data/EXIST2024_training.json", encoding="utf-8")), orient="index")
d["n_yes"] = d["labels_task1"].apply(lambda l: sum(x == "YES" for x in l))
votes = d["n_yes"].value_counts().sort_index()
hard = d[d["n_yes"] != 3].assign(label=lambda x: np.where(x["n_yes"] > 3, "YES", "NO"))
by_lang = hard.groupby(["lang", "label"]).size().unstack()

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
colors = [BLUE] * 3 + [GREY] + [ORANGE] * 3
axes[0].bar(votes.index, votes.values, color=colors, width=0.8)
for x, v in zip(votes.index, votes.values):
    axes[0].text(x, v + 20, f"{v}", ha="center", va="bottom", fontsize=7.5, color=INK2)
axes[0].set_xticks(range(7))
axes[0].set_xlabel("Number of annotators (out of 6) who answered YES")
axes[0].set_ylabel("Tweets")
axes[0].set_ylim(0, votes.max() * 1.15)
axes[0].set_title("Annotator agreement (NO / tie / YES)", loc="left")

langs = ["en", "es"]
x = np.arange(2)
axes[1].bar(x - 0.2, by_lang.loc[langs, "NO"], 0.38, color=BLUE, label="NO")
axes[1].bar(x + 0.2, by_lang.loc[langs, "YES"], 0.38, color=ORANGE, label="YES")
for xi, lang in zip(x, langs):
    for dx, lab in [(-0.2, "NO"), (0.2, "YES")]:
        v = by_lang.loc[lang, lab]
        axes[1].text(xi + dx, v + 20, f"{v}", ha="center", va="bottom", fontsize=7.5, color=INK2)
axes[1].set_xticks(x, ["English", "Spanish"])
axes[1].set_ylim(0, by_lang.values.max() * 1.3)
axes[1].set_title("Hard labels per language", loc="left")
axes[1].legend(loc="upper center", ncol=2, fontsize=8)
fig.tight_layout()
save(fig, "fig_data")

# --- Figure 2: TF-IDF models (dev F1 of the YES class) ------------------------------------
models = ["Logistic regression", "Linear SVM", "Ridge", "Naive Bayes", "XGBoost", "Random forest"]
f1_wc = [0.781, 0.775, 0.771, 0.703, None, None]
f1_w = [0.756, 0.762, 0.757, 0.737, 0.726, 0.694]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
y = np.arange(len(models))
for vals, dy, color, label in [(f1_wc, -0.2, BLUE, "Words + characters"), (f1_w, 0.2, ORANGE, "Words")]:
    for yi, v in zip(y, vals):
        if v is None:
            continue
        ax.barh(yi + dy, v - 0.6, left=0.6, height=0.38, color=color, label=label if yi == 0 else None)
        ax.text(v + 0.002, yi + dy, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(y, models)
ax.invert_yaxis()
ax.set_xlim(0.6, 0.81)
ax.set_xlabel("F1 of the YES class on the dev set (axis starts at 0.60)")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="y", visible=False)
save(fig, "fig_tfidf")

# --- Figure 3: TF-IDF error rate vs annotator agreement -----------------------------------
n_yes = [0, 1, 2, 4, 5, 6]
err = [0.113, 0.206, 0.250, 0.389, 0.196, 0.147]
size = [203, 160, 116, 144, 148, 163]
fig, ax = plt.subplots(figsize=(4.6, 2.4))
ax.bar([str(n) for n in n_yes], err, color=BLUE, width=0.7)
for i, (e, n) in enumerate(zip(err, size)):
    ax.text(i, e + 0.008, f"{e:.0%}\n(n={n})", ha="center", va="bottom", fontsize=7, color=INK2)
ax.set_ylim(0, 0.5)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_xlabel("Number of YES votes out of 6 (3 = tie, not evaluated)")
ax.set_ylabel("Error rate")
ax.grid(axis="x", visible=False)
save(fig, "fig_errors")

# --- Figure 4: RoBERTa training curves -------------------------------------------------------
epochs = [1, 2, 3]
curves = {"Hard labels": ([0.510, 0.317, 0.209], [0.735, 0.756, 0.959], BLUE),
          "Soft labels": ([0.594, 0.514, 0.481], [0.556, 0.558, 0.558], ORANGE)}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
for name, (train, val, color) in curves.items():
    axes[0].plot(epochs, train, "-o", color=color, lw=2, ms=6, label=name)
    axes[1].plot(epochs, val, "-o", color=color, lw=2, ms=6, label=name)
axes[0].set_title("Training loss", loc="left")
axes[1].set_title("Validation cross-entropy (lower is better)", loc="left")
for ax in axes:
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch")
axes[1].legend(loc="upper left", fontsize=8)
fig.tight_layout()
save(fig, "fig_roberta_curves")

# --- Figure 5: summary of all configurations ------------------------------------------------------
summary = [
    ("TF-IDF + logistic regression", 0.781, GREY),
    ("Twitter-XLM-R, soft labels", 0.834, BLUE),
    ("Twitter-XLM-R, hard labels", 0.843, BLUE),
    ("Twitter-XLM-R, average hard + soft", 0.846, BLUE),
    ("BERTweet-large + linear (translated)", 0.833, ORANGE),
    ("EDOS detector + features, 4 epochs", 0.839, ORANGE),
    ("BERTweet-large + MultiCNN + features", 0.846, ORANGE),
    ("EDOS detector + linear", 0.847, ORANGE),
    ("EDOS detector + MultiCNN + features", 0.848, ORANGE),
]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for i, (name, f1, color) in enumerate(summary):
    ax.plot(f1, i, "o", color=color, ms=7)
    ax.text(f1 + 0.003, i, f"{f1:.3f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(summary)), [s[0] for s in summary])
ax.invert_yaxis()
ax.set_xlim(0.77, 0.86)
ax.set_xlabel("F1 of the YES class on the dev set (axis starts at 0.77)")
ax.grid(axis="y", visible=False)
ax.plot([], [], "o", color=BLUE, label="Main pipeline (original tweets)")
ax.plot([], [], "o", color=ORANGE, label="Second pipeline (Spanish translated to English)")
ax.plot([], [], "o", color=GREY, label="Baseline")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
save(fig, "fig_summary")
print("figures written to", OUT)
