"""Figures and tables for Preprint 5. Run: .venv/bin/python -I code/04_figures.py"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D, F = ROOT / "data", ROOT / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
C = {"VSH_LINEAR": "#0072B2", "VSH_LARIONOV_TERT": "#E69F00", "VSH_LARIONOV_OLD": "#009E73",
     "VSH_CLAVIER": "#CC79A7", "VSH_ND": "#D55E00"}
NAMES = {"VSH_LINEAR": "Linear GR", "VSH_LARIONOV_TERT": "Larionov (Tertiary)", "VSH_LARIONOV_OLD": "Larionov (older rocks)",
         "VSH_CLAVIER": "Clavier", "VSH_ND": "Neutron-density"}

# Fig 1: cutoff sensitivity
cl = pd.read_csv(D / "results_classical_cutoffs.csv")
fig, ax = plt.subplots(figsize=(4.5, 3.2))
for m, g in cl.groupby("method", sort=False):
    ax.plot(g.cutoff, g.balanced_acc, "o-", color=C[m], label=NAMES[m])
ax.axvline(0.5, color="grey", ls=":", lw=1)
ax.set(xlabel="Vsh cutoff (shale if Vsh > cutoff)", ylabel="Balanced accuracy", xticks=[0.35, 0.4, 0.5, 0.6])
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(F / "fig1_cutoff_sensitivity.png"); plt.close(fig)

# Fig 2: method comparison
ml = pd.read_csv(D / "results_ml_vs_classical.csv")
LBL = {"HGB_ALL": "Gradient boosting, all logs", "RF_ALL": "Random forest, all logs", "HGB_GR": "Gradient boosting, GR + depth",
       "LINEAR_0.50": "Linear GR, 0.50", "LINEAR_0.35": "Linear GR, 0.35", "ND_0.50": "Neutron-density, 0.50", "ND_0.40": "Neutron-density, 0.40"}
sub = ml[ml.subset == "complete logs only"].set_index("method").loc[list(LBL)].sort_values("balanced_acc")
fig, ax = plt.subplots(figsize=(5, 3.2))
cols = ["#D55E00" if m.startswith(("HGB", "RF")) else "#0072B2" for m in sub.index]
ax.barh([LBL[m] for m in sub.index], sub.balanced_acc, color=cols)
for i, v in enumerate(sub.balanced_acc): ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)
ax.set(xlim=(0.5, 1.0), xlabel="Balanced accuracy, held-out wells (samples with all logs, n = %d)" % sub.n.iloc[0])
fig.tight_layout(); fig.savefig(F / "fig2_method_comparison.png"); plt.close(fig)

# Fig 3: per-well comparison
pw = pd.read_csv(D / "results_per_well.csv").dropna(subset=["HGB_ALL", "LINEAR_0.35"])
fig, ax = plt.subplots(figsize=(3.8, 3.6))
win = pw.HGB_ALL > pw["LINEAR_0.35"]
ax.scatter(pw["LINEAR_0.35"][win], pw.HGB_ALL[win], s=14, color="#D55E00", label=f"ML better ({win.sum()} wells)")
ax.scatter(pw["LINEAR_0.35"][~win], pw.HGB_ALL[~win], s=14, color="#0072B2", label=f"Cutoff as good or better ({(~win).sum()})")
ax.plot([0.3, 1], [0.3, 1], color="grey", lw=1, ls=":")
ax.set(xlabel="Linear GR, cutoff 0.35", ylabel="Gradient boosting, all logs", xlim=(0.3, 1), ylim=(0.3, 1), title="Balanced accuracy per well")
ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(F / "fig3_per_well.png"); plt.close(fig)

# Fig 4: Vsh distribution by labelled class
v = pd.read_pickle(D / "vsh_classical.pkl")
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
for ax, m in zip(axes, ["VSH_LINEAR", "VSH_ND"]):
    for lith, col in [("Sandstone", "#E69F00"), ("Sand/Shale", "#999999"), ("Shale", "#0072B2")]:
        x = v.loc[v.LITH == lith, m].dropna()
        ax.hist(x, bins=50, range=(0, 1), density=True, histtype="step", lw=1.5, color=col, label=lith)
    ax.axvline(0.5, color="k", ls=":", lw=1); ax.set(xlabel=NAMES[m] + " Vsh")
axes[0].set_ylabel("Density"); axes[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(F / "fig4_vsh_by_class.png"); plt.close(fig)

# Fig 5: example well, labels vs predictions (well closest to median HGB score)
pr = pd.read_pickle(D / "predictions.pkl")
w = pw.iloc[(pw.HGB_ALL - pw.HGB_ALL.median()).abs().argsort().iloc[0]].WELL
g = pr[pr.WELL == w].sort_values("DEPTH_MD").join(v[["VSH_LINEAR"]])
raw = pd.read_pickle(D / "force_all_wells.pkl").loc[g.index]
fig, axes = plt.subplots(1, 5, figsize=(6.5, 6), sharey=True, gridspec_kw={"width_ratios": [2, 2, 0.6, 0.6, 0.6]})
axes[0].plot(raw.GR, g.DEPTH_MD, lw=0.5, color="#009E73"); axes[0].set_xlabel("GR (API)")
axes[1].plot(g.VSH_LINEAR, g.DEPTH_MD, lw=0.5, color="#0072B2"); axes[1].axvline(0.35, ls=":", color="k"); axes[1].set_xlabel("Linear Vsh")
cmap = matplotlib.colors.ListedColormap(["#E69F00", "#0072B2"])
for ax, col, t in zip(axes[2:], ["y", "LINEAR_0.35", "HGB_ALL"], ["Label", "GR 0.35", "ML"]):
    ax.imshow(g[col].values[:, None].astype(float), aspect="auto", cmap=cmap, vmin=0, vmax=1,
              extent=[0, 1, g.DEPTH_MD.max(), g.DEPTH_MD.min()], interpolation="nearest")
    ax.set_xticks([]); ax.set_xlabel(t)
axes[0].set_ylabel("Measured depth (m)"); axes[0].invert_yaxis() if not axes[0].yaxis_inverted() else None
fig.suptitle(f"Well {w}: orange = sandstone, blue = shale", fontsize=9)
fig.tight_layout(); fig.savefig(F / "fig5_example_well.png"); plt.close(fig)

# Tables (markdown, for the draft)
t1 = cl.pivot(index="method", columns="cutoff", values="balanced_acc").rename(index=NAMES).round(3)
t2 = ml.pivot(index="method", columns="subset", values="balanced_acc").rename(index=LBL).round(3)
(ROOT / "draft/tables.md").write_text("Table 1. Balanced accuracy by cutoff\n\n" + t1.to_markdown() +
                                      "\n\nTable 2. Balanced accuracy, held-out wells\n\n" + t2.to_markdown() + "\n")
print("example well:", w); print(t1); print(t2)
