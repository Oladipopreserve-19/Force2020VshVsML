"""Classical shale-volume estimators and sand/shale classification by cutoff.

Endpoints are picked per well from the logs alone (P5 and P95 of GR), never from
the lithology labels, so the test is not contaminated by the answer key.
Run: .venv/bin/python -I code/02_classical.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_pickle(ROOT / "data/force_all_wells.pkl")

LITH = {30000: "Sandstone", 65030: "Sand/Shale", 65000: "Shale"}
d["LITH"] = d.FORCE_2020_LITHOFACIES_LITHOLOGY.map(LITH)
d = d[d.LITH.notna() & d.GR.notna()].copy()

# Gamma-ray index with per-well P5/P95 endpoints
g = d.groupby("WELL").GR
gmin, gmax = g.transform(lambda s: s.quantile(0.05)), g.transform(lambda s: s.quantile(0.95))
igr = ((d.GR - gmin) / (gmax - gmin)).clip(0, 1)
d["VSH_LINEAR"] = igr
d["VSH_LARIONOV_TERT"] = (0.083 * (2 ** (3.7 * igr) - 1)).clip(0, 1)
d["VSH_LARIONOV_OLD"] = (0.33 * (2 ** (2 * igr) - 1)).clip(0, 1)
d["VSH_CLAVIER"] = (1.7 - np.sqrt(np.clip(3.38 - (igr + 0.7) ** 2, 0, None))).clip(0, 1)

# Neutron-density separation, shale point = per-well P95 of separation (logs only)
nphi = np.where(d.NPHI > 1.5, d.NPHI / 100, d.NPHI)
phid = (2.65 - d.RHOB) / (2.65 - 1.0)
sep = pd.Series(nphi - phid, index=d.index)
sep_sh = sep.groupby(d.WELL).transform(lambda s: s.quantile(0.95))
sep_cl = sep.groupby(d.WELL).transform(lambda s: s.quantile(0.05))
d["VSH_ND"] = ((sep - sep_cl) / (sep_sh - sep_cl)).clip(0, 1)

METHODS = ["VSH_LINEAR", "VSH_LARIONOV_TERT", "VSH_LARIONOV_OLD", "VSH_CLAVIER", "VSH_ND"]
CUTOFFS = [0.35, 0.40, 0.50, 0.60]

binary = d[d.LITH.isin(["Sandstone", "Shale"])]
rows = []
for m in METHODS:
    sub = binary[binary[m].notna()]
    y = (sub.LITH == "Shale").astype(int)
    for c in CUTOFFS:
        p = (sub[m] > c).astype(int)
        rows.append(dict(method=m, cutoff=c, n=len(sub), wells=sub.WELL.nunique(),
                         accuracy=accuracy_score(y, p), balanced_acc=balanced_accuracy_score(y, p),
                         f1_shale=f1_score(y, p), f1_sand=f1_score(1 - y, 1 - p)))
res = pd.DataFrame(rows)
res.to_csv(ROOT / "data/results_classical_cutoffs.csv", index=False)

# Mixed class: where do Sand/Shale samples fall?
mixed = d[d.LITH == "Sand/Shale"][METHODS].median().rename("median_vsh_mixed_class")
mixed.to_csv(ROOT / "data/results_mixed_class.csv")

d[["WELL", "DEPTH_MD", "LITH"] + METHODS].to_pickle(ROOT / "data/vsh_classical.pkl")
pd.set_option("display.width", 160)
print(res.round(3).to_string(index=False))
print(mixed.round(3))
