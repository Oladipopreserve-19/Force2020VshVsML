"""Tree-based classifiers vs classical Vsh, sand/shale, tested on held-out wells.

5-fold GroupKFold by well: no well is ever in both training and test.
Run: .venv/bin/python -I code/03_ml.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_pickle(ROOT / "data/force_all_wells.pkl")
vsh = pd.read_pickle(ROOT / "data/vsh_classical.pkl")
d = raw.loc[vsh.index].join(vsh[["LITH", "VSH_LINEAR", "VSH_ND"]])
d = d[d.LITH.isin(["Sandstone", "Shale"])].copy()
d["y"] = (d.LITH == "Shale").astype(int)
d["LOG_RDEP"] = np.log10(d.RDEP.clip(lower=0.01))
d["NPHI"] = np.where(d.NPHI > 1.5, d.NPHI / 100, d.NPHI)

FEATS_ALL = ["GR", "RHOB", "NPHI", "DTC", "PEF", "CALI", "LOG_RDEP", "DEPTH_MD"]
FEATS_GR = ["GR", "DEPTH_MD"]

gkf = GroupKFold(n_splits=5)
pred = pd.DataFrame(index=d.index)
rng = np.random.default_rng(42)
for k, (tr, te) in enumerate(gkf.split(d, d.y, d.WELL)):
    trd, ted = d.iloc[tr], d.iloc[te]
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=42)
    hgb.fit(trd[FEATS_ALL], trd.y)
    pred.loc[ted.index, "HGB_ALL"] = hgb.predict(ted[FEATS_ALL])
    hgb_gr = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=42)
    hgb_gr.fit(trd[FEATS_GR], trd.y)
    pred.loc[ted.index, "HGB_GR"] = hgb_gr.predict(ted[FEATS_GR])
    # Random forest needs complete rows; train on a 200k subsample for speed
    cc = trd.dropna(subset=FEATS_ALL)
    cc = cc.iloc[rng.choice(len(cc), min(200_000, len(cc)), replace=False)]
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=42)
    rf.fit(cc[FEATS_ALL], cc.y)
    tc = ted.dropna(subset=FEATS_ALL)
    pred.loc[tc.index, "RF_ALL"] = rf.predict(tc[FEATS_ALL])
    print(f"fold {k+1}/5 done ({ted.WELL.nunique()} test wells)", flush=True)

pred["LINEAR_0.50"] = (d.VSH_LINEAR > 0.5).astype(int)
pred["LINEAR_0.35"] = (d.VSH_LINEAR > 0.35).astype(int)
pred["ND_0.50"] = (d.VSH_ND > 0.5).where(d.VSH_ND.notna())
pred["ND_0.40"] = (d.VSH_ND > 0.4).where(d.VSH_ND.notna())

def score(mask_name, mask):
    out = []
    for m in pred.columns:
        ok = mask & pred[m].notna()
        y, p = d.y[ok], pred[m][ok].astype(int)
        out.append(dict(subset=mask_name, method=m, n=ok.sum(), accuracy=accuracy_score(y, p),
                        balanced_acc=balanced_accuracy_score(y, p), f1_shale=f1_score(y, p), f1_sand=f1_score(1-y, 1-p)))
    return out

full = d[FEATS_ALL].notna().all(axis=1)
res = pd.DataFrame(score("all samples", pd.Series(True, index=d.index)) + score("complete logs only", full))
res.to_csv(ROOT / "data/results_ml_vs_classical.csv", index=False)

# Per-well balanced accuracy, to show spread across wells
pw = []
for w, g in d.groupby("WELL"):
    if g.y.nunique() < 2: continue
    row = {"WELL": w, "n": len(g), "shale_frac": g.y.mean()}
    for m in ["HGB_ALL", "LINEAR_0.50", "LINEAR_0.35", "ND_0.50"]:
        ok = pred.loc[g.index, m].notna()
        if ok.sum() > 100:
            row[m] = balanced_accuracy_score(g.y[ok], pred.loc[g.index, m][ok].astype(int))
    pw.append(row)
pw = pd.DataFrame(pw); pw.to_csv(ROOT / "data/results_per_well.csv", index=False)
pred.join(d[["WELL", "DEPTH_MD", "y"]]).to_pickle(ROOT / "data/predictions.pkl")

pd.set_option("display.width", 160)
print(res.round(3).to_string(index=False))
print("\nPer-well balanced accuracy (median, IQR):")
for m in ["HGB_ALL", "LINEAR_0.50", "LINEAR_0.35", "ND_0.50"]:
    s = pw[m].dropna(); print(f"  {m:12s} n={len(s):3d} median={s.median():.3f} IQR={s.quantile(.25):.3f}-{s.quantile(.75):.3f}")
print("Wells where HGB_ALL beats LINEAR_0.35:", int((pw.HGB_ALL > pw["LINEAR_0.35"]).sum()), "of", int(pw[["HGB_ALL","LINEAR_0.35"]].notna().all(axis=1).sum()))
