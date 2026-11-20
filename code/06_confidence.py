"""Repeat the evaluation on high-confidence labels only (FORCE confidence = 1).
Predictions are the held-out predictions from 03_ml.py; only the scoring subset changes.
Run: .venv/bin/python -I code/06_confidence.py
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
pr = pd.read_pickle(D / "predictions.pkl")
conf = pd.read_pickle(D / "force_all_wells.pkl").loc[pr.index, "FORCE_2020_LITHOFACIES_CONFIDENCE"]
print("confidence values (sand/shale samples):"); print(conf.value_counts(dropna=False).sort_index().to_string())

METHODS = ["HGB_ALL", "HGB_GR", "RF_ALL", "LINEAR_0.35", "LINEAR_0.50", "ND_0.40", "ND_0.50"]
rows = []
for name, mask in [("all labels", conf.notna() | conf.isna()), ("confidence 1 only", conf == 1)]:
    for m in METHODS:
        ok = mask & pr[m].notna()
        y, p = pr.y[ok], pr[m][ok].astype(int)
        rows.append(dict(labels=name, method=m, n=int(ok.sum()), balanced_acc=round(balanced_accuracy_score(y, p), 3),
                         f1_sand=round(f1_score(1 - y, 1 - p), 3)))
res = pd.DataFrame(rows)
res.to_csv(D / "results_confidence.csv", index=False)
print(res.pivot(index="method", columns="labels", values="balanced_acc").loc[METHODS].to_string())
print(res[res.labels == "confidence 1 only"][["method", "n", "f1_sand"]].to_string(index=False))
