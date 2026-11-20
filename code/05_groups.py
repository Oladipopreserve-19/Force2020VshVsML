"""Assign NPD lithostratigraphic groups to samples and compare methods by group,
and characterise the 27 wells where the GR 0.35 cutoff matched or beat ML.
Run: .venv/bin/python -I code/05_groups.py
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
pr = pd.read_pickle(D / "predictions.pkl")
tops = pd.read_excel(D / "lithostrat_groups.xlsx")
tops = tops.rename(columns={"Well identifier": "WELL"})
tops["WELL"] = tops.WELL.str.strip()
tops["GROUP"] = tops.Surface.str.replace(" Top", "", regex=False).str.strip().str.title()
pr["WELL"] = pr.WELL.str.strip()

# LAS well names carry field names ("15/9-13 Sleipner East Appr") or lack spaces
# ("30/3-5S"); match to NPD identifiers by comparing without spaces.
def to_npd(name, npd=sorted(set(tops.WELL), key=len, reverse=True)):
    a = name.replace(" ", "")
    for t in npd:
        b = t.replace(" ", "")
        if a == b or (a.startswith(b) and not a[len(b)].isdigit()):
            return t
    return name
name_map = {w: to_npd(w) for w in pr.WELL.unique()}
pd.Series(name_map, name="npd_well").to_csv(D / "well_name_map.csv")
pr["WELL_LAS"] = pr.WELL
pr["WELL"] = pr.WELL.map(name_map)
matched = sorted(set(pr.WELL) & set(tops.WELL))
print(f"wells with NPD group tops: {len(matched)} of {pr.WELL.nunique()}")

parts = []
for w, g in pr.groupby("WELL"):
    t = tops[tops.WELL == w].sort_values("MD")[["MD", "GROUP"]]
    g = g.sort_values("DEPTH_MD").reset_index()
    if t.empty:
        g["GROUP"] = None
    else:
        g = pd.merge_asof(g, t, left_on="DEPTH_MD", right_on="MD", direction="backward").drop(columns="MD")
    parts.append(g)
s = pd.concat(parts).set_index("index")
print(f"samples assigned a group: {s.GROUP.notna().mean():.1%}")

METHODS = ["HGB_ALL", "LINEAR_0.35", "ND_0.40"]
rows = []
for grp, g in s.dropna(subset=["GROUP"]).groupby("GROUP"):
    if len(g) < 5000 or g.y.nunique() < 2: continue
    r = {"group": grp, "n": len(g), "wells": g.WELL.nunique(), "shale_frac": round(g.y.mean(), 3)}
    for m in METHODS:
        ok = g[m].notna()
        r[m] = round(balanced_accuracy_score(g.y[ok], g[m][ok].astype(int)), 3) if ok.sum() > 1000 and g.y[ok].nunique() == 2 else None
    r["ML_gain_vs_GR"] = round(r["HGB_ALL"] - r["LINEAR_0.35"], 3)
    rows.append(r)
bygrp = pd.DataFrame(rows).sort_values("n", ascending=False)
bygrp.to_csv(D / "results_by_group.csv", index=False)

# The 27 wells
pw = pd.read_csv(D / "results_per_well.csv").dropna(subset=["HGB_ALL", "LINEAR_0.35"])
pw["WELL"] = pw.WELL.str.strip().map(name_map)
pw["cutoff_ok"] = pw["LINEAR_0.35"] >= pw.HGB_ALL
dom = s.dropna(subset=["GROUP"]).groupby("WELL").GROUP.agg(lambda x: x.value_counts().index[0])
sand_dom = s.dropna(subset=["GROUP"])[lambda x: x.y == 0].groupby("WELL").GROUP.agg(lambda x: x.value_counts().index[0])
pw = pw.join(dom.rename("dominant_group"), on="WELL").join(sand_dom.rename("main_sand_group"), on="WELL")
pw.to_csv(D / "results_per_well_groups.csv", index=False)

pd.set_option("display.width", 180)
print("\nBalanced accuracy by NPD group (groups with >= 5000 samples):")
print(bygrp.to_string(index=False))
print("\nCutoff-as-good wells vs ML-better wells:")
print(pw.groupby("cutoff_ok")[["shale_frac", "LINEAR_0.35", "HGB_ALL"]].median().round(3))
print("\nMain sand-bearing group, cutoff-as-good wells (27):")
print(pw[pw.cutoff_ok].main_sand_group.value_counts().to_string())
print("\nMain sand-bearing group, ML-better wells (91):")
print(pw[~pw.cutoff_ok].main_sand_group.value_counts().to_string())
