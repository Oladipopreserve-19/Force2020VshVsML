"""Load FORCE 2020 LAS files into one table.

Source: FORCE 2020 well log and lithofacies dataset, Zenodo doi:10.5281/zenodo.4351156
(CC BY 4.0; source logs under NLOD 2.0). Run from the Preprint5_FORCE folder:
    .venv/bin/python -I code/01_load.py
"""
import zipfile, io, sys
from pathlib import Path
import lasio
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ZIP = DATA / "force_las.zip"
OUT = DATA / "force_all_wells.pkl"

CURVES = ["DEPTH_MD", "GR", "RHOB", "NPHI", "DTC", "PEF", "CALI", "RDEP", "RMED",
          "FORCE_2020_LITHOFACIES_LITHOLOGY", "FORCE_2020_LITHOFACIES_CONFIDENCE"]

frames, log = [], []
with zipfile.ZipFile(ZIP) as z:
    names = [n for n in z.namelist() if n.lower().endswith(".las")]
    for n in names:
        las = lasio.read(io.StringIO(z.read(n).decode("utf-8", errors="replace")))
        df = las.df().reset_index()
        df = df.rename(columns={df.columns[0]: "DEPTH_MD"})
        well = str(las.well["WELL"].value).strip() or Path(n).stem
        keep = [c for c in CURVES if c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df = df[list(dict.fromkeys(keep))].copy()
        df["WELL"] = well
        df["SET"] = Path(n).parent.name or "unknown"
        frames.append(df)
        log.append((well, Path(n).parent.name, len(df), ",".join(keep)))

all_df = pd.concat(frames, ignore_index=True)
all_df.to_pickle(OUT)
pd.DataFrame(log, columns=["well", "set", "rows", "curves"]).to_csv(DATA / "provenance_wells.csv", index=False)
print(f"{len(names)} LAS files, {all_df.WELL.nunique()} wells, {len(all_df):,} rows")
print(all_df.groupby("SET").WELL.nunique())
