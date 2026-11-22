"""Apply the classical shale-volume estimators and a cutoff to one well.

This is the how-to example for users who want to run the method on their own LAS file.
It uses the same rules as code/02_classical.py: endpoints are the 5th and 95th percentiles
of the well's own gamma-ray log, and a sample is called shale where Vsh exceeds the cutoff.

Examples:
    .venv/bin/python -I code/example_single_well.py --synthetic
    .venv/bin/python -I code/example_single_well.py --las path/to/well.las --cutoff 0.35

Outputs (in the folder given by --out, default example_output/):
    <well>_vsh.csv   depth, GR, Vsh by each method, and the sand/shale call
    <well>_vsh.png   gamma ray, linear Vsh with the cutoff, and the sand/shale call
If the LAS file has FORCE 2020 lithology labels, balanced accuracy against them is printed.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LITH = {30000: "Sandstone", 65000: "Shale"}


def vsh_estimators(gr, rhob=None, nphi=None):
    """Return a DataFrame of Vsh from the gamma-ray index and, if available, neutron-density."""
    gmin, gmax = np.nanpercentile(gr, 5), np.nanpercentile(gr, 95)
    igr = np.clip((gr - gmin) / (gmax - gmin), 0, 1)
    out = pd.DataFrame({
        "VSH_LINEAR": igr,
        "VSH_LARIONOV_TERT": np.clip(0.083 * (2 ** (3.7 * igr) - 1), 0, 1),
        "VSH_LARIONOV_OLD": np.clip(0.33 * (2 ** (2 * igr) - 1), 0, 1),
        "VSH_CLAVIER": np.clip(1.7 - np.sqrt(np.clip(3.38 - (igr + 0.7) ** 2, 0, None)), 0, 1),
    })
    if rhob is not None and nphi is not None:
        nphi = np.where(nphi > 1.5, nphi / 100, nphi)  # percent to fraction
        sep = nphi - (2.65 - rhob) / (2.65 - 1.0)
        lo, hi = np.nanpercentile(sep, 5), np.nanpercentile(sep, 95)
        out["VSH_ND"] = np.clip((sep - lo) / (hi - lo), 0, 1)
    return out


def synthetic_well(seed=0):
    """A simple sand-shale sequence with noise, for testing without downloading data."""
    rng = np.random.default_rng(seed)
    depth = np.arange(1000.0, 1500.0, 0.152)
    shale = (np.sin(depth / 12.0) + 0.3 * rng.standard_normal(depth.size)) > -0.2
    gr = np.where(shale, 105, 40) + 8 * rng.standard_normal(depth.size)
    rhob = np.where(shale, 2.50, 2.25) + 0.03 * rng.standard_normal(depth.size)
    nphi = np.where(shale, 0.35, 0.20) + 0.02 * rng.standard_normal(depth.size)
    lith = np.where(shale, 65000, 30000)
    return "SYNTHETIC", pd.DataFrame({"DEPTH_MD": depth, "GR": gr, "RHOB": rhob, "NPHI": nphi,
                                      "FORCE_2020_LITHOFACIES_LITHOLOGY": lith})


def read_las(path):
    import lasio
    las = lasio.read(path)
    df = las.df().reset_index()
    df = df.rename(columns={df.columns[0]: "DEPTH_MD"}).loc[:, lambda x: ~x.columns.duplicated()]
    well = str(las.well["WELL"].value).strip() or Path(path).stem
    return well, df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--las", help="path to a LAS file with a GR curve")
    src.add_argument("--synthetic", action="store_true", help="use a built-in synthetic well")
    ap.add_argument("--cutoff", type=float, default=0.35, help="Vsh cutoff for shale (default 0.35)")
    ap.add_argument("--out", default="example_output", help="output folder")
    a = ap.parse_args()

    well, df = synthetic_well() if a.synthetic else read_las(a.las)
    if "GR" not in df:
        raise SystemExit("The LAS file has no GR curve.")
    df = df[df.GR.notna()].reset_index(drop=True)
    has_nd = {"RHOB", "NPHI"} <= set(df.columns)
    v = vsh_estimators(df.GR.values, df.RHOB.values if has_nd else None, df.NPHI.values if has_nd else None)
    res = pd.concat([df[["DEPTH_MD", "GR"]], v], axis=1)
    res["CALL_LINEAR"] = np.where(res.VSH_LINEAR > a.cutoff, "Shale", "Sandstone")

    out = Path(a.out); out.mkdir(exist_ok=True)
    name = well.replace("/", "_").replace(" ", "_")
    res.to_csv(out / f"{name}_vsh.csv", index=False)
    print(f"{well}: {len(res):,} samples, cutoff {a.cutoff}, shale fraction called {np.mean(res.CALL_LINEAR == 'Shale'):.2f}")

    if "FORCE_2020_LITHOFACIES_LITHOLOGY" in df:
        from sklearn.metrics import balanced_accuracy_score
        lab = df.FORCE_2020_LITHOFACIES_LITHOLOGY.map(LITH)
        m = lab.notna()
        for col in v.columns:
            call = np.where(res.loc[m, col] > a.cutoff, "Shale", "Sandstone")
            print(f"  {col:<18} balanced accuracy {balanced_accuracy_score(lab[m], call):.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(7, 8), sharey=True)
    ax[0].plot(res.GR, res.DEPTH_MD, lw=0.5, color="0.2"); ax[0].set_xlabel("GR (API)")
    ax[1].plot(res.VSH_LINEAR, res.DEPTH_MD, lw=0.5, color="0.2"); ax[1].axvline(a.cutoff, ls=":", color="k")
    ax[1].set_xlabel("Linear Vsh"); ax[1].set_xlim(0, 1)
    ax[2].fill_betweenx(res.DEPTH_MD, 0, (res.CALL_LINEAR == "Shale").astype(int), step="mid", color="tab:blue", lw=0)
    ax[2].set_xlabel("Shale call"); ax[2].set_xticks([])
    ax[0].set_ylabel("Measured depth (m)"); ax[0].invert_yaxis()
    fig.tight_layout(); fig.savefig(out / f"{name}_vsh.png", dpi=150)
    print(f"Wrote {out / (name + '_vsh.csv')} and {out / (name + '_vsh.png')}")


if __name__ == "__main__":
    main()
