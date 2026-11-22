# User guide

This guide describes the inputs, outputs, options and expected behaviour of each script in `code/`.
For a first run, follow `docs/TUTORIAL.md`.

## Requirements

* Python 3.10 or later (tested with Python 3.15 on macOS).
* Packages pinned in `requirements.txt`: lasio, numpy, pandas, scikit-learn, matplotlib, openpyxl, tabulate.
* Memory: the full analysis loads about 2.3 million depth samples (a 220 MB table). 16 GB of RAM is enough; 8 GB should work but has not been tested.
* Disk: about 1 GB for the downloaded data and intermediate files.
* No GPU is needed. All models run on the CPU.

## Input data

The analysis uses the FORCE 2020 well log and lithofacies dataset (Bormann et al., 2020,
https://doi.org/10.5281/zenodo.4351156). Two files must be downloaded into `data/`:

| File on Zenodo | Save as |
|---|---|
| `LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip` | `data/force_las.zip` |
| `NPD_Lithostratigraphy_groups_all_wells.xlsx` | `data/lithostrat_groups.xlsx` |

The raw data are not included in this repository because they are already openly archived by their authors.

## Scripts, run in order

| Script | Input | Output | What it does |
|---|---|---|---|
| `01_load.py` | `data/force_las.zip` | `data/force_all_wells.pkl`, `data/provenance_wells.csv` | Reads the 118 LAS files and keeps depth, logs and lithology labels. |
| `02_classical.py` | `force_all_wells.pkl` | `data/vsh_classical.pkl`, `data/results_classical_cutoffs.csv`, `data/results_mixed_class.csv` | Computes linear, Larionov (Tertiary and older rocks), Clavier and neutron-density Vsh with per-well 5th and 95th percentile endpoints, and scores each at cutoffs 0.35, 0.40, 0.50 and 0.60. |
| `03_ml.py` | `force_all_wells.pkl`, `vsh_classical.pkl` | `data/predictions.pkl`, `data/results_ml_vs_classical.csv`, `data/results_per_well.csv` | Trains histogram gradient boosting (all logs; gamma ray and depth only) and random forest with five-fold cross-validation grouped by well, so no test well is seen during training. |
| `04_figures.py` | results files | `figures/fig1` to `fig5` | Draws the figures used in the paper. |
| `05_groups.py` | predictions, `lithostrat_groups.xlsx` | `data/results_by_group.csv`, `data/results_per_well_groups.csv` | Assigns samples to NPD lithostratigraphic groups and scores each method by group. |
| `06_confidence.py` | predictions | `data/results_confidence.csv` | Repeats the scores on high-confidence labels only. |
| `example_single_well.py` | one LAS file, or none with `--synthetic` | CSV and PNG in `example_output/` | Applies the classical estimators and a cutoff to a single well. See below. |

Scripts `07_export.py` and `08_eartharxiv.py` only build the manuscript files and are not needed to reproduce the results.

## Options for `example_single_well.py`

| Option | Default | Meaning |
|---|---|---|
| `--las PATH` | | LAS file to process. It must contain a `GR` curve. `RHOB` and `NPHI` are used if present. |
| `--synthetic` | | Use a built-in synthetic sand-shale well instead of a LAS file. |
| `--cutoff X` | 0.35 | Vsh above which a sample is called shale. |
| `--out DIR` | `example_output` | Folder for the output files. |

Output columns in `<well>_vsh.csv`: `DEPTH_MD`, `GR`, `VSH_LINEAR`, `VSH_LARIONOV_TERT`, `VSH_LARIONOV_OLD`,
`VSH_CLAVIER`, `VSH_ND` (only when density and neutron logs exist) and `CALL_LINEAR` (Sandstone or Shale).

## Expected behaviour

* Running scripts 01 to 06 in order reproduces the tables and figures in the paper. Random seeds are fixed (42),
  so results should match to the third decimal place with the pinned package versions.
* `example_single_well.py --synthetic` prints balanced accuracies close to 1.0, because the synthetic well has
  clean, strongly contrasted sand and shale.
* On a real FORCE well with labels, it prints a balanced accuracy for each method. For well 25/6-3 the linear
  estimator at 0.35 gives 0.780.
* Endpoints are always taken from the well's own logs, never from the labels, so the method can be applied to wells without lithology labels.

## Limitations

The method uses one pair of endpoints per well. Shales whose radioactivity changes strongly with depth or mineralogy
are handled poorly (see Section 4.4 of the paper). The classifiers use measured depth as a feature, which may not
transfer to other basins.
