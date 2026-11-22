# When is a gamma-ray cutoff enough?

Code and results comparing classical shale-volume (Vsh) estimators with machine learning for sandstone/shale classification on **118 Norwegian North Sea wells** (FORCE 2020 dataset).

**Author:** Oladipo Preserve Oluwadimimu, University of Ilorin · ORCID [0009-0008-3770-1704](https://orcid.org/0009-0008-3770-1704)
**Preprint:** EarthArXiv (submitted 22 November 2026; DOI to be added after moderation)

## Main results (held-out wells, balanced accuracy, sandstone vs shale)

| Method | Balanced accuracy |
|---|---|
| Gradient boosting, all logs | 0.888 |
| Neutron-density, cutoff 0.40 | 0.786 |
| Linear gamma ray, cutoff 0.35 | 0.780 |
| Gradient boosting, gamma ray + depth only | 0.744 |
| Linear gamma ray, cutoff 0.50 | 0.736 |

* Machine learning with all logs beat the best gamma-ray cutoff in 91 of 118 wells.
* With gamma ray alone, machine learning was no better than a cutoff: the gain comes from the extra logs.
* Larionov and Clavier corrections reduced accuracy at every cutoff tested.
* The benefit depends on stratigraphy: largest in the Rogaland and Viking groups, absent in the Hegre and Vestland groups.

## Documentation

* `docs/TUTORIAL.md`: install, a test on a synthetic well, one real well, and the full reproduction
* `docs/USER_GUIDE.md`: inputs, outputs, options, requirements and expected behaviour of every script

Quick test with no downloads:

```bash
.venv/bin/python -I code/example_single_well.py --synthetic
```

Requirements: Python 3.10 or later, packages in `requirements.txt`, about 16 GB RAM and 1 GB disk for the full run. CPU only.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# Download into data/ from https://doi.org/10.5281/zenodo.4351156:
#   LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip  -> data/force_las.zip
#   NPD_Lithostratigraphy_groups_all_wells.xlsx                       -> data/lithostrat_groups.xlsx
.venv/bin/python -I code/01_load.py        # read 118 LAS files
.venv/bin/python -I code/02_classical.py   # classical Vsh estimators and cutoff sensitivity
.venv/bin/python -I code/03_ml.py          # classifiers, 5-fold cross-validation grouped by well
.venv/bin/python -I code/04_figures.py     # figures and tables
.venv/bin/python -I code/05_groups.py      # results by NPD lithostratigraphic group
.venv/bin/python -I code/06_confidence.py  # high-confidence labels only
```

## Contents

* `code/` analysis scripts, run in order; `code/example_single_well.py` applies the method to any LAS file
* `docs/` user guide and tutorial
* `data/results_*.csv` derived results tables; `data/provenance_wells.csv` list of wells and curves; `data/well_name_map.csv` LAS to NPD well names
* `figures/` figures 1 to 5 (300 dpi)

## Data and licence

Input data: Bormann et al. (2020), FORCE 2020 well log and lithofacies dataset, [doi:10.5281/zenodo.4351156](https://doi.org/10.5281/zenodo.4351156), CC BY 4.0; source logs under the Norwegian Licence for Open Government Data (NLOD) 2.0. Raw data are not redistributed here.

Code: MIT licence. Results tables and figures: CC BY 4.0.

## Use of AI tools

Claude (Anthropic) assisted with writing code and text. The author designed the study, chose the cutoffs and checked all results.
