# Tutorial

Three short exercises, from a test with no downloads to a full reproduction of the paper.

## 1. Install

```bash
git clone https://github.com/Oladipopreserve-19/Force2020VshVsML.git
cd Force2020VshVsML
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Test the method on a synthetic well (no downloads)

```bash
.venv/bin/python -I code/example_single_well.py --synthetic
```

Expected output: about 3,290 samples, a shale fraction near 0.56 and balanced accuracies near 1.0 for every method.
A CSV and a plot are written to `example_output/`.

## 3. Run the method on one real well

Download `LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip` from
https://doi.org/10.5281/zenodo.4351156 into `data/force_las.zip`, then extract one well and run the example:

```bash
unzip -j data/force_las.zip "*/25_6-3.las" -d example_input
.venv/bin/python -I code/example_single_well.py --las example_input/25_6-3.las --cutoff 0.35
```

Expected output: 14,894 samples and a balanced accuracy of 0.780 for the linear estimator.
Try `--cutoff 0.5` to see how the score changes with the cutoff, which is the main point of Section 4.1 of the paper.

You can use the same command on your own LAS file, as long as it has a `GR` curve.

## 4. Reproduce the full study

Also download `NPD_Lithostratigraphy_groups_all_wells.xlsx` into `data/lithostrat_groups.xlsx`, then:

```bash
.venv/bin/python -I code/01_load.py
.venv/bin/python -I code/02_classical.py
.venv/bin/python -I code/03_ml.py
.venv/bin/python -I code/04_figures.py
.venv/bin/python -I code/05_groups.py
.venv/bin/python -I code/06_confidence.py
```

Compare `data/results_ml_vs_classical.csv` with Table 2 of the paper: gradient boosting with all logs should give
a balanced accuracy of 0.888 on all samples, and linear gamma ray at 0.35 should give 0.780.
