# Mixed-Categorical-Predictors-and-Numerical-Target-Correlation-Screening-Pipeline
> Exhaustive, disk-backed feature-transformation grid with nested-CV Elastic Net to deliver out-of-sample R² scores per target

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-grey.svg)](#)

## Overview
We built a correlation screening pipeline for mixed predictors and targets that tested multiple preprocessors in combination, including imputers (mean, median, most frequent), scalers (standard, min-max, robust), and encoders (one-hot, target, frequency). For each feature-transform variant, the pipeline computed the appropriate association metric by type (Pearson or Spearman for numeric, point-biserial for binary targets, Cramér’s V and mutual information for categorical pairs) with stratified cross-validation and FDR adjustment. To handle thousands of combinations efficiently, I streamed batches and stored intermediate matrices and result arrays in NumPy memmap files and then reported ranked tables with effect sizes and adjusted p-values.

## Features
- Provides a plethora of CLI flags for I/O paths, targets, CV settings, grid caps, and switches to enable/disable transformation families.
- Loads raw data and calculates the correlation of one target at a time.
- Scans predictors' content and splits them into categorical vs numeric.
- Enumerates base predictor combos (size ≥ 2), then builds per-predictor transformation menus (cat: one-hot, rare one-hot, OOF, smoothed; num: raw, raw+{standard|minmax|robust}, imputers, imputed+scalers), pruned based on CLI flags.
- Forms the full cartesian product of choices per combo (optionally capped) ensuring each grid contains exactly one variant per predictor.
- Materializes each grid to a disk-backed matrix (numpy.memmap) by streaming transformed blocks and mean-filling NaNs. Additionally, it optionally applies a two-pass, blockwise standardization to z-score all features in place.
- Runs nested cross-validation Elastic Net: inner CV tunes 𝛼 and 𝑙1_𝑟𝑎𝑡𝑖𝑜 and outer CV reports out-of-sample $𝑅^2$ (mean ± std) for fair generalization.
- Logs results per grid (target, combo/grid IDs, base columns, variant description, row/column counts, $𝑅^2$ stats, tuned 𝛼, 𝑙1_𝑟𝑎𝑡𝑖𝑜), offering optional verbose progress prints.
- Writes one CSV per target, enabling the scientist to sort/filter for the best-performing combinations and understand which transformations helped the most.


## Quick Start
```bash
# 1) Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate    # on Windows: .venv\Scripts\activate

# 2) Install necessary Python modules
pip install -r requirements.txt

# 3) Run a minimal example
python correlation_pipeline_grid_memmap_cv_enet.py --data /mnt/data/source_data.csv --print_output_dir /mnt/data --targets  "Target1_Name" "Target2_Name" --output_dir /mnt/data --memmap_dir /mnt/data/tmp --cv_folds 5 --enet_cv_folds 5 --standardize --disable_cat_oof --disable_cat_smoothed

