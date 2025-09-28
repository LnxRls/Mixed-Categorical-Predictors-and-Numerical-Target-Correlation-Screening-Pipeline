# Mixed-Categorical-Predictors-and-Numerical-Target-Correlation-Screening-Pipeline
> ONE-LINE TAGLINE THAT EXPLAINS WHAT THE SCRIPT DOES

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-grey.svg)](#)

## Overview
We built a correlation screening pipeline for mixed predictors and targets that tested multiple preprocessors in combination, including imputers (mean, median, most frequent), scalers (standard, min-max, robust), and encoders (one-hot, target, frequency). For each feature-transform variant, the pipeline computed the appropriate association metric by type (Pearson or Spearman for numeric, point-biserial for binary targets, Cramér’s V and mutual information for categorical pairs) with stratified cross-validation and FDR adjustment. To handle thousands of combinations efficiently, I streamed batches and stored intermediate matrices and result arrays in NumPy memmap files and then reported ranked tables with effect sizes and adjusted p-values.

## Features
- Parses CLI flags for I/O paths, targets, CV settings, grid caps, and switches to enable/disable transformation families.
- Loads data and, per target, excludes other targets, splits predictors into categorical vs numeric.
- Enumerates base predictor combos (size ≥ 2), then builds per-predictor transformation menus (cat: one-hot, rare one-hot, OOF, smoothed; num: raw, raw+{standard|minmax|robust}, imputers, imputed+scalers), pruned by flags.
- Forms the full cartesian product of choices per combo (optionally capped) so each grid contains exactly one variant per predictor.
- Materializes each grid to a disk-backed matrix (numpy.memmap) by streaming transformed blocks and mean-filling NaNs; optionally applies a two-pass, blockwise standardization to z-score all features in place.
- Runs nested cross-validation Elastic Net: inner CV tunes 𝛼 and 𝑙1_𝑟𝑎𝑡𝑖𝑜; outer CV reports out-of-sample 𝑅^{2} (mean ± std) to fairly estimate generalization.
- Logs results per grid (target, combo/grid IDs, base columns, variant description, row/column counts, 𝑅^{2} stats, tuned 𝛼, 𝑙1_𝑟𝑎𝑡𝑖𝑜); optional verbose progress prints.
- Writes one CSV per target, enabling you to sort/filter for the best-performing combinations and understand which transformations helped.


## Quick Start
```bash
# 1) Create and activate an environment
python -m venv .venv && source .venv/bin/activate    # on Windows: .venv\Scripts\activate

# 2) Install
pip install -r requirements.txt

# 3) Run a minimal example
python PATH/TO/SCRIPT.py --input data/sample.csv --output out/results.csv
