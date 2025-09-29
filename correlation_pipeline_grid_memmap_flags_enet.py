"""
Cross-Validated Elastic Net - based (using Overall R² as Perfomance Metric) 
Correlation of Memmap Grid Combinations of Predictors and their 
Transformations with one or more Targets, one Target at a time.   
=======================================================================

This script evaluates each base predictor combination and the full cartesian
product of their transformations (using a series of parameter flags), but instead of 
calculating the univariate Pearson coefficient r for each column (Predictor) - Target pair 
it calculates a SINGLE **out-of-sample R²** summary for each grid. A grid is computed using 
**cross-validated Elastic Net Regression** with **inner CV tuning** (ElasticNetCV). 
The script also uses **numpy.memmap** to avoid large in-memory hog matrices.

Quick start
-----------
python correlation_pipeline_grid_memmap_cv_enet.py \
  --data /path/to/text.csv \
  --print_output_dir /path/to/print_output \
  --targets "Target1_Name" "Target2_Name" \
  --output_dir /path/to/out \
  --memmap_dir /path/to/out/tmp \
  --cv_folds 5 \
  --enet_cv_folds 5 \
  --standardize \
  --max_combos_per_target 50 \
  --max_grids_per_combo 200 \
  --disable_cat_oof \
  --disable_cat_smoothed \
  --disable_num_imputer_scalers \
  --block_size 512 \
  --verbose

Code Execution Example (UnWrap the following long command to one straight line)
----------------------  
python correlation_pipeline_grid_memmap_cv_enet.py --data /mnt/data/source_data.csv --print_output_dir /mnt/data --targets  "Target1_Name" "Target2_Name" --output_dir /mnt/data --memmap_dir /mnt/data/tmp --cv_folds 5 --enet_cv_folds 5 --standardize --disable_cat_oof --disable_cat_smoothed

Outputs
-------
One CSV file per target:  correlations_<TARGET>_GRID_MEMMAP_ENET_R2.csv
Columns: target, combo_id, grid_id, base_columns, variant, n_rows, n_cols, r2_mean, r2_std, alpha, l1_ratio

Notes
-----
- If the target has zero variance, R² is set to NaN.
- For very wide grids, consider enabling --standardize (in-place z-scoring of columns on the memmap) to make penalties comparable
"""

from __future__ import annotations
import argparse
import os
import tempfile
import time

from itertools import combinations, product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score


# ----------------------------- CLI flags & Feature Eng Config ----------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memmap-enabled grid search with CV Elastic Net R² overall summary.")

    # I/O and general arguments
    parser.add_argument("--data", type=str, required=True, help="Path to CSV or Excel file")
    parser.add_argument("--print_output_dir", type=str, required=True, help="Path to file receiving stdout")
    parser.add_argument("--sheet", type=str, default=None, help="Excel sheet name (if reading xlsx)")
    parser.add_argument("--targets", type=str, nargs="+", default= ["Target2"],     # ["Target1", "Target2"],
                        help="List of target column names to iterate over")
    parser.add_argument("--output_dir", type=str, default=".", help="Where to write CSV outputs")
    parser.add_argument("--memmap_dir", type=str, default=None, 
                        help="Directory for memmap temp files (defaults to output_dir)")

    # Encoding/scaling hyperparams
    parser.add_argument("--min_count_threshold", type=int, default=30, help="Rare-level grouping threshold (categoricals)")
    parser.add_argument("--k_folds", type=int, default=5, help="K-folds for OOF encoding")
    parser.add_argument("--random_state", type=int, default=54, help="Random state for KFold")
    parser.add_argument("--block_size", type=int, default=1024, help="Number of columns to process per block for standardization/calc")
    parser.add_argument("--standardize", action="store_true", help="Z-score columns in memmap before model fitting (recommended)")

    # Verbosity and caps
    parser.add_argument("--verbose", action="store_true", help="Print per-grid summaries.")
    parser.add_argument("--max_combos_per_target", type=int, default=None, help="Cap number of base predictor combinations per target")
    parser.add_argument("--max_grids_per_combo", type=int, default=None, help="Cap number of transformation grids per base combo")

    # Elastic Net CV settings
    parser.add_argument("--cv_folds", type=int, default=5, help="Outer CV folds for out-of-sample R²")
    parser.add_argument("--enet_cv_folds", type=int, default=5, help="Inner CV folds for ElasticNetCV tuning")
    parser.add_argument("--l1_ratio_grid", type=float, nargs="+", default=[0.1, 0.5, 0.9], help="Grid of l1_ratio values")
    parser.add_argument("--alphas_grid", type=float, nargs="+", default=100, help="Optional grid of alpha values. If None, ElasticNetCV picks automatically")

    # --------------------------------- Disable Flags (for grid size mngt) --------------------------
    # Categorical families
    parser.add_argument("--disable_cat_one_hot", action="store_true", help="Disable categorical one-hot")
    parser.add_argument("--disable_cat_rare", action="store_true", help="Disable categorical one-hot with rare-level grouping")
    parser.add_argument("--disable_cat_oof", action="store_true", help="Disable categorical OOF target encoding")
    parser.add_argument("--disable_cat_smoothed", action="store_true", help="Disable categorical smoothed target encoding")

    # Numeric families
    parser.add_argument("--disable_num_raw", action="store_true", help="Disable numeric raw (untransformed)")
    parser.add_argument("--disable_num_raw_scalers", action="store_true", help="Disable numeric raw+{scaler} variants")
    parser.add_argument("--disable_num_imputers", action="store_true", help="Disable numeric imputers (mean/median/most_frequent)")
    parser.add_argument("--disable_num_imputer_scalers", action="store_true", help="Disable numeric imputed+{scaler} variants")

    # Fine-grained toggles
    parser.add_argument("--disable_scaler_standard", action="store_true", help="Disable StandardScaler variants")
    parser.add_argument("--disable_scaler_minmax", action="store_true", help="Disable MinMaxScaler variants")
    parser.add_argument("--disable_scaler_robust", action="store_true", help="Disable RobustScaler variants")

    parser.add_argument("--disable_imputer_mean", action="store_true", help="Disable mean imputer")
    parser.add_argument("--disable_imputer_median", action="store_true", help="Disable median imputer")
    parser.add_argument("--disable_imputer_most_frequent", action="store_true", help="Disable most_frequent imputer")

    return parser.parse_args()


# ------------------------------ Execution Mngt & stdout Redirection ----------------------------

def user_ctrl_exec():
    """
    Pauses code execution and proceeds per end user's response, i.e., stops for 'n' and resumes for 'y' 

    Args:
        None 
    """
    user_response = input("Continue? (y/n): ").lower()

    if user_response == "n":
        print("Program terminated by user !!!")
        exit()  # Stops the program execution
    elif user_response == "y":
        print("Resuming code execution ...")
    else:
        print("Invalid input, enter 'y' or 'n'")


# function to save screen stdout to a file and show it to screen

def save_stdout(txt2print, args, cr_lf=False, title=None):
    """
    Sets print text to the console and saves it to a file

    Args:
        txt2print (str): String to be printed and saved to an output file
        file_path (str): The network location of the file where the output will be saved
                         Defaults to "print_output.txt".
        title (str, optional): An optional title to prepend to the text
        cr_lf (bool): If True, adds a carriage return and a newline
                                    between the title and the text, if a title is provided
    """
    
    output=''

    if title:
        output += "\n"
        output += title

        if cr_lf:
            output += "\n"
        else:
            output += "  "
    
    if isinstance(txt2print, str):
        output += txt2print
    else:
        output += str(txt2print) 

    # print to screen 
    print(output)

    # append to a file
    try:
        with open(args.print_output_dir + "/output_stdout.txt", 'a', encoding='utf-8') as file:
            file.write('\n')
            file.write(output, )
    except IOError as e:
        print(f"Error saving to file '{args.print_output_dir}' '/output_stdout.txt': {e}")


# --------------------------- Data I/O Helper Functions --------------------------------

def read_data(path: str, sheet: str | None = None) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)


def chk_name(s: str) -> str:
    return "".join([ch if ch.isalnum() or ch in ("_",) else "_" for ch in s.replace(" ", "_")])


# ---------------------------- Transformation Helper Functions -------------------------

######## --------------------- Identify Column Data Types ------------------------------

def identify_column_data_types(df: pd.DataFrame, target: str, mutually_exclusive: set) -> Tuple[List[str], List[str]]:
    exclude_cols = set([target]) | (mutually_exclusive - {target})
    predictors = [c for c in df.columns if c not in exclude_cols]
    cat_cols = [c for c in predictors if df[c].dtype == object]
    num_cols = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c])]
    return cat_cols, num_cols


def identify_rare_levels(series: pd.Series, min_count: int) -> pd.Series:
    counts = series.value_counts(dropna=False)
    keep = counts[counts >= min_count].index
    return series.where(series.isin(keep), other="Other")


######## ------------------------- Encoding Methods ---------------------------------

def oof_target_encode(x: pd.Series, y: pd.Series, n_splits: int, random_state: int) -> pd.Series:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    encoded = pd.Series(index=x.index, dtype=float)
    global_mean = y.mean()
    for tr_idx, va_idx in kf.split(x):
        x_tr, y_tr = x.iloc[tr_idx], y.iloc[tr_idx]
        means = y_tr.groupby(x_tr).mean()
        encoded.iloc[va_idx] = x.iloc[va_idx].map(means).fillna(global_mean).values
    return encoded


def smoothed_target_encode(x: pd.Series, y: pd.Series, m: float = 10.0) -> pd.Series:
    global_mean = y.mean()
    stats = y.groupby(x).agg(["mean", "count"]).rename(columns={"mean": "cat_mean", "count": "cat_count"})
    smooth = (stats["cat_count"] * stats["cat_mean"] + m * global_mean) / (stats["cat_count"] + m)
    return x.map(smooth).fillna(global_mean)


def one_hot_encode(x: pd.Series) -> pd.DataFrame:
    return pd.get_dummies(x.astype("object"), prefix=x.name, dtype=float)


######## ------------- Enabling/Disabling of Encoding/Imputation/Scaling Methods ---------------------

def enabled_scalers(args) -> List[str]:
    scalers = []
    if not args.disable_scaler_standard:
        scalers.append("standard")
    if not args.disable_scaler_minmax:
        scalers.append("minmax")
    if not args.disable_scaler_robust:
        scalers.append("robust")
    return scalers


def enabled_imputers(args) -> List[str]:
    imps = []
    if not args.disable_imputer_mean:
        imps.append("mean")
    if not args.disable_imputer_median:
        imps.append("median")
    if not args.disable_imputer_most_frequent:
        imps.append("most_frequent")
    return imps


######## ------ App'n of Encoding/Imputation/Scaling Methods to Categorical/Numeric Columns -------

def apply_categorical_method(col: str, series: pd.Series, y: pd.Series, method: str,
                             min_count: int, k_folds: int, random_state: int) -> Tuple[str, pd.DataFrame]:
    s = series.astype("object")
    if method == "one_hot":
        return f"{col}=one_hot", one_hot_encode(s)
    if method == "min_sample_threshold":
        rare = identify_rare_levels(s, min_count)
        return f"{col}=one_hot_rare", one_hot_encode(rare).add_suffix("__rare")
    if method == "oof_target":
        enc = oof_target_encode(s, y, n_splits=k_folds, random_state=random_state)
        return f"{col}=oof_target", pd.DataFrame({f"{col}__oof_target": enc})
    if method == "smoothed_target":
        enc = smoothed_target_encode(s, y, m=10.0)
        return f"{col}=smoothed_target", pd.DataFrame({f"{col}__smoothed_target": enc})
    raise ValueError(f"Unknown categorical method: {method}")


def apply_numeric_imputer(series: pd.Series, strategy: str) -> pd.Series:
    imputer = SimpleImputer(strategy=strategy)
    values = imputer.fit_transform(series.to_frame())
    return pd.Series(values.ravel(), index=series.index, name=series.name)


def apply_numeric_scaler(series: pd.Series, method: str) -> pd.Series:
    if method == "standard":
        scaler = StandardScaler()
        suffix = "__std"
    elif method == "minmax":
        scaler = MinMaxScaler()
        suffix = "__minmax"
    elif method == "robust":
        scaler = RobustScaler()
        suffix = "__robust"
    else:
        raise ValueError(f"Unknown numeric scaler: {method}")
    values = scaler.fit_transform(series.to_frame())
    return pd.Series(values.ravel(), index=series.index, name=series.name + suffix)


def categorical_options(series: pd.Series, y: pd.Series, args) -> List[Tuple[str, pd.DataFrame]]:
    col = series.name
    methods = []
    if not args.disable_cat_one_hot:
        methods.append("one_hot")
    if not args.disable_cat_rare:
        methods.append("min_sample_threshold")
    if not args.disable_cat_oof:
        methods.append("oof_target")
    if not args.disable_cat_smoothed:
        methods.append("smoothed_target")

    opts: List[Tuple[str, pd.DataFrame]] = []
    for m in methods:
        lbl, part = apply_categorical_method(col, series, y, m, args.min_count_threshold, args.k_folds, args.random_state)
        opts.append((lbl, part))
    return opts


def numeric_options(series: pd.Series, args) -> List[Tuple[str, pd.DataFrame]]:
    col = series.name
    opts: List[Tuple[str, pd.DataFrame]] = []

    scalers = enabled_scalers(args)
    imputers = enabled_imputers(args)

    if not args.disable_num_raw:
        opts.append((f"{col}=raw", pd.DataFrame({col: series})))

    if (not args.disable_num_raw_scalers) and scalers:
        for s in scalers:
            scaled = apply_numeric_scaler(series.fillna(series.mean()), s)
            opts.append((f"{col}=raw+{s}", scaled.to_frame()))

    if not args.disable_num_imputers and imputers:
        for imp in imputers:
            imputed = apply_numeric_imputer(series, imp)
            opts.append((f"{col}=imp_{imp}", pd.DataFrame({f"{col}__imp_{imp}": imputed})))
            if not args.disable_num_imputer_scalers and scalers:
                for s in scalers:
                    scaled = apply_numeric_scaler(imputed, s)
                    opts.append((f"{col}=imp_{imp}+{s}", scaled.to_frame()))

    return opts


# --------------------------- Helper to Count Choices w/o Building Dataframes ------------------

def _count_options_for_col(series, args):
    if series.dtype == object:  # categorical
        return int(not args.disable_cat_one_hot) \
             + int(not args.disable_cat_rare) \
             + int(not args.disable_cat_oof) \
             + int(not args.disable_cat_smoothed)
    else:  # numeric
        n_scalers = int(not args.disable_scaler_standard) + int(not args.disable_scaler_minmax) + int(not args.disable_scaler_robust)
        n_imps    = int(not args.disable_imputer_mean) + int(not args.disable_imputer_median) + int(not args.disable_imputer_most_frequent)
        cnt = 0
        if not args.disable_num_raw: cnt += 1
        if (not args.disable_num_raw_scalers) and n_scalers: cnt += n_scalers
        if not args.disable_num_imputers:
            cnt += n_imps
            if (not args.disable_num_imputer_scalers) and n_scalers:
                cnt += n_imps * n_scalers
        return cnt
    

# ------------------------------- Total Number of Grids ----------------------------------

def _estimate_total_grids(df, base_combos, args):
    total = 0
    save_stdout('', args, cr_lf=True, title="\nTotal Number of Grids Estimation") 

    for combo in base_combos:
        m = 1
        for col in combo:
            m *= _count_options_for_col(df[col], args)
        save_stdout(f"For '{col}' in combination: {combo} there are {m} transformation options", args, cr_lf=False, title=None) 

        if args.max_grids_per_combo is not None:
            m = min(m, args.max_grids_per_combo)
        save_stdout(f"For '{col}' in combination: {combo} there are {m} grids, or the min({m}, {args.max_grids_per_combo})", args, cr_lf=False, title=None) 

        total += m
    save_stdout(f"\nThe Total Number of Grids is: {total}", args, cr_lf=True, title=None) 
    return total


# ----------------------- Memmap Assembly & Standardization ------------------------

def create_memmap(parts: List[pd.DataFrame],
                  part_names: List[List[str]],
                  memmap_dir: str,
                  dtype=np.float64) -> Tuple[np.memmap, List[str], str]:
    n = parts[0].shape[0] if parts else 0
    total_cols = sum(len(names) for names in part_names)
    if total_cols == 0:
        return None, [], ""

    os.makedirs(memmap_dir, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=memmap_dir, suffix=".dat")
    tmp_path = tmp.name
    tmp.close()

    X_mm = np.memmap(tmp_path, mode="w+", dtype=dtype, shape=(n, total_cols))

    colnames: List[str] = []
    col_ptr = 0
    for df_part, names in zip(parts, part_names):
        vals = df_part.values.astype(dtype, copy=False)
        means = np.nanmean(vals, axis=0).astype(dtype)
        nan_rows, nan_cols = np.where(np.isnan(vals))
        if nan_rows.size > 0:
            vals[nan_rows, nan_cols] = means[nan_cols]

        width = vals.shape[1]
        X_mm[:, col_ptr:col_ptr+width] = vals
        colnames.extend(list(names))
        col_ptr += width

    X_mm.flush()
    return X_mm, colnames, tmp_path


def standardize_memmap_inplace(X_mm: np.memmap, block_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score columns in-place on the memmap. Returns (means, stds) for reference."""
    n, p = X_mm.shape
    means = np.empty(p, dtype=np.float64)
    stds  = np.empty(p, dtype=np.float64)

    # Pass 1: compute means/stds in blocks
    start = 0
    while start < p:
        end = min(start + block_size, p)
        block = np.array(X_mm[:, start:end], dtype=np.float64)
        means[start:end] = block.mean(axis=0)
        stds[start:end]  = block.std(axis=0, ddof=0)
        start = end

    # Avoid zero std
    stds[stds == 0] = 1.0

    # Pass 2: apply in-place transform
    start = 0
    while start < p:
        end = min(start + block_size, p)
        block = np.array(X_mm[:, start:end], dtype=np.float64)
        block -= means[start:end]
        block /= stds[start:end]
        X_mm[:, start:end] = block.astype(X_mm.dtype, copy=False)
        start = end

    X_mm.flush()
    return means, stds


# ----------------------- CV Elastic Net on Memmap -----------------------------

def cv_elasticnet_r2(X_mm: np.memmap,
                     y: np.ndarray,
                     outer_folds: int,
                     inner_folds: int,
                     l1_ratio_grid: List[float],
                     alphas_grid: List[float] | None,
                     random_state: int) -> Tuple[float, float, float, float]:
    """Return (r2_mean, r2_std, alpha_mean, l1_ratio_best) across outer folds."""
    y = y.astype(np.float64)
    if np.std(y, ddof=0) == 0 or np.isnan(np.std(y, ddof=0)):
        return np.nan, np.nan, np.nan, np.nan

    kf = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    r2s = []
    alphas = []
    l1s = []

    for tr_idx, te_idx in kf.split(y):
        X_tr = np.asarray(X_mm[tr_idx, :], dtype=np.float64)
        y_tr = y[tr_idx]
        X_te = np.asarray(X_mm[te_idx, :], dtype=np.float64)
        y_te = y[te_idx]

        enetcv = ElasticNetCV(
            l1_ratio=l1_ratio_grid,
            alphas=alphas_grid,
            cv=inner_folds,
            random_state=random_state,
            # n_alphas=100 if alphas_grid is None else None,
            max_iter=10000,
            precompute=False
        )

        enetcv.fit(X_tr, y_tr, )
        y_pred = enetcv.predict(X_te)
        r2 = r2_score(y_te, y_pred)
        r2s.append(r2)
        alphas.append(enetcv.alpha_)
        l1s.append(float(enetcv.l1_ratio_) if hasattr(enetcv, "l1_ratio_") else float(enetcv.l1_ratio))

    return float(np.nanmean(r2s)), float(np.nanstd(r2s)), float(np.nanmean(alphas)), float(np.nanmean(l1s))


def build_transform_grid_for_combo(df: pd.DataFrame, combo: Tuple[str, ...], y: pd.Series, args) -> List[List[Tuple[str, pd.DataFrame]]]:
    per_pred_options: List[List[Tuple[str, pd.DataFrame]]] = []
    for col in combo:
        s = df[col]
        if s.dtype == object:
            opts = categorical_options(s, y, args)
        else:
            opts = numeric_options(s, args)
        if not opts:
            return []
        per_pred_options.append(opts)
    grid_choices = list(product(*per_pred_options))
    return [list(choice) for choice in grid_choices]


# ----------------------------------- Main ------------------------------------

def main():
    args = parse_args()
    df = read_data(args.data, sheet=args.sheet)

    out_dir = args.output_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    memmap_dir = args.memmap_dir or out_dir
    os.makedirs(memmap_dir, exist_ok=True)

    mutually_exclusive = set(args.targets)

    for target in args.targets:
        if target not in df.columns:
            print(f"[WARN] Target '{target}' not found. Skipping Target '{target}'.")
            continue

        save_stdout(f"Now processing TARGET: {target}", args, cr_lf=True, title=None) 

        # Identify columns for this target
        cat_cols, num_cols = identify_column_data_types(df, target, mutually_exclusive)
        y_series = df[target].astype(float)
        y = y_series.values.astype(np.float64)
        predictors = cat_cols + num_cols

        # Build all base combinations (size >= 2)
        base_combos: List[Tuple[str, ...]] = []
        for r in range(2, len(predictors) + 1):
            base_combos.extend(combinations(predictors, r))

        # Optional cap for speed
        if args.max_combos_per_target is not None and args.max_combos_per_target < len(base_combos):
            base_combos = base_combos[: args.max_combos_per_target]

        start_time = time.time()
        total_expected_grids = _estimate_total_grids(df, base_combos, args)
        processed_grids = 0

        rows: List[Dict] = []
        combo_id = 0

        for combo in base_combos:
            combo_id += 1

            # Full transformation grid for this combo based on flags
            grid_choices = build_transform_grid_for_combo(df, combo, y_series, args)

            if not grid_choices:
                if args.verbose:
                    save_stdout(f"[INFO] Skipping combo {combo} due to zero options from flags.", args, cr_lf=True, title=None) 
                continue

            # Optional cap per combo
            if args.max_grids_per_combo is not None and args.max_grids_per_combo < len(grid_choices):
                grid_choices = grid_choices[: args.max_grids_per_combo]

            grid_id = 0
            for choice in grid_choices:
                grid_id += 1

                processed_grids += 1
                if args.verbose:
                    elapsed = time.time() - start_time
                    save_stdout(f"[{target}] progress: grid {processed_grids}/{total_expected_grids} | elapsed {elapsed:.1f}s", args, cr_lf=True, title=None) 

                parts: List[pd.DataFrame] = []
                part_names: List[List[str]] = []
                variant_labels = []
                for label, df_part in choice:
                    parts.append(df_part)
                    part_names.append(list(df_part.columns))
                    variant_labels.append(label)

                # Assemble memmap
                X_mm, colnames, tmp_path = create_memmap(parts, part_names, memmap_dir, dtype=np.float64)
                if X_mm is None or X_mm.shape[1] == 0:
                    r2_mean = r2_std = alpha_mean = l1_best = np.nan
                    n_cols = 0
                else:
                    n_cols = int(X_mm.shape[1])
                    # Optional standardization in-place
                    if args.standardize:
                        standardize_memmap_inplace(X_mm, block_size=args.block_size)

                    # Outer-CV ENet R^2
                    r2_mean, r2_std, alpha_mean, l1_best = cv_elasticnet_r2(
                        X_mm, y,
                        outer_folds=args.cv_folds,
                        inner_folds=args.enet_cv_folds,
                        l1_ratio_grid=args.l1_ratio_grid,
                        alphas_grid=args.alphas_grid,
                        random_state=args.random_state,
                    )

                # Cleanup temp memmap file
                try:
                    del X_mm
                except Exception:
                    pass
                try:
                    if tmp_path:
                        os.unlink(tmp_path)
                except Exception:
                    pass

                # Record
                rows.append(
                    {
                        "target": target,
                        "combo_id": combo_id,
                        "grid_id": grid_id,
                        "base_columns": "|".join(combo),
                        "variant": " | ".join(variant_labels),
                        "n_rows": int(len(y)),
                        "n_cols": int(n_cols),
                        "r2_mean": r2_mean,
                        "r2_std": r2_std,
                        "alpha": alpha_mean,
                        "l1_ratio": l1_best,
                    }
                )

                if args.verbose:
                    save_stdout(f"[{target}] combo={combo_id} grid={grid_id} cols={n_cols} R2(mean±std)={r2_mean:.4f}±{r2_std:.4f}  alpha≈{alpha_mean:.4g}  l1≈{l1_best:.2f}", args, cr_lf=True, title=None)

        # Save output to a CSV file
        result_df = pd.DataFrame(rows).sort_values(["combo_id", "grid_id"], ascending=[True, True]).reset_index(drop=True)
        out_path = f"{out_dir}/correlations_{chk_name(target)}_GRID_MEMMAP_ENET_R2.csv"
        result_df.to_csv(out_path, index=False)
        save_stdout(f"[OK] Wrote {len(result_df):,} rows for target '{target}' -> {out_path}", args, cr_lf=True, title=None) 


if __name__ == "__main__":
    main()
