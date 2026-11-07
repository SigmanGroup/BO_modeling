# Nested Cross-Validation Workflows for Regression and Classification

Created by: Pauline Bianchi (May 1, 2026)  
Edited by: Samantha Sloane and Pauline Bianchi  
Last update: November 6, 2026  
Sigman Lab, University of Utah, Version 1.1

This repository contains two parallel data science workflows for chemical datasets:

- **Regression:** `bo_regression`
- **Classification:** `bo_classification`

Both workflows use repeated nested cross-validation, Bayesian hyperparameter optimization with Optuna, leakage-safe feature filtering, feature selection, model comparison, out-of-fold evaluation, full-training ensemble refitting, descriptor exploration, optional SHAP interpretation, virtual screening, and external validation.

The classification workflow additionally optimizes the thresholds used to convert a continuous response into ordered classes.

---

## 1. Repository structure

```text
.
├── bo_regression.ipynb
├── bo_regression_functions.py
├── bo_regression.py
├── bo_regression.slurm
├── bo_classification.ipynb
├── bo_classification_functions.py
├── bo_classification.py
├── bo_classification.slurm
├── bo_env.yml
└── README.md
```

### Main workflow modules

- `bo_regression_functions.py` contains the reusable regression functions.
- `bo_classification_functions.py` contains the reusable classification functions.

Each module is organized into 12 main sections covering data loading, preprocessing, nested cross-validation, result aggregation, plotting, external prediction, SHAP, and descriptor exploration.

### Notebook interfaces

- `bo_regression.ipynb`
- `bo_classification.ipynb`

The notebooks provide an interactive interface for configuration, training, plotting, descriptor exploration, SHAP analysis, virtual screening, and external validation. They also include an optional quick model check for preliminary model ranking and response-distribution inspection; this notebook-only check can be skipped without affecting the full workflow. Some interactive visualizations are notebook-only conveniences.

### Command-line interfaces

- `bo_regression.py`
- `bo_classification.py`

The scripts call the same workflow functions as the notebooks and are designed for non-interactive execution on local workstations or high-performance computing clusters.

### SLURM examples

- `bo_regression.slurm`
- `bo_classification.slurm`

These files are templates and must be adapted to the local cluster configuration, conda installation, project directory, and dataset paths.

---

## 2. Installation

A common conda environment is provided for both workflows.

```bash
conda env create -f bo_env.yml
conda activate bo_env
```

The environment uses a Python 3.11 / NumPy 1.26-compatible stack for scikit-learn, XGBoost, Optuna, SHAP, pandas, and plotting libraries.

---

## 3. Quick start

Create and activate the environment:

```bash
conda env create -f bo_env.yml
conda activate bo_env
```

Then either:

- open `bo_regression.ipynb` or `bo_classification.ipynb` for interactive use; or
- run `bo_regression.py` or `bo_classification.py` from the command line.

For a first test, use the supplied example dataset with a small number of folds and Optuna trials. Results are written to the directory specified by `results_dir` in the notebooks or `--results-dir` from the command line.

---

## 4. Core methodology


Both workflows follow the same general architecture:

1. Load and validate the descriptor dataset.
2. Create outer cross-validation splits.
3. Within each outer training fold:
   - remove highly correlated descriptors;
   - standardize descriptors;
   - optimize feature selection;
   - optimize model hyperparameters with Optuna;
   - evaluate candidate pipelines using inner cross-validation.
4. Refit the selected pipeline on the complete outer training fold.
5. Evaluate it on the untouched outer validation fold.
6. Aggregate out-of-fold predictions and metrics across all repeated outer folds.
7. Build an outer-fold ensemble for cross-validation diagnostics.
8. Refit the selected outer-fold pipelines on the complete training dataset to create a full-training ensemble.
9. Use the full-training ensemble for full-dataset visualization, SHAP interpretation, external validation, and virtual screening.
10. Export numerical results and diagnostic figures.

The workflows retain two distinct ensemble objects:

- `results["models_dict"]`: models fitted on the outer training folds. These are used for nested-CV evaluation and out-of-fold predictions.
- `results["full_models_dict"]`: corresponding pipelines refitted on the complete training dataset. These are used for deployment-oriented tasks such as SHAP, external prediction, and virtual screening.

This separation prevents the full-training refit from being confused with unbiased out-of-fold evaluation.

Optionally, the one-standard-error rule can select the smallest feature set whose mean inner-validation score is within one standard error of the best candidate.

### Selection of the `best` model

The model selected through `best` follows the value of `optimize_metric`.

For regression:

- `r2`: highest out-of-fold R²;
- `rmse`: lowest out-of-fold RMSE;
- `mae`: lowest out-of-fold MAE.

For classification:

- `f1_macro`: highest macro F1 score;
- `balanced_accuracy`: highest balanced accuracy;
- `accuracy`: highest accuracy;
- `mcc`: highest Matthews correlation coefficient.

The same criterion is therefore used consistently for Optuna optimization and final model-family ranking.

---

## 5. Workflows

### 4.1. Regression

#### Supported models

- Multiple Linear Regression (`MLR`)
- Ridge regression (`Ridge`)
- Lasso regression (`Lasso`)
- Bayesian Ridge Regression (`BRR`)
- Kernel Ridge Regression (`KRR`)
- Support Vector Regression (`SVR`)
- K-Nearest Neighbors (`KNN`)
- Random Forest (`RF`)
- XGBoost (`XGB`)
- Neural network (`NN`)

#### Main regression metrics

- R²
- Root mean squared error (RMSE)
- Mean absolute error (MAE)

#### Supported outer split strategies

- random repeated K-fold
- stratification using response-value bins
- response-equidistant splitting
- group-based splitting
- leave-group-out validation
- predefined splitting
- Kennard–Stone holdout

### 4.2. Classification

The classification workflow starts from a continuous response and converts it into ordered classes.

#### Supported models

- Logistic Regression (`LogReg`)
- Support Vector Classifier (`SVC`)
- K-Nearest Neighbors (`KNN`)
- Random Forest (`RF`)
- Histogram Gradient Boosting (`HistGB`)
- XGBoost (`XGB`)
- Neural network (`NN`)

#### Main classification metrics

- accuracy
- balanced accuracy
- macro F1 score
- Matthews correlation coefficient
- ROC AUC, when available

#### Supported outer split strategies

- random repeated K-fold
- stratification using response-value bins
- response-equidistant splitting
- group-based splitting
- leave-group-out validation
- predefined splitting
- Kennard–Stone holdout

#### Threshold optimization

Thresholds are optimized exclusively within the inner cross-validation of each outer training fold.

Two modes are available:

- `quantiles`: candidate quantile positions are converted to numerical thresholds from the current inner-training fold;
- `fixed-grid`: numerical threshold candidates are supplied directly.

This avoids using outer validation observations to define class boundaries.

Each final full-training ensemble member retains the threshold configuration selected in its corresponding outer fold while its pipeline is refitted on the complete training dataset.

---

## 6. Use

### 6.1. Notebooks

Open one of the following:

```text
bo_regression.ipynb
bo_classification.ipynb
```

The optional quick model check may be run first for a lightweight preliminary assessment. For the full workflow, edit the main configuration cell and run the remaining notebook sequentially. The quick check can be skipped.

Typical parameters include:

- input workbook and sheet;
- target column;
- identifier, metadata, and dropped columns;
- model list;
- outer and inner cross-validation settings;
- Optuna trial count;
- Scoring metric for the optimization;
- feature-selection bounds;
- correlation threshold;
- RFE step size (`rfe_step`): values between 0 and 1 remove a fraction of the remaining features per step, while values of 1 or greater are treated as an integer number of features;
- parallel outer folds (`n_jobs`);
- optional one-standard-error selection (`one_se`);
- numerical precision (`round_digits`);
- output directory;
- plotting and SHAP options.

### 6.2. Command-line execution

The runner and function module must be in the same directory:

```text
bo_regression.py
bo_regression_functions.py
```

or:

```text
bo_classification.py
bo_classification_functions.py
```

The command blocks below are example configurations rather than a statement of every runner default. Adjust model lists, cross-validation settings, and optional analyses for the intended run.

#### Recommended regression example

```bash
python bo_regression.py \
    --excel-file "$EXCEL_FILE" \
    --sheet-name Model \
    --target-column "ΔG‡ (kcal/mol)" \
    --id-columns Nitroso \
    --drop-columns Nitroso "G(Nu) (Ha)" "G(nitroso) (Ha)" "G(TS) (Ha)" "G(TS_true) (Ha)" \
    --results-dir Results/workflow_example_260727_regression \
    --output-prefix "$PREFIX" \
    --outer-n-splits 2 \
    --outer-n-repeats 2 \
    --inner-cv-splits 2 \
    --inner-cv-repeats 2 \
    --n-trials 5 \
    --n-jobs 1 \
    --rfe-step 0.1 \
    --one-se \
    --round-digits 15 \
    --outer-split-strategy random \
    --inner-split-strategy random \
    --min-features 4 \
    --max-features-cap 10 \
    --corr-threshold 0.75 \
    --overfit-penalty-weight 1.0 \
    --optimize-metric r2 \
    --random-state 42 \
    --models MLR Ridge Lasso \
    --plot-response-distribution \
    --response-distribution-bins 15 \
    --plot-correlation-matrix \
    --plot-split-distribution \
    --split-distribution-density \
    --plot-model-comparisons \
    --plot-train-test-gap \
    --plot-full-training-ensemble \
    --figure-format svg \
    --figure-dpi 300
```

#### Recommended classification example

```bash
python bo_classification.py \
    --excel-file "$EXCEL_FILE" \
    --sheet-name Model \
    --target-column "ΔG‡ (kcal/mol)" \
    --id-columns Nitroso \
    --drop-columns Nitroso "G(Nu) (Ha)" "G(nitroso) (Ha)" "G(TS) (Ha)" "G(TS_true) (Ha)" \
    --results-dir Results/workflow_example_260727_classification \
    --output-prefix "$PREFIX" \
    --n-classes 2 \
    --threshold-mode quantiles \
    --threshold-quantiles 0.25 0.50 0.75 \
    --min-class-count 2 \
    --min-quantile-gap 0.1 \
    --outer-n-splits 2 \
    --outer-n-repeats 2 \
    --inner-cv-splits 2 \
    --inner-cv-repeats 2 \
    --n-trials 5 \
    --n-jobs 1 \
    --rfe-step 0.1 \
    --one-se \
    --round-digits 15 \
    --outer-split-strategy random \
    --inner-split-strategy random \
    --min-features 4 \
    --max-features-cap 10 \
    --corr-threshold 0.75 \
    --overfit-penalty-weight 1.0 \
    --optimize-metric f1_macro \
    --random-state 42 \
    --models LogReg RF XGB \
    --plot-response-distribution \
    --response-distribution-bins 15 \
    --plot-correlation-matrix \
    --plot-split-distribution \
    --split-distribution-density \
    --plot-model-comparisons \
    --plot-train-test-gap \
    --plot-oof-feature-maps \
    --oof-feature-map-models best \
    --plot-confusion-matrix \
    --plot-thresholds \
    --feature-map-top-n 3 \
    --figure-format svg \
    --figure-dpi 300
```

Confusion matrices and threshold-distribution plots are optional in the classification runner and can be enabled with `--plot-confusion-matrix` and `--plot-thresholds`, respectively.

### 6.3. SLURM execution

Adapt the paths and cluster directives in the provided templates, then submit with:

```bash
sbatch bo_regression.slurm
sbatch bo_classification.slurm
```

The SLURM files do not need to list every parser option. Options that are omitted use their defaults. The templates should explicitly include the parameters that define the intended run and the analyses that should be enabled. Adjust the `#SBATCH --array` range to the number of lines in `excel_files.txt`; paths in that file are interpreted relative to the directory from which the job is submitted.

---

## 7. Outputs

Depending on the enabled options, each run can generate Excel workbooks containing:

- configuration;
- summary metrics;
- fold-level metrics;
- out-of-fold predictions, including per-sample prediction standard deviation for repeated regression OOF predictions;
- full-training ensemble predictions, including per-sample prediction standard deviation for regression;
- optimized hyperparameters;
- selected features;
- feature-selection frequencies;
- split assignments;
- threshold summaries for classification;
- external predictions, including regression ensemble prediction uncertainty when enabled;
- SHAP values and global importance.

Figures may include:

- response-variable distributions;
- correlation heatmaps;
- target split distributions;
- regression parity plots with per-point ±1 SD uncertainty and 95% confidence bands around the regression fit;
- model comparisons;
- train-validation gaps;
- full-training ensemble parity plots;
- feature-selection frequencies;
- feature-target relationships;
- feature-pair plots;
- confusion matrices;
- threshold stability plots;
- classification feature maps showing class thresholds on the response axis and corresponding feature thresholds on the descriptor axis;
- SHAP beeswarm and importance plots.

---

## 8. Specifications

### 8.1. Reproducibility

Random seeds are controlled through:

```bash
--random-state 42
```

Optuna uses seeded TPE samplers. Numerical inputs are rounded to a fixed number of decimal places before modeling (`round_digits`, default: 15) to reduce sensitivity to platform-dependent last-digit differences. Outer folds can be parallelized with `n_jobs`; Optuna trials remain sequential.

Exact results may still depend on:

- package versions;
- operating system;
- cluster hardware;
- the internal behavior of multithreaded numerical libraries.

#### Small datasets

Small datasets are particularly sensitive to small numerical differences and to the exact cross-validation split. For datasets with fewer than approximately 20 observations, it is recommended to use at least 5 outer repeats to assess the stability of the reported performance and predictions. Repeated inner cross-validation is also important for model and feature-selection stability: at least 2 inner repeats are recommended when the dataset size permits it. Increasing the number of Optuna trials is not a substitute for repeated outer validation: on very small datasets, a larger hyperparameter search can increase the risk of overfitting the inner cross-validation splits.

### 8.2. Data requirements

The input workbook must contain:

- one continuous response column;
- numeric descriptor columns;
- optional identifier columns;
- optional metadata columns used for grouped or predefined splitting.

Rows containing missing values in the target or descriptor matrix are removed during loading.

For external and screening datasets, all training descriptor columns must be present and numeric.

---

## 9. License

The code in this repository is licensed under the MIT License. You are free to use, modify, and distribute the code under the terms of this license.

Copyright (c) 2026 Sigman Lab, University of Utah

---

## 10. Citation and contact

If you use or adapt these workflows, please cite the repository and the associated publication when available.

For questions, bug reports, or reproducibility issues, please use the repository issue tracker.
