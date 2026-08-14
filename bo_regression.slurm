#!/bin/csh
#SBATCH --job-name=regression_bo
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --account=sigman-np
#SBATCH --partition=sigman-np
#SBATCH --array=0-9%10
# Adjust the array range to match the number of lines in excel_files.txt.
#SBATCH -o logs/regression_bo.%A_%a.out
#SBATCH -e logs/regression_bo.%A_%a.err

module purge
setenv MPLBACKEND Agg

# Activate the same Python environment used for the notebook.
# Example:
# module load miniforge3
# conda activate bo_env

mkdir -p logs

set FILELIST = excel_files.txt
set LINE = `expr $SLURM_ARRAY_TASK_ID + 1`
set EXCEL_FILE = `sed -n "${LINE}p" "$FILELIST"`

if ( "$EXCEL_FILE" == "" ) then
    echo "No Excel file found for array index $SLURM_ARRAY_TASK_ID"
    exit 1
endif

if ( ! -e "$EXCEL_FILE" ) then
    echo "Excel file does not exist: $EXCEL_FILE"
    exit 1
endif

set PREFIX = `basename "$EXCEL_FILE" .xlsx`

echo "SLURM array index: $SLURM_ARRAY_TASK_ID"
echo "Excel file: $EXCEL_FILE"
echo "Output prefix: $PREFIX"

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
    --outer-split-strategy random \
    --inner-split-strategy random \
    --y-stratify-bins 3 \
    --ks-test-fraction 0.15 \
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
    --plot-pairwise-model-comparison \
    --comparison-model-x Ridge \
    --comparison-model-y MLR \
    --plot-full-training-ensemble \
    --full-ensemble-errorbars \
    --plot-feature-frequencies \
    --plot-selector-frequencies \
    --plot-feature-histograms \
    --plot-feature-target-correlations \
    --plot-feature-target-scatter \
    --plot-feature-pairs \
    --feature-frequency-top-n 5 \
    --feature-histogram-bins 4 \
    --feature-histogram-top-n 5 \
    --feature-target-method pearson \
    --feature-target-top-n 7 \
    --feature-pair-mode representative \
    --feature-pair-model best \
    --feature-pair-top-n 7 \
    --feature-pair-max-pairs 10 \
    --feature-pair-adjacent-only \
    --split-distribution-fold 1 \
    --split-distribution-bins 15 \
    --run-shap \
    --shap-models best \
    --shap-background-size 20 \
    --shap-max-display 10 \
    --screening-file "$EXCEL_FILE" \
    --screening-sheet Screening \
    --screening-models best \
    --plot-screening-distribution \
    --screening-histogram-bins 20 \
    --external-file "$EXCEL_FILE" \
    --external-sheet Test \
    --external-models all \
    --figure-format svg \
    --figure-dpi 300
