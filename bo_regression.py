#!/usr/bin/env python3
# coding: utf-8

"""CHPC/SLURM entry point for bo_regression.

This file only converts command-line arguments into the same function calls used
by bo_regression.ipynb. The modeling code remains in bo_regression_functions.py,
which must be in the same directory or available on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from bo_regression_functions import (
    combine_training_and_external_predictions,
    compute_ensemble_shap,
    compute_feature_target_correlations,
    evaluate_external_dataset,
    export_workflow_results,
    load_regression_data,
    make_feature_pairs,
    plot_correlation_heatmap,
    plot_ensemble_shap,
    plot_external_overlay,
    plot_feature_frequency,
    plot_feature_histograms,
    plot_feature_pairs,
    plot_feature_target_correlations,
    plot_feature_target_scatter,
    plot_full_ensemble_parity,
    plot_model_metric_comparison,
    plot_model_prediction_comparison,
    plot_oof_parity,
    plot_overfitting_gap,
    plot_selector_frequency,
    plot_target_distribution,
    plot_target_split_distribution,
    plot_virtual_screening_distribution,
    run_nested_regression,
    save_ensemble_shap_results,
    save_external_results,
)


def parse_args():
    parser = argparse.ArgumentParser( description="Run the same non-interactive workflow as bo_regression.ipynb.", formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument("--excel-file", required=True)
    parser.add_argument("--sheet-name", default="training")
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--metadata-columns", nargs="*", default=[])
    parser.add_argument("--id-columns", nargs="*", default=[])
    parser.add_argument("--drop-columns", nargs="*", default=[])
    parser.add_argument("--results-dir", default="regression_results")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--outer-n-splits", type=int, default=3)
    parser.add_argument("--outer-n-repeats", type=int, default=3)
    parser.add_argument("--inner-cv-splits", type=int, default=3)
    parser.add_argument("--inner-cv-repeats", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel outer-fold workers; use 1 for serial execution.")
    parser.add_argument("--rfe-step", type=float, default=1.0, help="RFE step: integer-like 1 removes one feature; fractions such as 0.1 remove 10%% per step.")
    parser.add_argument("--one-se", action="store_true", help="Use the one-standard-error rule to prefer fewer descriptors with comparable inner-CV score.")
    parser.add_argument("--round-digits", type=int, choices=[10, 15], default=15, help="Round numeric input data to a fixed precision for reproducibility.")
    parser.add_argument(
        "--outer-split-strategy",
        choices=[
            "random", "y_stratified", "y_equidistant", "group",
            "leave_group_out", "predefined", "ks_holdout",
        ],
        default="random",
    )
    parser.add_argument( "--inner-split-strategy", choices=["random", "y_stratified", "y_equidistant", "group"], default="random", )
    parser.add_argument("--y-stratify-bins", type=int, default=3)
    parser.add_argument("--ks-test-fraction", type=float, default=0.20)
    parser.add_argument("--split-group-columns", nargs="*", default=[])
    parser.add_argument("--held-out-groups", nargs="*", default=[])
    parser.add_argument("--predefined-split-column", default=None)
    parser.add_argument("--predefined-train-label", default="train")
    parser.add_argument("--min-features", type=int, default=1)
    parser.add_argument("--max-features-cap", type=int, default=10)
    parser.add_argument("--corr-threshold", type=float, default=0.75)
    parser.add_argument("--overfit-penalty-weight", type=float, default=1.0)
    parser.add_argument( "--optimize-metric", choices=["r2", "rmse", "mae"], default="r2", )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument( "--models", nargs="+", default=["MLR", "Ridge", "Lasso", "BRR", "KRR", "SVR", "KNN", "RF", "XGB"], )
    parser.add_argument("--plot-correlation-matrix", action="store_true")
    parser.add_argument("--plot-response-distribution", action="store_true")
    parser.add_argument("--response-distribution-bins", type=int, default=15)
    parser.add_argument("--plot-split-distribution", action="store_true")
    parser.add_argument("--plot-model-comparisons", action="store_true")
    parser.add_argument("--plot-train-test-gap", action="store_true")
    parser.add_argument("--plot-pairwise-model-comparison", action="store_true")
    parser.add_argument("--comparison-model-x", default="Ridge")
    parser.add_argument("--comparison-model-y", default="MLR")
    parser.add_argument("--plot-full-training-ensemble", action="store_true")
    parser.add_argument("--full-ensemble-errorbars", action="store_true")
    parser.add_argument("--plot-feature-frequencies", action="store_true")
    parser.add_argument("--plot-selector-frequencies", action="store_true")
    parser.add_argument("--plot-feature-histograms", action="store_true")
    parser.add_argument("--plot-feature-target-correlations", action="store_true")
    parser.add_argument("--plot-feature-target-scatter", action="store_true")
    parser.add_argument("--plot-feature-pairs", action="store_true")
    parser.add_argument("--feature-frequency-top-n", type=int, default=15)
    parser.add_argument("--feature-histogram-bins", type=int, default=15)
    parser.add_argument("--feature-histogram-top-n", type=int, default=10)
    parser.add_argument( "--feature-target-method", choices=["pearson", "spearman"], default="pearson", )
    parser.add_argument("--feature-target-top-n", type=int, default=7)
    parser.add_argument( "--feature-pair-mode", choices=["representative", "correlation"], default="representative", )
    parser.add_argument("--feature-pair-model", default="best")
    parser.add_argument("--feature-pair-top-n", type=int, default=7)
    parser.add_argument("--feature-pair-max-pairs", type=int, default=10)
    parser.add_argument("--feature-pair-adjacent-only", action="store_true")
    parser.add_argument("--split-distribution-fold", type=int, default=1)
    parser.add_argument("--split-distribution-bins", type=int, default=15)
    parser.add_argument("--split-distribution-density", action="store_true")
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--shap-models", nargs="*", default=["best"])
    parser.add_argument("--shap-background-size", type=int, default=20)
    parser.add_argument("--shap-max-display", type=int, default=10)
    parser.add_argument("--screening-file", default=None)
    parser.add_argument("--screening-sheet", default="Sheet1")
    parser.add_argument("--screening-models", nargs="*", default=["best"])
    parser.add_argument("--plot-screening-distribution", action="store_true")
    parser.add_argument("--screening-histogram-bins", type=int, default=20)
    parser.add_argument("--external-file", default=None)
    parser.add_argument("--external-sheet", default="Sheet1")
    parser.add_argument("--external-models", nargs="*", default=["all"])
    parser.add_argument( "--figure-format", choices=["svg", "png", "pdf"], default="svg", )
    parser.add_argument("--figure-dpi", type=int, default=300)
    return parser.parse_args()


def resolve_models(specification, results):
    if specification == ["best"]:
        return [results["summary_mean"].iloc[0]["Model"]]
    if specification == ["all"]:
        return list(results["models_dict"])
    return specification


def main() -> None:
    args = parse_args()
    if args.outer_n_splits < 2 or args.inner_cv_splits < 2:
        raise ValueError("Outer and inner split counts must be at least 2.")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be at least 1.")
    if not 0 < args.corr_threshold <= 1:
        raise ValueError("--corr-threshold must be in (0, 1].")

    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    excel_file = Path(args.excel_file).expanduser().resolve()
    if not excel_file.is_file():
        raise FileNotFoundError(f"Excel file not found: {excel_file}")
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_prefix = args.output_prefix or excel_file.stem
    results_dir.mkdir(parents=True, exist_ok=True)

    metadata_columns = list(dict.fromkeys( args.metadata_columns + args.split_group_columns + ([args.predefined_split_column] if args.predefined_split_column else []) ))

    plot_settings = {
        "font_size": 8,
        "title_size": 9,
        "label_size": 8,
        "tick_size": 8,
        "legend_size": 8,
        "parity_figure_size": (3, 3),
        "bar_figure_size": (3, 3),
        "point_size": 60,
        "point_alpha": 1.0,
        "line_width": 1.0,
        "dpi": args.figure_dpi,
        "file_format": args.figure_format,
        "show_grid": False,
        "grid_alpha": 0.25,
        "equal_aspect": True,
        "oof_color": "#298c8c",
        "external_color": "#a00000",
        "full_distribution_color": "#298c8c",
        "training_color": "#f1a226",
        "validation_color": "#800074",
        "test_color": "#a00000",
        "bar_color": "#9fc8c8",
        "bar_edge_color": "black",
        "bar_edge_width": 1.0,
        "reference_line_color": "black",
        "point_edge_color": "#000000",
        "point_edge_width": 1.0,
    }

    print(f"Input Excel: {excel_file}", flush=True)
    print(f"Results directory: {results_dir}", flush=True)

    X, y, metadata = load_regression_data(
        excel_file=excel_file,
        sheet_name=args.sheet_name,
        target_column=args.target_column,
        id_columns=args.id_columns,
        drop_columns=args.drop_columns,
        metadata_columns=metadata_columns,
        round_digits=args.round_digits,
    )
    print( f"Loaded dataset: {X.shape[0]} rows, {X.shape[1]} features.", flush=True, )

    if args.plot_response_distribution:
        plot_target_distribution(
            y=y,
            target_label=args.target_column,
            results_dir=results_dir,
            output_prefix=output_prefix,
            bins=args.response_distribution_bins,
            save_figures=True,
            show_figures=False,
            plot_settings=plot_settings,
        )

    if args.plot_correlation_matrix:
        plot_correlation_heatmap( X=X, results_dir=results_dir, output_prefix=output_prefix, save_figures=True, show_figures=False, plot_settings=plot_settings, )

    results = run_nested_regression(
        X=X,
        y=y,
        metadata=metadata,
        model_names=args.models,
        outer_n_splits=args.outer_n_splits,
        outer_n_repeats=args.outer_n_repeats,
        inner_cv_splits=args.inner_cv_splits,
        inner_cv_repeats=args.inner_cv_repeats,
        min_features=args.min_features,
        max_features_cap=args.max_features_cap,
        corr_threshold=args.corr_threshold,
        n_trials=args.n_trials,
        overfit_penalty_weight=args.overfit_penalty_weight,
        optimize_metric=args.optimize_metric,
        random_state=args.random_state,
        verbose=True,
        outer_split_strategy=args.outer_split_strategy,
        inner_split_strategy=args.inner_split_strategy,
        split_group_columns=args.split_group_columns,
        y_stratify_bins=args.y_stratify_bins,
        ks_test_fraction=args.ks_test_fraction,
        held_out_groups=args.held_out_groups,
        predefined_split_column=args.predefined_split_column,
        predefined_train_label=args.predefined_train_label,
        export_split_indices=False,
        export_test_groups=False,
        max_test_groups_to_export=3,
        rfe_step=args.rfe_step,
        n_jobs=args.n_jobs,
        one_se=args.one_se,
    )

    configuration = vars(args).copy()
    configuration["excel_file"] = str(excel_file)
    configuration["results_dir"] = str(results_dir)

    export_workflow_results( results=results, output_file=results_dir / f"ML_results_{output_prefix}.xlsx", save_excel=True, configuration=configuration, )

    if args.plot_split_distribution:
        plot_target_split_distribution(
            y=y,
            split_assignments=results["split_assignments"],
            fold=args.split_distribution_fold,
            results_dir=results_dir,
            output_prefix=output_prefix,
            target_label=args.target_column,
            bins=args.split_distribution_bins,
            density=args.split_distribution_density,
            save_figures=True,
            show_figures=False,
            plot_settings=plot_settings,
        )

    plot_oof_parity(
        results=results,
        results_dir=results_dir,
        output_prefix=output_prefix,
        target_label=args.target_column,
        save_figures=True,
        show_figures=False,
        plot_settings=plot_settings,
    )

    if args.plot_model_comparisons:
        for metric in ["R2", "MAE", "RMSE"]:
            plot_model_metric_comparison(
                results=results,
                results_dir=results_dir,
                output_prefix=output_prefix,
                metric=metric,
                save_figures=True,
                show_figures=False,
                plot_settings=plot_settings,
            )

    if args.plot_train_test_gap:
        plot_overfitting_gap(
            results=results,
            results_dir=results_dir,
            output_prefix=output_prefix,
            metric=args.optimize_metric,
            save_figures=True,
            show_figures=False,
            plot_settings=plot_settings,
        )

    if args.plot_full_training_ensemble:
        plot_full_ensemble_parity(
            results=results,
            results_dir=results_dir,
            output_prefix=output_prefix,
            target_label=args.target_column,
            model_names=args.models,
            save_figures=True,
            show_figures=False,
            show_prediction_errorbars=args.full_ensemble_errorbars,
            plot_settings=plot_settings,
        )

    if args.plot_pairwise_model_comparison:
        plot_model_prediction_comparison(
            results=results,
            model_x=args.comparison_model_x,
            model_y=args.comparison_model_y,
            results_dir=results_dir,
            output_prefix=output_prefix,
            save_figures=True,
            show_figures=False,
            plot_settings=plot_settings,
        )

    if args.plot_selector_frequencies:
        plot_selector_frequency( results=results, results_dir=results_dir, output_prefix=output_prefix, save_figures=True, show_figures=False, plot_settings=plot_settings, )

    if args.plot_feature_frequencies:
        plot_feature_frequency(
            results=results,
            results_dir=results_dir,
            output_prefix=output_prefix,
            top_n=args.feature_frequency_top_n,
            save_figures=True,
            show_figures=False,
            plot_settings=plot_settings,
        )

    correlations = compute_feature_target_correlations( X=X, y=y, method=args.feature_target_method, )
    correlations.to_excel( results_dir / f"Feature_target_correlations_{output_prefix}.xlsx", index=False, )

    if args.plot_feature_histograms:
        features = correlations["Feature"].head( args.feature_histogram_top_n ).tolist()
        plot_feature_histograms(
            X=X,
            results_dir=results_dir,
            output_prefix=output_prefix,
            features=features,
            bins=args.feature_histogram_bins,
            save_figures=True,
            show_figures=False,
            plot_settings=plot_settings,
        )

    if args.plot_feature_target_correlations:
        plot_feature_target_correlations(
            correlation_table=correlations,
            results_dir=results_dir,
            output_prefix=output_prefix,
            top_n=args.feature_target_top_n,
            save_figures=True,
            show_figures=False,
            plot_settings=plot_settings,
        )

    if args.plot_feature_target_scatter:
        plot_feature_target_scatter(
            X=X,
            y=y,
            correlation_table=correlations,
            results_dir=results_dir,
            output_prefix=output_prefix,
            top_n=args.feature_target_top_n,
            save_figures=True,
            show_figures=False,
            plot_settings=plot_settings,
        )

    if args.plot_feature_pairs:
        if args.feature_pair_mode == "correlation":
            pair_features = correlations["Feature"].head( args.feature_pair_top_n ).tolist()
        else:
            model_name = ( results["summary_mean"].iloc[0]["Model"] if args.feature_pair_model == "best" else args.feature_pair_model )
            pair_features = results["representative_features"].get( model_name, [], )[:args.feature_pair_top_n]
        feature_pairs = make_feature_pairs( pair_features, args.feature_pair_max_pairs, args.feature_pair_adjacent_only, )
        if feature_pairs:
            plot_feature_pairs(
                X=X,
                y=y,
                feature_pairs=feature_pairs,
                results_dir=results_dir,
                output_prefix=output_prefix,
                bins=args.feature_histogram_bins,
                color_by_target=True,
                save_figures=True,
                show_figures=False,
                plot_settings=plot_settings,
            )

    if args.run_shap:
        shap_results = {}
        for model_name in resolve_models(args.shap_models, results):
            shap_result = compute_ensemble_shap(
                ensemble_model=results["full_models_dict"][model_name],
                X_background=X,
                X_explain=X.reset_index(drop=True),
                model_name=model_name,
                dataset_labels=["Training"] * len(X),
                sample_ids=list(range(len(X))),
                background_size=args.shap_background_size,
                max_evals=None,
                random_state=args.random_state,
            )
            shap_results[model_name] = shap_result
            plot_ensemble_shap(
                shap_result=shap_result,
                model_name=model_name,
                results_dir=results_dir,
                output_prefix=output_prefix,
                max_display=args.shap_max_display,
                save_figures=True,
                show_figures=False,
                plot_settings=plot_settings,
            )
        save_ensemble_shap_results( shap_results=shap_results, output_file=( results_dir / f"Ensemble_SHAP_train_{output_prefix}.xlsx" ), save_excel=True, )

    if args.screening_file:
        screening_data = pd.read_excel( args.screening_file, sheet_name=args.screening_sheet, )
        screening_models = resolve_models(args.screening_models, results)
        screening_predictions, _ = evaluate_external_dataset(
            df_external=screening_data,
            training_columns=X.columns,
            models_dict=results["full_models_dict"],
            target_column=args.target_column,
            id_columns=args.id_columns,
            drop_columns=args.drop_columns,
            model_names=screening_models,
            include_prediction_std=True,
        )
        screening_predictions.to_excel( results_dir / f"Virtual_screening_{output_prefix}.xlsx", index=False, )
        if args.plot_screening_distribution:
            plot_virtual_screening_distribution(
                y_training=y,
                screening_results=screening_predictions,
                model_names=screening_models,
                results_dir=results_dir,
                output_prefix=output_prefix,
                target_label=args.target_column,
                bins=args.screening_histogram_bins,
                save_figures=True,
                show_figures=False,
                plot_settings=plot_settings,
            )

    if args.external_file:
        external_data = pd.read_excel( args.external_file, sheet_name=args.external_sheet, )
        external_models = resolve_models(args.external_models, results)
        external_predictions, external_metrics = evaluate_external_dataset(
            df_external=external_data,
            training_columns=X.columns,
            models_dict=results["full_models_dict"],
            target_column=args.target_column,
            id_columns=args.id_columns,
            drop_columns=args.drop_columns,
            model_names=external_models,
            include_prediction_std=True,
        )
        combined_predictions = combine_training_and_external_predictions(
            results=results,
            external_results=external_predictions,
            target_column=args.target_column,
            model_names=external_models,
        )
        save_external_results(
            detailed_results=external_predictions,
            metrics=external_metrics,
            output_file=results_dir / f"External_test_{output_prefix}.xlsx",
            save_excel=True,
            combined_training_external=combined_predictions,
        )
        if not external_metrics.empty:
            plot_external_overlay(
                results=results,
                external_results=external_predictions,
                external_metrics=external_metrics,
                target_column=args.target_column,
                results_dir=results_dir,
                output_prefix=output_prefix,
                external_label="External test",
                model_names=external_models,
                save_figures=True,
                show_figures=False,
                plot_settings=plot_settings,
                training_reference="full_dataset",
            )

    print(f"Completed. Results saved in: {results_dir}", flush=True)


if __name__ == "__main__":
    main()
