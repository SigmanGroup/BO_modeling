#!/usr/bin/env python3
# coding: utf-8

"""Nested cross-validation regression workflow with Optuna.

This module contains the complete regression workflow used in bo_regression (.ipynb or .py).
It deliberately keeps feature filtering, scaling, feature selection, 
hyperparameter optimization and model fitting inside each cross-validation
training fold to avoid data leakage.

Search for the section banners beginning with ``###`` to jump directly to:
1. Data loading and validation
2. Leakage-safe feature filtering
3. Model and feature-selector factories
4. Split-generation strategies
5. Inner optimization and outer-fold training
6. Repeated nested-CV orchestration
7. Result aggregation and Excel export
8. General plotting utilities
9. Model and split diagnostic plots
10. External-test and virtual-screening utilities
11. Ensemble SHAP interpretation
12. Descriptor exploration plots
"""

from __future__ import annotations

from pathlib import Path
import re
from joblib import Parallel, delayed, parallel_backend
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from scipy.stats import pearsonr, spearmanr, t

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

import warnings
import logging
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)

### 1. Data loading and validation ###
def safe_rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_regression( y_true: Iterable[float], y_pred: Iterable[float], ) -> dict[str, float]:
    """Return the common regression metrics used throughout the workflow."""
    return { "r2": float(r2_score(y_true, y_pred)), "rmse": safe_rmse(y_true, y_pred), "mae": float(mean_absolute_error(y_true, y_pred)), }


def regression_optimization_score( y_true: Iterable[float], y_pred: Iterable[float], metric: str, ) -> float:
    """Convert the selected regression metric to an Optuna maximization score."""
    metric = metric.lower()
    metrics = evaluate_regression(y_true, y_pred)
    if metric == "r2":
        return metrics["r2"]
    if metric == "rmse":
        return -metrics["rmse"]
    if metric == "mae":
        return -metrics["mae"]
    raise ValueError("optimize_metric must be one of: r2, rmse, mae.")


def load_regression_data(
    excel_file: str | Path,
    sheet_name: str,
    target_column: str,
    id_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
    metadata_columns: list[str] | None = None,
    round_digits: int = 15,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load numeric regression data and retain identifiers/split metadata."""
    id_columns = id_columns or []
    drop_columns = drop_columns or []
    metadata_columns = metadata_columns or []
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' was not found.")
    retained_metadata=[]
    for column in [*id_columns,*metadata_columns]:
        if not isinstance(column,str):
            raise TypeError(f"Metadata entries must be column-name strings, got {column!r}.")
        if column in df.columns and column not in retained_metadata:
            retained_metadata.append(column)
    metadata=df[retained_metadata].copy()
    df=df.drop(columns=drop_columns,errors='ignore').dropna(subset=[target_column])
    X=df.drop(columns=[target_column]+retained_metadata,errors='ignore').apply(pd.to_numeric,errors='coerce')
    y=pd.to_numeric(df[target_column],errors='coerce')
    valid=X.notna().all(axis=1)&y.notna()
    X=X.loc[valid].reset_index(drop=True)
    y=y.loc[valid].reset_index(drop=True)
    metadata=metadata.loc[valid].reset_index(drop=True)
    X = X.round(round_digits)
    y = y.round(round_digits)
    if X.empty: raise ValueError('No complete numeric rows remain after cleaning.')
    return X,y,metadata

def prepare_external_features(
    df_external: pd.DataFrame,
    training_columns: Iterable[str],
    target_column: str,
    id_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Create an external feature matrix with strict training-column validation."""
    id_columns = id_columns or []
    drop_columns = drop_columns or []
    training_columns = list(training_columns)

    X_external = df_external.drop( columns=[target_column] + id_columns + drop_columns, errors="ignore", )
    X_external = X_external.apply(pd.to_numeric, errors="coerce")

    missing_columns = [c for c in training_columns if c not in X_external.columns]
    if missing_columns:
        raise ValueError( f"External dataset is missing {len(missing_columns)} required features: " f"{missing_columns}" )

    X_external = X_external.loc[:, training_columns].reset_index(drop=True)
    nan_columns = X_external.columns[X_external.isna().any()].tolist()
    if nan_columns:
        raise ValueError( "External dataset contains missing or non-numeric values in: " f"{nan_columns}" )

    return X_external

### 2. Leakage-safe feature filtering ###
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove mutually correlated features inside each fitted training fold.
    Features are ranked by absolute univariate correlation with y. Starting from
    the highest-ranked feature, a candidate is retained only when its absolute
    correlation with every already-retained feature is below ``threshold``.
    Set threshold=None or threshold>=1 to disable filtering.
    """
    def __init__(self, threshold: float | None = 0.80):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None):
        X_df = self._to_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)

        if self.threshold is None or self.threshold >= 1:
            self.selected_features_ = list(X_df.columns)
            self.support_mask_ = np.ones(X_df.shape[1], dtype=bool)
            return self

        if not 0 < self.threshold < 1:
            raise ValueError("corr_threshold must be between 0 and 1, or None.")

        feature_corr = X_df.corr().abs().fillna(0.0)

        if y is None:
            ranked_features = list(X_df.columns)
        else:
            y_series = pd.Series(np.asarray(y), index=X_df.index)
            target_scores = X_df.apply(lambda col: col.corr(y_series)).abs().fillna(0.0)
            ranked_features = target_scores.sort_values(ascending=False).index.tolist()

        selected: list[str] = []
        for feature in ranked_features:
            if all(feature_corr.loc[feature, kept] < self.threshold for kept in selected):
                selected.append(feature)

        if not selected:
            selected = [ranked_features[0]]

        # Restore original column order for predictable downstream output.
        selected_set = set(selected)
        self.selected_features_ = [c for c in X_df.columns if c in selected_set]
        self.support_mask_ = np.asarray( [c in selected_set for c in X_df.columns], dtype=bool )
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        X_df = self._to_dataframe(X, columns=self.feature_names_in_)
        return X_df.loc[:, self.selected_features_]

    def get_support(self) -> np.ndarray:
        return self.support_mask_.copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.selected_features_, dtype=object)

    @staticmethod
    def _to_dataframe(X, columns=None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if columns is None:
            columns = [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
        return pd.DataFrame(X, columns=columns)

class SafeSelectKBest(SelectKBest):
    """SelectKBest that caps k at the number of available input features."""

    def fit(self, X, y):
        requested_k = self.k
        if requested_k != "all":
            self.k = min(int(requested_k), np.asarray(X).shape[1])
        try:
            return super().fit(X, y)
        finally:
            self.k = requested_k


class SafeRFE(RFE):
    """RFE that caps the requested feature count after correlation filtering."""

    def fit(self, X, y, **fit_params):
        requested_n = self.n_features_to_select
        if isinstance(requested_n, int):
            self.n_features_to_select = min(requested_n, np.asarray(X).shape[1])
        try:
            return super().fit(X, y, **fit_params)
        finally:
            self.n_features_to_select = requested_n

### 3. Model and feature-selector factories ###
def build_model_from_trial(model_name: str, trial: optuna.Trial, random_state: int = 42, n_jobs: int = 1):
    if model_name == "MLR":
        return LinearRegression()
    if model_name == "Ridge":
        return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True))
    if model_name == "Lasso":
        return Lasso( alpha=trial.suggest_float("alpha", 1e-3, 100, log=True), max_iter=50000, )
    if model_name == "BRR":
        return BayesianRidge(
            alpha_1=trial.suggest_float("alpha_1", 1e-8, 1e-2, log=True),
            alpha_2=trial.suggest_float("alpha_2", 1e-8, 1e-2, log=True),
            lambda_1=trial.suggest_float("lambda_1", 1e-8, 1e-2, log=True),
            lambda_2=trial.suggest_float("lambda_2", 1e-8, 1e-2, log=True),
        )
    if model_name == "KRR":
        return KernelRidge(
            alpha=trial.suggest_float("alpha", 1e-3, 100, log=True),
            gamma=trial.suggest_float("gamma", 1e-3, 100, log=True),
            kernel=trial.suggest_categorical("kernel", ["rbf", "laplacian"]),
        )
    if model_name == "SVR":
        return SVR(
            C=trial.suggest_float("C", 0.01, 10, log=True),
            epsilon=trial.suggest_float("epsilon", 0.05, 1, log=True),
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
            kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]),
        )
    if model_name == "KNN":
        return KNeighborsRegressor( n_neighbors=trial.suggest_int("n_neighbors", 1, 5), weights=trial.suggest_categorical("weights", ["uniform", "distance"]), )
    if model_name == "RF":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 20, 200),
            max_depth=trial.suggest_int("max_depth", 1, 5),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 8),
            min_samples_split=trial.suggest_int("min_samples_split", 4, 12),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if model_name == "XGB":
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=trial.suggest_int("n_estimators", 20, 200),
            max_depth=trial.suggest_int("max_depth", 1, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 100, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 20),
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if model_name == "NN":
        return MLPRegressor(
            hidden_layer_sizes=trial.suggest_categorical(
                "hidden_layer_sizes", [(5,), (10,), (5, 5), (10, 5)]
            ),
            activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
            alpha=trial.suggest_float("alpha", 1e-3, 1, log=True),
            learning_rate_init=trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
            early_stopping=True,
            validation_fraction=0.05,
            n_iter_no_change=30,
            max_iter=3000,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model: {model_name}")


def remove_feature_selection_params(params: dict[str, Any]) -> dict[str, Any]:
    fs_params = {
        "feature_selector",
        "selectkbest_k",
        "alpha_feature",
        "lasso_threshold_feature",
        "rf_threshold_feature",
        "rf_n_estimators_feature",
        "rf_max_depth_feature",
        "rfe_n_features",
        "rfe_alpha",
    }
    return {k: v for k, v in params.items() if k not in fs_params}


def build_final_model_from_params(model_name: str, params: dict[str, Any], random_state: int = 42, n_jobs: int = 1):
    params = remove_feature_selection_params(dict(params))
    if model_name == "MLR":
        return LinearRegression()
    if model_name == "Ridge":
        return Ridge(**params)
    if model_name == "Lasso":
        return Lasso(**params, max_iter=50000)
    if model_name == "BRR":
        return BayesianRidge(**params)
    if model_name == "KRR":
        return KernelRidge(**params)
    if model_name == "SVR":
        return SVR(**params)
    if model_name == "KNN":
        return KNeighborsRegressor(**params)
    if model_name == "RF":
        return RandomForestRegressor(**params, random_state=random_state, n_jobs=n_jobs)
    if model_name == "XGB":
        return XGBRegressor( **params, objective="reg:squarederror", random_state=random_state, n_jobs=n_jobs, )
    if model_name == "NN":
        return MLPRegressor( **params, early_stopping=True, validation_fraction=0.2, n_iter_no_change=30, max_iter=3000, random_state=random_state, )
    raise ValueError(f"Unknown model: {model_name}")


def build_feature_selector_from_trial(trial: optuna.Trial, max_features_cap: int, min_features: int = 1, random_state: int = 42, rfe_step: int | float = 1, n_jobs: int = 1):
    min_features = max(1, min(int(min_features), int(max_features_cap)))
    selector_name = trial.suggest_categorical( "feature_selector", ["selectkbest", "lasso_feature", "rf_feature", "rfe_feature"], )

    if selector_name == "selectkbest":
        selector = SafeSelectKBest( score_func=f_regression, k=trial.suggest_int("selectkbest_k", min_features, max_features_cap), )
    elif selector_name == "lasso_feature":
        selector = SelectFromModel(
            estimator=Lasso(
                alpha=trial.suggest_float("alpha_feature", 1e-3, 100, log=True),
                max_iter=50000,
            ),
            threshold=trial.suggest_categorical(
                "lasso_threshold_feature", ["mean", "median"]
            ),
            max_features=max_features_cap,
        )
    elif selector_name == "rf_feature":
        selector = SelectFromModel(
            estimator=RandomForestRegressor(
                n_estimators=trial.suggest_int("rf_n_estimators_feature", 20, 200),
                max_depth=trial.suggest_int("rf_max_depth_feature", 1, 5),
                random_state=random_state,
                n_jobs=n_jobs,
            ),
            threshold=trial.suggest_categorical(
                "rf_threshold_feature", ["mean", "median"]
            ),
            max_features=max_features_cap,
        )
    elif selector_name == "rfe_feature":
        selector = SafeRFE(
            estimator=Ridge(
                alpha=trial.suggest_float("rfe_alpha", 1e-3, 100, log=True)
            ),
            n_features_to_select=trial.suggest_int(
                "rfe_n_features", min_features, max_features_cap
            ),
            step=rfe_step,
        )
    else:
        raise ValueError(f"Unknown feature selector: {selector_name}")

    return selector, selector_name


def get_selected_feature_names(fitted_pipe: Pipeline) -> list[str]:
    names = fitted_pipe.named_steps["correlation_filter"].get_feature_names_out()
    selector = fitted_pipe.named_steps["selector"]
    if hasattr(selector, "get_support"):
        names = np.asarray(names)[selector.get_support()]
    return list(names)

### 4. Split-generation strategies ###
def _make_y_bins( y: pd.Series, requested_bins: int, n_splits: int, ) -> np.ndarray:
    """Create quantile bins with enough observations for stratified folds."""
    max_bins = min(int(requested_bins), int(pd.Series(y).nunique()))

    for n_bins in range(max_bins, 1, -1):
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        counts = pd.Series(bins).value_counts()
        if len(counts) >= 2 and counts.min() >= n_splits:
            return np.asarray(bins, dtype=int)

    raise ValueError( "Could not create y-stratification bins with at least " f"{n_splits} observations per bin." )


def _repeated_group_splits( groups: Iterable[Any], n_splits: int, n_repeats: int, random_state: int, ):
    """Repeated group-wise folds with approximately balanced row counts."""
    groups = np.asarray(groups)
    unique_groups = np.asarray(pd.unique(groups))

    if len(unique_groups) < n_splits:
        raise ValueError( f"Group split requires at least {n_splits} unique groups; " f"found {len(unique_groups)}." )

    group_sizes = { group: int(np.sum(groups == group)) for group in unique_groups }
    rng = np.random.RandomState(random_state)

    for _ in range(n_repeats):
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)

        fold_groups = [[] for _ in range(n_splits)]
        fold_sizes = np.zeros(n_splits, dtype=int)

        for group in sorted( shuffled, key=lambda value: group_sizes[value], reverse=True, ):
            fold_index = int(np.argmin(fold_sizes))
            fold_groups[fold_index].append(group)
            fold_sizes[fold_index] += group_sizes[group]

        for test_groups in fold_groups:
            test_mask = np.isin(groups, test_groups)
            yield np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def _y_equidistant_splits( y: pd.Series, n_splits: int, n_repeats: int, random_state: int, ):
    """Distribute sorted y values across folds so every fold spans the y range."""
    y_values = np.asarray(y, dtype=float)
    rng = np.random.RandomState(random_state)

    for _ in range(n_repeats):
        jitter = rng.normal(0.0, 1e-12, size=len(y_values))
        ordered = np.argsort(y_values + jitter, kind="mergesort")

        fold_assignment = np.empty(len(y_values), dtype=int)
        for block_start in range(0, len(ordered), n_splits):
            block = ordered[block_start:block_start + n_splits]
            if (block_start // n_splits) % 2 == 1:
                block = block[::-1]
            for fold_index, row_index in enumerate(block):
                fold_assignment[row_index] = fold_index

        for fold_index in range(n_splits):
            test_index = np.flatnonzero(fold_assignment == fold_index)
            train_index = np.flatnonzero(fold_assignment != fold_index)
            yield train_index, test_index


def kennard_stone_indices( X: pd.DataFrame, test_fraction: float = 0.20, ) -> tuple[np.ndarray, np.ndarray]:
    """Create one Kennard–Stone holdout split using standardized X distances."""
    if not 0 < test_fraction < 1:
        raise ValueError("ks_test_fraction must be between 0 and 1.")

    values = np.asarray(X, dtype=float)
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales == 0] = 1.0
    standardized = (values - means) / scales

    n_samples = len(standardized)
    n_test = max(1, int(round(n_samples * test_fraction)))
    if n_test >= n_samples:
        raise ValueError("Kennard–Stone test set must be smaller than the dataset.")

    squared_norms = np.sum(standardized ** 2, axis=1)
    distances = ( squared_norms[:, None] + squared_norms[None, :] - 2 * standardized @ standardized.T )
    np.maximum(distances, 0, out=distances)

    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    selected = [int(first)]
    if second != first and len(selected) < n_test:
        selected.append(int(second))

    minimum_distance = np.min(distances[:, selected], axis=1)
    minimum_distance[selected] = -np.inf

    while len(selected) < n_test:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        minimum_distance = np.minimum( minimum_distance, distances[:, next_index], )
        minimum_distance[selected] = -np.inf

    test_index = np.asarray(sorted(selected), dtype=int)
    train_index = np.setdiff1d( np.arange(n_samples), test_index, assume_unique=True, )
    return train_index, test_index


def _predefined_splits( labels: Iterable[Any], train_label: Any = "train", ):
    """Build user-defined train/test splits from a metadata column."""
    labels = pd.Series(labels).reset_index(drop=True)
    if labels.isna().any():
        raise ValueError("The predefined split column contains missing values.")

    unique_labels = list(pd.unique(labels))
    normalized = labels.astype(str).str.lower()
    train_label_normalized = str(train_label).lower()

    if train_label_normalized in set(normalized):
        train_index = np.flatnonzero( normalized.to_numpy() == train_label_normalized )
        test_labels = [ label for label in unique_labels if str(label).lower() != train_label_normalized ]
        if not test_labels:
            raise ValueError( "The predefined split column contains a train label but no test label." )
        return [ ( train_index, np.flatnonzero(labels.to_numpy() == test_label), ) for test_label in test_labels ]

    if len(unique_labels) < 2:
        raise ValueError( "The predefined split column must contain at least two labels." )

    splits = []
    for test_label in unique_labels:
        test_mask = labels.to_numpy() == test_label
        splits.append( ( np.flatnonzero(~test_mask), np.flatnonzero(test_mask), ) )
    return splits


def make_regression_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    strategy: str = "random",
    groups: Iterable[Any] | None = None,
    y_stratify_bins: int = 5,
    ks_test_fraction: float = 0.20,
    held_out_groups: list[Any] | None = None,
    predefined_labels: Iterable[Any] | None = None,
    predefined_train_label: Any = "train",
):
    """Build configurable outer or inner regression splits."""
    strategy = strategy.lower()

    if strategy == "random":
        splitter = RepeatedKFold( n_splits=n_splits, n_repeats=n_repeats, random_state=random_state, )
        return list(splitter.split(X))

    if strategy == "y_stratified":
        y_bins = _make_y_bins(y, y_stratify_bins, n_splits)
        splitter = RepeatedStratifiedKFold( n_splits=n_splits, n_repeats=n_repeats, random_state=random_state, )
        return list(splitter.split(X, y_bins))

    if strategy == "y_equidistant":
        return list( _y_equidistant_splits( y=y, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state, ) )

    if strategy in {"group", "cluster"}:
        if groups is None:
            raise ValueError( "A group split requires group labels." )
        if pd.Series(groups).isna().any():
            raise ValueError("Group labels contain missing values.")
        return list( _repeated_group_splits( groups=groups, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state, ) )

    if strategy == "leave_group_out":
        if groups is None or not held_out_groups:
            raise ValueError( "leave_group_out requires split_group_column and held_out_groups." )
        group_series = pd.Series(groups).reset_index(drop=True)
        test_mask = group_series.isin(held_out_groups).to_numpy()
        if not test_mask.any():
            raise ValueError("None of the held_out_groups were found.")
        if test_mask.all():
            raise ValueError("held_out_groups contains every observation.")
        return [ ( np.flatnonzero(~test_mask), np.flatnonzero(test_mask), ) ]

    if strategy == "predefined":
        if predefined_labels is None:
            raise ValueError( "predefined split requires predefined_split_column." )
        return _predefined_splits( labels=predefined_labels, train_label=predefined_train_label, )

    if strategy in {"ks", "ks_holdout"}:
        return [ kennard_stone_indices( X=X, test_fraction=ks_test_fraction, ) ]

    raise ValueError( "Unknown split strategy. Choose: random, y_stratified, " "y_equidistant, group, leave_group_out, predefined, or ks_holdout." )


### 5. Inner optimization and outer-fold training ###
def train_outer_fold(
    model_name: str,
    X_train_outer: pd.DataFrame,
    y_train_outer: pd.Series,
    X_test_outer: pd.DataFrame,
    y_test_outer: pd.Series,
    inner_cv_splits: int,
    inner_cv_repeats: int,
    max_features_cap: int,
    min_features: int,
    corr_threshold: float | None,
    n_trials: int,
    overfit_penalty_weight: float,
    optimize_metric: str,
    random_state: int,
    inner_split_strategy: str = "random",
    inner_groups: Iterable[Any] | None = None,
    y_stratify_bins: int = 5,
    rfe_step: int | float = 1,
    one_se: bool = False,
    n_jobs: int = 1,
):
    # The inner splits are created only from the current outer-training set.
    # They are used exclusively for Optuna model/selector selection.
    inner_splits = make_regression_splits(
        X=X_train_outer,
        y=y_train_outer,
        n_splits=inner_cv_splits,
        n_repeats=inner_cv_repeats,
        random_state=random_state,
        strategy=inner_split_strategy,
        groups=inner_groups,
        y_stratify_bins=y_stratify_bins,
    )

    def objective(trial: optuna.Trial) -> float:
        # One Optuna trial defines both a feature-selection strategy and the
        # model hyperparameters. The complete pipeline is then refitted inside
        # every inner fold so preprocessing never sees validation observations.
        selector, _ = build_feature_selector_from_trial(trial, max_features_cap=max_features_cap, min_features=min_features, random_state=random_state, rfe_step=rfe_step, n_jobs=n_jobs)
        model = build_model_from_trial(model_name, trial, random_state=random_state, n_jobs=n_jobs)

        pipe = Pipeline( [ ("correlation_filter", CorrelationFilter(corr_threshold)), ("scaler", StandardScaler()), ("selector", selector), ("model", model), ] )

        train_scores = []
        validation_scores = []
        feature_counts = []

        for train_index, validation_index in inner_splits:
            X_train = X_train_outer.iloc[train_index]
            X_validation = X_train_outer.iloc[validation_index]
            y_train = y_train_outer.iloc[train_index]
            y_validation = y_train_outer.iloc[validation_index]

            try:
                pipe.fit(X_train, y_train)
                feature_counts.append(len(get_selected_feature_names(pipe)))
                train_scores.append( regression_optimization_score( y_train, pipe.predict(X_train), optimize_metric ) )
                validation_scores.append( regression_optimization_score( y_validation, pipe.predict(X_validation), optimize_metric ) )
            except Exception:
                return -1e9

        mean_train = float(np.nanmean(train_scores))
        mean_validation = float(np.nanmean(validation_scores))

        if np.isnan(mean_validation):
            return -1e9

        # Penalize trials when the inner-training score exceeds the inner-validation score; reported outer metrics are unchanged.
        overfit_penalty = max(0.0, mean_train - mean_validation)
        trial.set_user_attr("n_features", float(np.mean(feature_counts)))
        trial.set_user_attr("validation_mean", mean_validation)
        trial.set_user_attr("validation_se", float(np.std(validation_scores, ddof=1) / np.sqrt(len(validation_scores))) if len(validation_scores) > 1 else 0.0)
        return mean_validation - overfit_penalty_weight * overfit_penalty

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    # Optionally select the smallest model within one standard error of the best inner-validation score.
    chosen_trial = study.best_trial
    if one_se:
        valid_trials = [t for t in study.trials if "validation_mean" in t.user_attrs and "n_features" in t.user_attrs]
        if valid_trials:
            best_validation_trial = max(valid_trials, key=lambda t: t.user_attrs["validation_mean"])
            cutoff = best_validation_trial.user_attrs["validation_mean"] - best_validation_trial.user_attrs.get("validation_se", 0.0)
            eligible = [t for t in valid_trials if t.user_attrs["validation_mean"] >= cutoff]
            chosen_trial = min(eligible, key=lambda t: (t.user_attrs["n_features"], -t.user_attrs["validation_mean"]))
    best_params = chosen_trial.params
    best_selector_name = best_params["feature_selector"]
    fixed_trial = optuna.trial.FixedTrial(best_params)

    final_selector, _ = build_feature_selector_from_trial(fixed_trial, max_features_cap=max_features_cap, min_features=min_features, random_state=random_state, rfe_step=rfe_step, n_jobs=n_jobs)
    final_model = build_final_model_from_params(model_name, best_params, random_state=random_state, n_jobs=n_jobs)

    final_pipe = Pipeline( [ ("correlation_filter", CorrelationFilter(corr_threshold)), ("scaler", StandardScaler()), ("selector", final_selector), ("model", final_model), ] )
    # After Optuna chooses the best trial, refit that complete pipeline on all
    # observations available in the current outer-training fold.
    final_pipe.fit(X_train_outer, y_train_outer)

    prediction_train = final_pipe.predict(X_train_outer)
    prediction_test = final_pipe.predict(X_test_outer)

    return {
        "model": final_pipe,
        "best_params": best_params,
        "feature_selection_method": best_selector_name,
        "selected_features": get_selected_feature_names(final_pipe),
        "n_features_after_correlation_filter": len(
            final_pipe.named_steps["correlation_filter"].selected_features_
        ),
        "pred_train": prediction_train,
        "pred_test": prediction_test,
        "metrics": {
            "r2_train": r2_score(y_train_outer, prediction_train),
            "rmse_train": safe_rmse(y_train_outer, prediction_train),
            "mae_train": mean_absolute_error(y_train_outer, prediction_train),
            "r2_test": r2_score(y_test_outer, prediction_test),
            "rmse_test": safe_rmse(y_test_outer, prediction_test),
            "mae_test": mean_absolute_error(y_test_outer, prediction_test),
        },
    }


### 6. Repeated nested-CV orchestration ###
class EnsembleModel:
    """Average predictions from all outer-fold pipelines of one model family."""

    def __init__(self, models: list[Pipeline]):
        self.models = models

    def predict_members(self, X_input: pd.DataFrame) -> np.ndarray:
        return np.column_stack([model.predict(X_input) for model in self.models])

    def predict(self, X_input: pd.DataFrame) -> np.ndarray:
        return self.predict_members(X_input).mean(axis=1)

    def predict_std(self, X_input: pd.DataFrame) -> np.ndarray:
        return self.predict_members(X_input).std(axis=1)


def refit_full_training_ensembles( cv_models_dict: dict[str, EnsembleModel], X: pd.DataFrame, y: pd.Series | Iterable[float], ) -> dict[str, EnsembleModel]:
    """Refit every selected outer-fold pipeline on the complete dataset.

    The fold ensembles remain reserved for nested-CV/OOF evaluation. The
    returned ensembles are deployment models for full-training visualization,
    SHAP, external testing and virtual screening.
    """
    X_full = X.reset_index(drop=True)
    y_full = pd.Series(y).reset_index(drop=True).astype(float)
    full_models: dict[str, EnsembleModel] = {}

    for model_name, cv_ensemble in cv_models_dict.items():
        refitted_models = []
        for fitted_pipeline in cv_ensemble.models:
            pipeline = clone(fitted_pipeline)
            pipeline.fit(X_full, y_full)
            refitted_models.append(pipeline)
        full_models[model_name] = EnsembleModel(refitted_models)

    return full_models


def build_composite_groups(metadata: pd.DataFrame, group_columns: list[str] | None, separator: str = " | ") -> pd.Series | None:
    """Build one group label from one or several metadata columns."""
    if not group_columns:
        return None
    missing=[c for c in group_columns if c not in metadata.columns]
    if missing:
        raise KeyError(f"Group column(s) not found in metadata: {missing}")
    if len(group_columns)==1:
        return metadata[group_columns[0]].astype(str)
    return metadata[group_columns].astype(str).agg(separator.join,axis=1)


def run_nested_regression(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: list[str],
    outer_n_splits: int,
    outer_n_repeats: int,
    inner_cv_splits: int,
    inner_cv_repeats: int,
    max_features_cap: int,
    min_features: int,
    corr_threshold: float | None,
    n_trials: int,
    overfit_penalty_weight: float,
    optimize_metric: str = "r2",
    random_state: int = 42,
    verbose: bool = True,
    metadata: pd.DataFrame | None = None,
    outer_split_strategy: str = "random",
    inner_split_strategy: str = "random",
    split_group_columns: list[str] | None = None,
    y_stratify_bins: int = 5,
    ks_test_fraction: float = 0.20,
    held_out_groups: list[Any] | None = None,
    predefined_split_column: str | None = None,
    predefined_train_label: Any = "train",
    export_split_indices: bool = False,
    export_test_groups: bool = False,
    max_test_groups_to_export: int = 50,
    rfe_step: int | float = 1,
    n_jobs: int = 1,
    one_se: bool = False,
) -> dict[str, Any]:
    """Run nested CV with configurable outer and inner split strategies."""
    max_features_cap=min(max_features_cap,X.shape[1])
    min_features=max(1,min(int(min_features),max_features_cap))
    metadata=metadata.reset_index(drop=True) if metadata is not None else pd.DataFrame(index=np.arange(len(X)))
    groups=build_composite_groups(metadata,split_group_columns)
    predefined_labels=None
    if predefined_split_column is not None:
        if predefined_split_column not in metadata.columns:
            raise KeyError(f"Predefined split column '{predefined_split_column}' was not retained.")
        predefined_labels=metadata[predefined_split_column].reset_index(drop=True)
    # Outer splits estimate generalization performance. Every observation in an
    # outer validation set remains unseen by tuning and fitting for that fold.
    outer_splits=make_regression_splits(X=X,y=y,n_splits=outer_n_splits,n_repeats=outer_n_repeats,random_state=random_state,strategy=outer_split_strategy,groups=groups,y_stratify_bins=y_stratify_bins,ks_test_fraction=ks_test_fraction,held_out_groups=held_out_groups,predefined_labels=predefined_labels,predefined_train_label=predefined_train_label)
    summary_results=[]; all_predictions=[]; all_params=[]; all_selected_features=[]; split_manifest=[]; split_assignments=[]; models_dict={}
    for fold_number,(train_index,test_index) in enumerate(outer_splits,start=1):
        for sample_index in train_index:
            split_assignments.append({ "Fold": fold_number, "Sample_ID": int(sample_index), "Set": "Training", "Target": float(y.iloc[int(sample_index)]), })
        for sample_index in test_index:
            split_assignments.append({ "Fold": fold_number, "Sample_ID": int(sample_index), "Set": "Validation", "Target": float(y.iloc[int(sample_index)]), })

        row={"Fold":fold_number,"Outer_Split_Strategy":outer_split_strategy,"N_Train":len(train_index),"N_Test":len(test_index)}
        if export_split_indices:
            row["Train_Indices"]=', '.join(map(str,train_index)); row["Test_Indices"]=', '.join(map(str,test_index))
        if groups is not None:
            train_groups=pd.unique(groups.iloc[train_index]); test_groups=pd.unique(groups.iloc[test_index])
            row["N_Train_Groups"]=len(train_groups); row["N_Test_Groups"]=len(test_groups)
            if export_test_groups:
                row["Test_Groups"]=', '.join(map(str,test_groups[:max_test_groups_to_export]))
                row["Test_Groups_Truncated"]=len(test_groups)>max_test_groups_to_export
        if predefined_labels is not None:
            row["Test_Predefined_Labels"]=', '.join(map(str,pd.unique(predefined_labels.iloc[test_index])))
        split_manifest.append(row)
    def _fit_outer(model_name, global_fold, train_index, test_index):
        inner_groups = (groups.iloc[train_index].reset_index(drop=True) if groups is not None and inner_split_strategy in {"group", "cluster"} else None)
        if inner_split_strategy in {"ks", "ks_holdout", "predefined", "leave_group_out"}:
            raise ValueError(f"{inner_split_strategy!r} is not supported for inner CV")
        return train_outer_fold(
            model_name=model_name,
            X_train_outer=X.iloc[train_index].reset_index(drop=True),
            y_train_outer=y.iloc[train_index].reset_index(drop=True),
            X_test_outer=X.iloc[test_index].reset_index(drop=True),
            y_test_outer=y.iloc[test_index].reset_index(drop=True),
            inner_cv_splits=inner_cv_splits, inner_cv_repeats=inner_cv_repeats,
            max_features_cap=max_features_cap, min_features=min_features,
            corr_threshold=corr_threshold, n_trials=n_trials,
            overfit_penalty_weight=overfit_penalty_weight,
            optimize_metric=optimize_metric, random_state=random_state,
            inner_split_strategy=inner_split_strategy, inner_groups=inner_groups,
            y_stratify_bins=y_stratify_bins, rfe_step=rfe_step, one_se=one_se, n_jobs=1)

    for model_name in model_names:
        if verbose: print(f"\nStarting {model_name}")
        y_true_all=[]; y_pred_all=[]; model_folds=[]
        if n_jobs > 1:
            with parallel_backend("loky", inner_max_num_threads=1):
                fold_results = Parallel(n_jobs=min(n_jobs, len(outer_splits)))( delayed(_fit_outer)(model_name, fold, tr, te) for fold, (tr, te) in enumerate(outer_splits, 1))
        else:
            fold_results = None
        for global_fold,(train_index,test_index) in enumerate(outer_splits,start=1):
            if verbose: print(f"  Fold {global_fold}/{len(outer_splits)}")
            inner_groups=groups.iloc[train_index].reset_index(drop=True) if groups is not None and inner_split_strategy in {'group','cluster'} else None
            if inner_split_strategy in {'ks','ks_holdout','predefined','leave_group_out'}:
                raise ValueError(f"'{inner_split_strategy}' is not supported for inner CV.")
            # Tune on the inner CV and evaluate once on this outer validation
            # fold. The returned pipeline becomes one member of the ensemble.
            result = (fold_results[global_fold - 1] if fold_results is not None else _fit_outer(model_name, global_fold, train_index, test_index))
            pred=result['pred_test']; y_test=y.iloc[test_index]; metrics=result['metrics']
            for local_i,(true,predicted) in enumerate(zip(y_test.values,pred)):
                original=int(test_index[local_i]); prow={"Model":model_name,"Fold":global_fold,"Sample_ID":original,"True":true,"Predicted":predicted,"Residual":true-predicted}
                for c in metadata.columns: prow[c]=metadata.iloc[original][c]
                all_predictions.append(prow); y_true_all.append(true); y_pred_all.append(predicted)
            all_params.append({"Model":model_name,"Fold":global_fold,"Outer_Split_Strategy":outer_split_strategy,"Inner_Split_Strategy":inner_split_strategy,"Optimize_Metric":optimize_metric,**metrics,"N_Features_After_Correlation_Filter":result['n_features_after_correlation_filter'],"Params":str(result['best_params']),"Feature_Selection_Method":result['feature_selection_method'],"Features":', '.join(result['selected_features'])})
            for feature in result['selected_features']:
                all_selected_features.append({"Model":model_name,"Fold":global_fold,"Feature_Selection_Method":result['feature_selection_method'],"Feature":feature})
            model_folds.append({"Pipeline":result['model'],"R2_test":metrics['r2_test']})
        fold_r2=[f['R2_test'] for f in model_folds]
        summary_results.append({"Model":model_name,"Optimize_Metric":optimize_metric,"Mean_R2_CV":np.mean(fold_r2),"Std_R2_CV":np.std(fold_r2),"R2_all_repeated_oof_predictions":r2_score(y_true_all,y_pred_all),"RMSE_all_repeated_oof_predictions":safe_rmse(y_true_all,y_pred_all),"MAE_all_repeated_oof_predictions":mean_absolute_error(y_true_all,y_pred_all)})
        # The final deployable predictor is the mean of all fitted outer-fold
        # pipelines. This preserves fold-specific preprocessing and selections.
        models_dict[model_name]=EnsembleModel([f['Pipeline'] for f in model_folds])
    results=_build_result_tables(models_dict,summary_results,all_predictions,all_params,all_selected_features,optimize_metric)
    results['split_manifest']=pd.DataFrame(split_manifest)
    results['split_assignments']=pd.DataFrame(split_assignments)
    full_models_dict = refit_full_training_ensembles(models_dict, X, y)
    results['full_models_dict'] = full_models_dict
    full_rows=[]
    for model_name, ensemble in full_models_dict.items():
        member_predictions=ensemble.predict_members(X)
        mean_predictions=member_predictions.mean(axis=1)
        std_predictions=member_predictions.std(axis=1)
        for sample_id,(true_value,predicted_value,prediction_std) in enumerate(zip(y,mean_predictions,std_predictions)):
            row={'Model':model_name,'Sample_ID':sample_id,'True':true_value,'Predicted':predicted_value,'Residual':true_value-predicted_value,'Prediction_STD':prediction_std,'CI95_Lower':predicted_value-1.96*prediction_std,'CI95_Upper':predicted_value+1.96*prediction_std}
            for c in metadata.columns:
                row[c]=metadata.iloc[sample_id][c]
            full_rows.append(row)
    results['predictions_full_ensemble']=pd.DataFrame(full_rows)
    return results


### 7. Result aggregation and Excel export ###
def _build_result_tables( models_dict, summary_results, all_predictions, all_params, all_selected_features, optimize_metric, ) -> dict[str, Any]:
    optimize_metric = str(optimize_metric).lower()
    nested_sort_columns = {
        "r2": ("R2_all_repeated_oof_predictions", False),
        "rmse": ("RMSE_all_repeated_oof_predictions", True),
        "mae": ("MAE_all_repeated_oof_predictions", True),
    }
    if optimize_metric not in nested_sort_columns:
        raise ValueError("optimize_metric must be one of: r2, rmse, mae.")
    nested_sort_column, nested_ascending = nested_sort_columns[optimize_metric]
    df_summary_nested = pd.DataFrame(summary_results).sort_values( by=nested_sort_column, ascending=nested_ascending )
    df_summary_nested.insert(1, "Model_selection_metric", optimize_metric)
    df_predictions_nested = pd.DataFrame(all_predictions)
    # Repeated CV gives several OOF predictions per sample. Average them here
    # so each sample contributes once to the principal summary metrics.
    df_predictions_mean = (
        df_predictions_nested.groupby(["Model", "Sample_ID"])
        .agg(**{"True":("True","mean"),"Predicted":("Predicted","mean"),"Prediction_STD":("Predicted",lambda x: np.std(x,ddof=0))})
        .reset_index()
    )
    df_predictions_mean["Residual"] = df_predictions_mean["True"] - df_predictions_mean["Predicted"]
    df_predictions_mean["CI95_Lower"] = df_predictions_mean["Predicted"] - 1.96 * df_predictions_mean["Prediction_STD"]
    df_predictions_mean["CI95_Upper"] = df_predictions_mean["Predicted"] + 1.96 * df_predictions_mean["Prediction_STD"]

    summary_mean_rows = []
    for model_name in df_predictions_mean["Model"].unique():
        df_m = df_predictions_mean[df_predictions_mean["Model"] == model_name]
        summary_mean_rows.append(
            {
                "Model": model_name,
                "R2_mean_oof_prediction_per_sample": r2_score(
                    df_m["True"], df_m["Predicted"]
                ),
                "RMSE_mean_oof_prediction_per_sample": safe_rmse(
                    df_m["True"], df_m["Predicted"]
                ),
                "MAE_mean_oof_prediction_per_sample": mean_absolute_error(
                    df_m["True"], df_m["Predicted"]
                ),
            }
        )
    mean_sort_columns = {
        "r2": ("R2_mean_oof_prediction_per_sample", False),
        "rmse": ("RMSE_mean_oof_prediction_per_sample", True),
        "mae": ("MAE_mean_oof_prediction_per_sample", True),
    }
    mean_sort_column, mean_ascending = mean_sort_columns[optimize_metric]
    df_summary_mean = pd.DataFrame(summary_mean_rows).sort_values( by=mean_sort_column, ascending=mean_ascending )
    df_summary_mean.insert(1, "Model_selection_metric", optimize_metric)

    df_summary_mean["r2_mean_prediction"] = df_summary_mean[ "R2_mean_oof_prediction_per_sample" ]
    df_summary_mean["rmse_mean_prediction"] = df_summary_mean[ "RMSE_mean_oof_prediction_per_sample" ]
    df_summary_mean["mae_mean_prediction"] = df_summary_mean[ "MAE_mean_oof_prediction_per_sample" ]
    df_summary_nested["R2_all_predictions"] = df_summary_nested[ "R2_all_repeated_oof_predictions" ]
    df_summary_nested["RMSE_all_predictions"] = df_summary_nested[ "RMSE_all_repeated_oof_predictions" ]
    df_summary_nested["MAE_all_predictions"] = df_summary_nested[ "MAE_all_repeated_oof_predictions" ]

    df_params = pd.DataFrame(all_params)
    df_train_mean = ( df_params.groupby("Model")[["r2_train", "rmse_train", "mae_train"]] .mean() .reset_index() )
    df_test_mean = ( df_params.groupby("Model")[["r2_test", "rmse_test", "mae_test"]] .mean() .reset_index() )

    df_selected_features = pd.DataFrame(all_selected_features)
    n_outer_folds = df_params["Fold"].nunique()

    if df_selected_features.empty:
        df_feature_frequency = pd.DataFrame()
        df_feature_frequency_by_selector = pd.DataFrame()
    else:
        df_feature_frequency = ( df_selected_features.groupby(["Model", "Feature"]) .size() .reset_index(name="Count") .sort_values(["Model", "Count"], ascending=[True, False]) )
        df_feature_frequency["Selection_Frequency"] = ( df_feature_frequency["Count"] / n_outer_folds )

        df_feature_frequency_by_selector = (
            df_selected_features.groupby(
                ["Model", "Feature_Selection_Method", "Feature"]
            )
            .size()
            .reset_index(name="Count")
            .sort_values(
                ["Model", "Feature_Selection_Method", "Count"],
                ascending=[True, True, False],
            )
        )
        df_feature_frequency_by_selector["Selection_Frequency"] = ( df_feature_frequency_by_selector["Count"] / n_outer_folds )

    df_selector_frequency = (
        df_params.groupby(["Model", "Feature_Selection_Method"])
        .size()
        .reset_index(name="Count")
        .sort_values(["Model", "Count"], ascending=[True, False])
    )
    df_selector_frequency["Selection_Method_Frequency"] = ( df_selector_frequency["Count"] / n_outer_folds )

    # Representative features are ranked by how frequently they were selected
    # across outer folds. They are intended for interpretation and plotting, not
    # as a second feature-selection step.
    representative_features = {}
    if not df_feature_frequency.empty:
        for model_name in df_feature_frequency["Model"].unique():
            representative_features[model_name] = (
                df_feature_frequency[df_feature_frequency["Model"] == model_name]
                .sort_values(["Count", "Feature"], ascending=[False, True])["Feature"]
                .tolist()
            )

    return {
        "models_dict": models_dict,
        "summary_nested": df_summary_nested,
        "summary_mean": df_summary_mean,
        "predictions_all": df_predictions_nested,
        "predictions_mean": df_predictions_mean,
        "params": df_params,
        "train_mean": df_train_mean,
        "test_mean": df_test_mean,
        "selected_features": df_selected_features,
        "feature_frequency": df_feature_frequency,
        "feature_frequency_by_selector": df_feature_frequency_by_selector,
        "selector_frequency": df_selector_frequency,
        "representative_features": representative_features,
        "threshold_summary": pd.DataFrame(),
        "candidate_inner_scores": pd.DataFrame(),
    }

def _configuration_to_dataframe(configuration: dict[str, Any] | None) -> pd.DataFrame:
    """Convert workflow settings to a two-column table for Excel export."""
    if not configuration:
        return pd.DataFrame(columns=["Parameter", "Value"])

    rows = []
    for parameter, value in configuration.items():
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, (list, tuple)):
            value = ", ".join(map(str, value))
        elif value is None:
            value = "None"

        rows.append({"Parameter": parameter, "Value": value})

    return pd.DataFrame(rows)


def export_workflow_results( results: dict[str, Any], output_file: str | Path, save_excel: bool = True, configuration: dict[str, Any] | None = None, ) -> Path | None:
    """Export the complete workflow output to one Excel workbook."""
    if not save_excel:
        return None

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_file) as writer:
        if configuration is not None:
            _configuration_to_dataframe(configuration).to_excel( writer, sheet_name="Configuration", index=False, )

        sheet_map = {
            "summary_nested": "Summary_all_predictions",
            "summary_mean": "Summary_mean_predictions",
            "predictions_all": "Predictions_all",
            "predictions_mean": "Predictions_mean",
            "params": "Params",
            "train_mean": "Train_mean",
            "test_mean": "Test_mean",
            "feature_frequency": "Feature_frequency",
            "feature_frequency_by_selector": "Feature_freq_by_selector",
            "selector_frequency": "Selector_frequency",
            "split_manifest": "Split_manifest",
            "split_assignments": "Split_assignments",
            "predictions_full_ensemble": "Predictions_full_ensemble",
            "candidate_inner_scores": "Candidate_scores",
            "threshold_summary": "Threshold_summary",
        }

        for result_key, sheet_name in sheet_map.items():
            dataframe = results.get(result_key)
            if isinstance(dataframe, pd.DataFrame):
                dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

    return output_file

### 8. General plotting utilities ###
def _finish_figure( output_path: str | Path | None, save_figures: bool, show_figures: bool, plot_settings: dict[str, Any] | None = None, ):
    """Save, display and close the current Matplotlib figure."""
    settings = _resolve_plot_settings(plot_settings)

    if save_figures and output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig( output_path, dpi=settings["dpi"], bbox_inches="tight", )

    if show_figures:
        plt.show()

    plt.close()

DEFAULT_PLOT_SETTINGS = {
    "font_size": 8,
    "title_size": 9,
    "label_size": 8,
    "tick_size":8,
    "legend_size": 8,
    "parity_figure_size": (3, 3),
    "bar_figure_size": (3, 3),
    "point_size": 60,
    "point_alpha": 1.0,
    "line_width": 1.0,
    "dpi": 300,
    "file_format": "svg",
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

def _resolve_plot_settings( plot_settings: dict[str, Any] | None = None, ) -> dict[str, Any]:
    settings = DEFAULT_PLOT_SETTINGS.copy()
    if plot_settings:
        unknown = sorted(set(plot_settings) - set(settings))
        if unknown:
            raise KeyError(f"Unknown plot setting(s): {unknown}")
        settings.update(plot_settings)
    return settings


def _style_axes(settings: dict[str, Any]):
    plt.xticks(fontsize=settings["tick_size"])
    plt.yticks(fontsize=settings["tick_size"])
    if settings["show_grid"]:
        plt.grid(alpha=settings["grid_alpha"])


def _annotate_bar_values( bars, values=None, horizontal: bool = False, fmt: str = ".1f", fontsize: float | None = None, ):
    bars = list(bars)
    if values is None:
        values = [bar.get_width() if horizontal else bar.get_height() for bar in bars]

    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue

        if horizontal:
            x_position = bar.get_x() + bar.get_width() / 2
            y_position = bar.get_y() + bar.get_height() / 2
        else:
            x_position = bar.get_x() + bar.get_width() / 2
            y_position = bar.get_y() + bar.get_height() / 2

        plt.text( x_position, y_position, format(float(value), fmt), ha="center", va="center", fontsize=fontsize, )



### 9. Model and split diagnostic plots ###
def plot_correlation_heatmap(
    X: pd.DataFrame,
    results_dir: str | Path,
    output_prefix: str,
    save_figures: bool = True,
    show_figures: bool = True,
    figure_size: tuple[float, float] = (5, 4),
    plot_settings: dict[str, Any] | None = None,
):
    """Plot the descriptor correlation matrix without dense annotations."""
    settings = _resolve_plot_settings(plot_settings)
    plt.figure(figsize=figure_size)
    matrix = X.corr()
    image = plt.imshow( matrix, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm", )
    plt.colorbar(image, label="Pearson correlation")
    plt.xticks( range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=max(4, settings["tick_size"] - 2), )
    plt.yticks( range(len(matrix.index)), matrix.index, fontsize=max(4, settings["tick_size"] - 2), )
    plt.title("Feature correlation matrix", fontsize=settings["title_size"])
    plt.tight_layout()

    extension = settings["file_format"]
    path = Path(results_dir) / f"correlation_heatmap_{output_prefix}.{extension}"
    _finish_figure(path, save_figures, show_figures, settings)
    return path


def plot_target_distribution(
    y: pd.Series | Iterable[float],
    results_dir: str | Path = ".",
    output_prefix: str = "workflow",
    target_label: str = "Target",
    bins: int = 15,
    density: bool = False,
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
) -> Path:
    """Plot the overall distribution of the continuous response variable."""
    settings = _resolve_plot_settings(plot_settings)
    y_values = np.asarray(list(y), dtype=float)
    y_values = y_values[np.isfinite(y_values)]

    if len(y_values) == 0:
        raise ValueError("No finite target values are available for plotting.")

    plt.figure(figsize=settings["bar_figure_size"])
    plt.hist( y_values, bins=bins, color=settings["bar_color"], edgecolor=settings["bar_edge_color"], linewidth=settings["bar_edge_width"], density=density, )
    plt.xlabel(target_label, fontsize=settings["label_size"])
    plt.ylabel( "Density" if density else "Frequency", fontsize=settings["label_size"], )
    _style_axes(settings)
    plt.tight_layout()

    extension = settings["file_format"]
    path = Path(results_dir) / f"target_distribution_{output_prefix}.{extension}"
    _finish_figure(path, save_figures, show_figures, settings)
    return path


def plot_target_split_distribution(
    y: pd.Series | Iterable[float],
    split_assignments: pd.DataFrame,
    fold: int = 1,
    results_dir: str | Path = ".",
    output_prefix: str = "regression",
    target_label: str = "Target",
    bins: int = 15,
    external_y: pd.Series | Iterable[float] | None = None,
    external_label: str = "External test",
    density: bool = False,
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
) -> Path:
    """Compare the full target distribution with one outer fold's subsets.

    In nested cross-validation, training and validation membership changes from
    fold to fold. This plot therefore shows one explicitly selected outer fold.
    All distributions use the same bin edges.
    """
    settings = _resolve_plot_settings(plot_settings)
    y_full = np.asarray(list(y), dtype=float)
    selected_fold = split_assignments[ split_assignments["Fold"] == int(fold) ].copy()

    if selected_fold.empty:
        available = sorted(split_assignments["Fold"].unique().tolist())
        raise ValueError( f"Fold {fold} was not found. Available outer folds: {available}" )

    y_training = selected_fold.loc[ selected_fold["Set"] == "Training", "Target" ].to_numpy(dtype=float)
    y_validation = selected_fold.loc[ selected_fold["Set"] == "Validation", "Target" ].to_numpy(dtype=float)

    values_for_bins = [y_full, y_training, y_validation]
    y_external = None
    if external_y is not None:
        y_external = np.asarray(list(external_y), dtype=float)
        y_external = y_external[np.isfinite(y_external)]
        if len(y_external):
            values_for_bins.append(y_external)

    all_values = np.concatenate(values_for_bins)
    all_values = all_values[np.isfinite(all_values)]
    bin_edges = np.histogram_bin_edges(all_values, bins=bins)

    plt.figure(figsize=settings["bar_figure_size"])

    # Full distribution as a clear outline.
    plt.hist(
        y_full,
        bins=bin_edges,
        alpha=0.75,
        color=settings["full_distribution_color"],
        edgecolor=settings["bar_edge_color"],
        linewidth=settings["bar_edge_width"],
        density=density,
        label=f"Full dataset (n={len(y_full)})",
    )

    # Selected outer-fold subsets.
    plt.hist(
        y_training,
        bins=bin_edges,
        alpha=0.75,
        color=settings["training_color"],
        edgecolor=settings["bar_edge_color"],
        linewidth=settings["bar_edge_width"],
        density=density,
        label=f"Training, fold {fold} (n={len(y_training)})",
    )
    plt.hist(
        y_validation,
        bins=bin_edges,
        alpha=0.75,
        color=settings["validation_color"],
        edgecolor=settings["bar_edge_color"],
        linewidth=settings["bar_edge_width"],
        density=density,
        label=f"Validation, fold {fold} (n={len(y_validation)})",
    )

    if y_external is not None and len(y_external):
        plt.hist(
            y_external,
            bins=bin_edges,
            histtype="step",
            linewidth=settings["line_width"],
            color=settings["test_color"],
            density=density,
            label=f"{external_label} (n={len(y_external)})",
        )

    plt.xlabel(target_label, fontsize=settings["label_size"])
    plt.ylabel( "Density" if density else "Frequency", fontsize=settings["label_size"], )
    plt.title( f"Target distribution across dataset subsets: outer fold {fold}", fontsize=settings["title_size"], )
    plt.legend(loc="upper right", fontsize=settings["legend_size"])
    _style_axes(settings)
    plt.tight_layout()

    extension = settings["file_format"]
    path = Path(results_dir) / ( f"target_split_distribution_fold_{fold}_{output_prefix}.{extension}" )
    _finish_figure(path, save_figures, show_figures, settings)
    return path


def plot_oof_parity(
    results: dict[str, Any],
    results_dir: str | Path,
    output_prefix: str,
    target_label: str,
    save_figures: bool = True,
    show_figures: bool = True,
    axis_limits: tuple[float, float] | None = None,
    plot_settings: dict[str, Any] | None = None,
):
    """Plot averaged out-of-fold predictions for each model family."""
    settings=_resolve_plot_settings(plot_settings); paths=[]
    for model_name in results["predictions_mean"]["Model"].unique():
        df_model=results["predictions_mean"][results["predictions_mean"]["Model"]==model_name].copy().sort_values("True"); summary=results["summary_mean"][results["summary_mean"]["Model"]==model_name].iloc[0]
        x=df_model["True"].to_numpy(float); pred=df_model["Predicted"].to_numpy(float); sd=df_model["Prediction_STD"].to_numpy(float)
        plt.figure(figsize=settings["parity_figure_size"]); plt.errorbar(x,pred,yerr=sd,fmt="none",ecolor="#666666",elinewidth=1.5,alpha=0.95,capsize=2,zorder=2,label="±1 SD")
        if len(x)>=3 and np.ptp(x)>0:
            slope,intercept=np.polyfit(x,pred,1); x_fit=np.linspace(x.min(),x.max(),200); y_fit=intercept+slope*x_fit; residuals=pred-(intercept+slope*x); dof=len(x)-2; s_err=np.sqrt(np.sum(residuals**2)/dof); sxx=np.sum((x-x.mean())**2); se_fit=s_err*np.sqrt(1/len(x)+(x_fit-x.mean())**2/sxx); delta=t.ppf(0.975,dof)*se_fit
            plt.fill_between(x_fit,y_fit-delta,y_fit+delta,color=settings["oof_color"],alpha=0.18,label="95% confidence interval",zorder=1); plt.plot(x_fit,y_fit,color=settings["oof_color"],linewidth=settings["line_width"],zorder=2,label="Regression fit")
        plt.scatter(x,pred,s=settings["point_size"],alpha=settings["point_alpha"],color=settings["oof_color"],edgecolor=settings["point_edge_color"],linewidth=settings["point_edge_width"],label="Out of fold",zorder=3)
        minimum=min(x.min(),pred.min()); maximum=max(x.max(),pred.max()); plt.plot([minimum,maximum],[minimum,maximum],linewidth=settings["line_width"],color=settings["reference_line_color"])
        plt.text(0.05,0.95,f"OOF R² = {summary['R2_mean_oof_prediction_per_sample']:.3f}\nOOF MAE = {summary['MAE_mean_oof_prediction_per_sample']:.3f}\nOOF RMSE = {summary['RMSE_mean_oof_prediction_per_sample']:.3f}",transform=plt.gca().transAxes,va="top",fontsize=settings["legend_size"]); plt.xlabel(f"Experimental {target_label}",fontsize=settings["label_size"]); plt.ylabel(f"Predicted {target_label}",fontsize=settings["label_size"]); plt.title(f"Ensemble {model_name}",fontsize=settings["title_size"]); plt.legend(loc="upper center",bbox_to_anchor=(0.5,-0.20),ncol=2,fontsize=settings["legend_size"],frameon=True)
        plt.subplots_adjust(bottom=0.28)
        if axis_limits is not None: plt.xlim(*axis_limits); plt.ylim(*axis_limits)
        if settings["equal_aspect"]: plt.gca().set_aspect("equal",adjustable="box")
        _style_axes(settings); plt.tight_layout(); extension=settings["file_format"]; path=Path(results_dir)/f"oof_parity_{model_name}_{output_prefix}.{extension}"; _finish_figure(path,save_figures,show_figures,plot_settings=settings); paths.append(path)
    return paths

def plot_model_metric_comparison(
    results: dict[str, Any],
    results_dir: str | Path,
    output_prefix: str,
    metric: str = "R2",
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
):
    """Compare model families using averaged OOF R², MAE or RMSE."""
    settings = _resolve_plot_settings(plot_settings)
    metric_key_map = { "R2": "R2_mean_oof_prediction_per_sample", "MAE": "MAE_mean_oof_prediction_per_sample", "RMSE": "RMSE_mean_oof_prediction_per_sample", }

    metric = metric.upper()
    if metric not in metric_key_map:
        raise ValueError("metric must be one of: R2, MAE, RMSE.")

    metric_column = metric_key_map[metric]
    summary = results["summary_mean"].sort_values( metric_column, ascending=(metric != "R2"), )

    plt.figure(figsize=settings["bar_figure_size"])
    bars = plt.bar( summary["Model"], summary[metric_column], color=settings["bar_color"], edgecolor=settings["bar_edge_color"], linewidth=settings["bar_edge_width"], )
    plt.ylabel(f"OOF {metric}", fontsize=settings["label_size"])
    plt.xlabel("Model", fontsize=settings["label_size"])
    plt.title( f"Model comparison: OOF {metric}", fontsize=settings["title_size"], )
    plt.xticks(rotation=45, ha="right", fontsize=settings["tick_size"])

    _annotate_bar_values( bars, summary[metric_column], fmt=".1f", fontsize=settings["tick_size"], )

    _style_axes(settings)
    plt.tight_layout()
    extension = settings["file_format"]
    path = Path(results_dir) / ( f"model_comparison_{metric.lower()}_{output_prefix}.{extension}" )
    _finish_figure( path, save_figures, show_figures, plot_settings=settings, )
    return path

def plot_overfitting_gap(
    results: dict[str, Any],
    results_dir: str | Path,
    output_prefix: str,
    metric: str = "r2",
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
):
    """Plot the mean outer-fold training-to-test performance gap."""
    settings = _resolve_plot_settings(plot_settings)
    metric = metric.lower()
    metric_columns = { "r2": ("r2_train", "r2_test", "R²"), "rmse": ("rmse_train", "rmse_test", "RMSE"), "mae": ("mae_train", "mae_test", "MAE"), }
    if metric not in metric_columns:
        raise ValueError("metric must be one of: r2, rmse, mae.")
    train_column, test_column, metric_label = metric_columns[metric]
    train = results["train_mean"][["Model", train_column]]
    test = results["test_mean"][["Model", test_column]]
    comparison = train.merge(test, on="Model", how="inner")
    if metric == "r2":
        comparison["Metric_gap"] = comparison[train_column] - comparison[test_column]
    else:
        comparison["Metric_gap"] = comparison[test_column] - comparison[train_column]
    comparison = comparison.sort_values("Metric_gap", ascending=False)
    plt.figure(figsize=settings["bar_figure_size"])
    bars = plt.bar( comparison["Model"], comparison["Metric_gap"], color=settings["bar_color"], edgecolor=settings["bar_edge_color"], linewidth=settings["bar_edge_width"], )
    plt.axhline(0, linewidth=settings["line_width"], color=settings["reference_line_color"])
    plt.ylabel(f"Training-to-test degradation in {metric_label}", fontsize=settings["label_size"])
    plt.xlabel("Model", fontsize=settings["label_size"])
    plt.title("Train–test performance gap", fontsize=settings["title_size"])
    plt.xticks(rotation=45, ha="right", fontsize=settings["tick_size"])
    _annotate_bar_values(bars, comparison["Metric_gap"], fmt=".2f", fontsize=settings["tick_size"])
    _style_axes(settings)
    plt.tight_layout()
    extension = settings["file_format"]
    path = Path(results_dir) / f"overfitting_gap_{output_prefix}.{extension}"
    _finish_figure(path, save_figures, show_figures, settings)
    return path

def plot_selector_frequency(
    results: dict[str, Any],
    results_dir: str | Path,
    output_prefix: str,
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
):
    """Plot how frequently each feature-selection strategy was selected."""
    settings = _resolve_plot_settings(plot_settings)
    dataframe = results["selector_frequency"].copy()
    if dataframe.empty:
        return None

    pivot = dataframe.pivot( index="Model", columns="Feature_Selection_Method", values="Selection_Method_Frequency", ).fillna(0)

    axis = pivot.plot( kind="bar", figsize=settings["bar_figure_size"], color=None, edgecolor=settings["bar_edge_color"], linewidth=settings["bar_edge_width"], )
    axis.set_ylabel("Selection frequency", fontsize=settings["label_size"])
    axis.set_xlabel("Model", fontsize=settings["label_size"])
    axis.set_title( "Feature-selection method frequency", fontsize=settings["title_size"], )
    axis.legend(title="Selector", loc="upper right", fontsize=settings["legend_size"])

    plt.xticks(rotation=45, ha="right", fontsize=settings["tick_size"])
    _style_axes(settings)
    plt.tight_layout()

    extension = settings["file_format"]
    path = Path(results_dir) / f"selector_frequency_{output_prefix}.{extension}"
    _finish_figure(path, save_figures, show_figures, settings)
    return path

def plot_feature_frequency(
    results: dict[str, Any],
    results_dir: str | Path,
    output_prefix: str,
    top_n: int = 15,
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
):
    """Plot the most frequently selected features separately for each model."""
    settings = _resolve_plot_settings(plot_settings)
    dataframe = results["feature_frequency"].copy()
    if dataframe.empty:
        return []

    paths = []
    for model_name in dataframe["Model"].unique():
        subset = ( dataframe[dataframe["Model"] == model_name] .nlargest(top_n, "Selection_Frequency") .sort_values("Selection_Frequency") )

        figure_height = max(3.0, 0.28 * len(subset))
        plt.figure(figsize=(settings["bar_figure_size"][0], figure_height))
        bars = plt.barh(
            subset["Feature"],
            subset["Selection_Frequency"],
            color=settings["bar_color"],
            edgecolor=settings["bar_edge_color"],
            linewidth=settings["bar_edge_width"],
        )
        plt.xlabel("Selection frequency", fontsize=settings["label_size"])
        plt.ylabel("Feature", fontsize=settings["label_size"])
        plt.title( f"Top selected features: {model_name}", fontsize=settings["title_size"], )
        plt.xlim(0, 1)
        _annotate_bar_values( bars, subset["Selection_Frequency"], horizontal=True, fmt=".2f", fontsize=settings["tick_size"], )
        _style_axes(settings)
        plt.tight_layout()

        extension = settings["file_format"]
        path = Path(results_dir) / ( f"feature_frequency_{model_name}_{output_prefix}.{extension}" )
        _finish_figure(path, save_figures, show_figures, settings)
        paths.append(path)

    return paths

def plot_model_prediction_comparison(
    results: dict[str, Any],
    model_x: str,
    model_y: str,
    results_dir: str | Path,
    output_prefix: str,
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
):
    """Compare averaged OOF predictions from two model families."""
    settings = _resolve_plot_settings(plot_settings)
    predictions = results["predictions_mean"]

    x_values = predictions[predictions["Model"] == model_x][ ["Sample_ID", "Predicted"] ].rename(columns={"Predicted": f"Predicted_{model_x}"})
    y_values = predictions[predictions["Model"] == model_y][ ["Sample_ID", "Predicted"] ].rename(columns={"Predicted": f"Predicted_{model_y}"})

    merged = x_values.merge(y_values, on="Sample_ID", how="inner")
    if merged.empty:
        raise ValueError(f"No common predictions found for {model_x} and {model_y}.")

    x_column = f"Predicted_{model_x}"
    y_column = f"Predicted_{model_y}"

    plt.figure(figsize=settings["parity_figure_size"])
    plt.scatter(
        merged[x_column],
        merged[y_column],
        s=settings["point_size"],
        alpha=settings["point_alpha"],
        color=settings["oof_color"],
        edgecolor=settings["point_edge_color"],
        linewidth=settings["point_edge_width"]
    )
    lower = min(merged[x_column].min(), merged[y_column].min())
    upper = max(merged[x_column].max(), merged[y_column].max())
    plt.plot( [lower, upper], [lower, upper], linewidth=settings["line_width"], color=settings["reference_line_color"], )
    plt.xlabel(f"{model_x} OOF prediction", fontsize=settings["label_size"])
    plt.ylabel(f"{model_y} OOF prediction", fontsize=settings["label_size"])
    plt.title(f"{model_x} vs {model_y}", fontsize=settings["title_size"])
    if settings["equal_aspect"]:
        plt.gca().set_aspect("equal", adjustable="box")
    _style_axes(settings)
    plt.tight_layout()

    extension = settings["file_format"]
    path = Path(results_dir) / ( f"model_prediction_comparison_{model_x}_vs_{model_y}_{output_prefix}.{extension}" )
    _finish_figure(path, save_figures, show_figures, settings)
    return path


### 10. External-test and virtual-screening utilities ###
def evaluate_external_dataset(
    df_external: pd.DataFrame,
    training_columns: Iterable[str],
    models_dict: dict[str, EnsembleModel],
    target_column: str,
    id_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
    model_names: list[str] | None = None,
    include_prediction_std: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict an external dataset using the supplied deployment ensemble."""
    X_external = prepare_external_features(
        df_external=df_external,
        training_columns=training_columns,
        target_column=target_column,
        id_columns=id_columns,
        drop_columns=drop_columns,
    )

    # Predictions are produced by the already-fitted fold ensemble. The external
    # dataset is never used to select features or tune hyperparameters.
    model_names = model_names or list(models_dict)
    output = df_external.reset_index(drop=True).copy()
    metrics_rows = []

    has_target = ( target_column in output.columns and output[target_column].notna().all() )
    if has_target:
        y_external = pd.to_numeric(output[target_column], errors="raise")

    for model_name in model_names:
        if model_name not in models_dict:
            raise KeyError(f"Model '{model_name}' is not available.")

        ensemble = models_dict[model_name]
        prediction = ensemble.predict(X_external)
        output[f"Predicted_{model_name}"] = prediction

        if include_prediction_std:
            prediction_std=ensemble.predict_std(X_external); output[f"Prediction_STD_{model_name}"]=prediction_std; output[f"CI95_Lower_{model_name}"]=prediction-1.96*prediction_std; output[f"CI95_Upper_{model_name}"]=prediction+1.96*prediction_std

        if has_target:
            metrics_rows.append(
                {
                    "Model": model_name,
                    "N_Ensemble_Members": len(ensemble.models),
                    "R2_external": r2_score(y_external, prediction),
                    "RMSE_external": safe_rmse(y_external, prediction),
                    "MAE_external": mean_absolute_error(y_external, prediction),
                }
            )

    return output, pd.DataFrame(metrics_rows)


def combine_training_and_external_predictions( results: dict[str, Any], external_results: pd.DataFrame, target_column: str, model_names: list[str] | None = None, ) -> pd.DataFrame:
    """Combine mean OOF training predictions and external predictions."""
    model_names = model_names or list(results["models_dict"])
    combined_rows = []

    training_predictions = results["predictions_mean"]
    for model_name in model_names:
        training_model = training_predictions[ training_predictions["Model"] == model_name ].copy()

        for _, row in training_model.iterrows():
            combined_rows.append(
                {
                    "Dataset": "Training_OOF",
                    "Model": model_name,
                    "Sample_ID": row["Sample_ID"],
                    "True": row["True"],
                    "Predicted": row["Predicted"],
                    "Residual": row["Residual"],
                    "Prediction_STD": row["Prediction_STD"],
                    "CI95_Lower": row["CI95_Lower"],
                    "CI95_Upper": row["CI95_Upper"],
                }
            )

        prediction_column = f"Predicted_{model_name}"
        std_column = f"Prediction_STD_{model_name}"

        for external_index, external_row in external_results.iterrows():
            true_value = ( external_row[target_column] if target_column in external_results.columns else np.nan )
            predicted_value = external_row[prediction_column]
            combined_rows.append(
                {
                    "Dataset": "External",
                    "Model": model_name,
                    "Sample_ID": external_index,
                    "True": true_value,
                    "Predicted": predicted_value,
                    "Residual": (
                        true_value - predicted_value
                        if pd.notna(true_value)
                        else np.nan
                    ),
                    "Prediction_STD": (external_row[std_column] if std_column in external_results.columns else np.nan),
                    "CI95_Lower": (external_row[f"CI95_Lower_{model_name}"] if f"CI95_Lower_{model_name}" in external_results.columns else np.nan),
                    "CI95_Upper": (external_row[f"CI95_Upper_{model_name}"] if f"CI95_Upper_{model_name}" in external_results.columns else np.nan),
                }
            )

    return pd.DataFrame(combined_rows)


def save_external_results(
    detailed_results: pd.DataFrame,
    metrics: pd.DataFrame,
    output_file: str | Path,
    save_excel: bool = True,
    combined_training_external: pd.DataFrame | None = None,
) -> Path | None:
    """Save external predictions, metrics, and optional combined OOF/external data."""
    if not save_excel:
        return None

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_file) as writer:
        detailed_results.to_excel( writer, sheet_name="External_predictions", index=False, )
        metrics.to_excel( writer, sheet_name="External_metrics", index=False, )
        if combined_training_external is not None:
            combined_training_external.to_excel( writer, sheet_name="Training_OOF_and_external", index=False, )

    return output_file

def plot_external_overlay(
    results: dict[str, Any],
    external_results: pd.DataFrame,
    external_metrics: pd.DataFrame,
    target_column: str,
    results_dir: str | Path,
    output_prefix: str,
    external_label: str = "External test",
    model_names: list[str] | None = None,
    save_figures: bool = True,
    show_figures: bool = True,
    axis_limits: tuple[float, float] | None = None,
    plot_settings: dict[str, Any] | None = None,
    training_reference: str = "full_dataset",
):
    """Overlay the external test on the full dataset or validation predictions."""
    settings=_resolve_plot_settings(plot_settings)
    if external_metrics.empty: raise ValueError("External metrics are empty. The external dataset must contain a complete " f"'{target_column}' column to create the overlay.")
    model_names=model_names or list(results["models_dict"]); paths=[]
    for model_name in model_names:
        reference=training_reference.lower()
        if reference in {"full","full_dataset","training"}:
            df_cv=results["predictions_full_ensemble"][results["predictions_full_ensemble"]["Model"]==model_name].copy(); training_label="Full dataset"; training_r2=r2_score(df_cv["True"],df_cv["Predicted"]); training_mae=mean_absolute_error(df_cv["True"],df_cv["Predicted"])
        elif reference in {"validation","oof"}:
            df_cv=results["predictions_mean"][results["predictions_mean"]["Model"]==model_name].copy(); training_label="Validation"; summary=results["summary_mean"][results["summary_mean"]["Model"]==model_name].iloc[0]; training_r2=summary["R2_mean_oof_prediction_per_sample"]; training_mae=summary["MAE_mean_oof_prediction_per_sample"]
        else: raise ValueError("training_reference must be 'full_dataset' or 'validation'.")
        metric=external_metrics[external_metrics["Model"]==model_name].iloc[0]; prediction_column=f"Predicted_{model_name}"; std_column=f"Prediction_STD_{model_name}"; ext=external_results.copy(); plt.figure(figsize=settings["parity_figure_size"])
        x=df_cv["True"].to_numpy(float); pred=df_cv["Predicted"].to_numpy(float); sd=df_cv["Prediction_STD"].to_numpy(float) if "Prediction_STD" in df_cv.columns else None
        if sd is not None: plt.errorbar(x,pred,yerr=sd,fmt="none",ecolor="#666666",elinewidth=1.3,alpha=0.9,capsize=2,zorder=2,label=f"{training_label} ±1 SD")
        if len(x)>=3 and np.ptp(x)>0:
            slope,intercept=np.polyfit(x,pred,1); x_fit=np.linspace(x.min(),x.max(),200); y_fit=intercept+slope*x_fit; residuals=pred-(intercept+slope*x); dof=len(x)-2; s_err=np.sqrt(np.sum(residuals**2)/dof); sxx=np.sum((x-x.mean())**2); se_fit=s_err*np.sqrt(1/len(x)+(x_fit-x.mean())**2/sxx); delta=t.ppf(0.975,dof)*se_fit
            plt.fill_between(x_fit,y_fit-delta,y_fit+delta,color=settings["oof_color"],alpha=0.14,label=f"{training_label} 95% CI",zorder=1); plt.plot(x_fit,y_fit,color=settings["oof_color"],linewidth=settings["line_width"],zorder=2)
        x_ext=ext[target_column].to_numpy(float); pred_ext=ext[prediction_column].to_numpy(float); sd_ext=ext[std_column].to_numpy(float) if std_column in ext.columns else None
        if sd_ext is not None: plt.errorbar(x_ext,pred_ext,yerr=sd_ext,fmt="none",ecolor="#666666",elinewidth=1.5,alpha=1.0,capsize=3,zorder=3,label=f"{external_label} ±1 SD")
        if len(x_ext)>=3 and np.ptp(x_ext)>0:
            slope_ext,intercept_ext=np.polyfit(x_ext,pred_ext,1); x_fit_ext=np.linspace(x_ext.min(),x_ext.max(),200); y_fit_ext=intercept_ext+slope_ext*x_fit_ext; residuals_ext=pred_ext-(intercept_ext+slope_ext*x_ext); dof_ext=len(x_ext)-2; s_err_ext=np.sqrt(np.sum(residuals_ext**2)/dof_ext); sxx_ext=np.sum((x_ext-x_ext.mean())**2); se_fit_ext=s_err_ext*np.sqrt(1/len(x_ext)+(x_fit_ext-x_ext.mean())**2/sxx_ext); delta_ext=t.ppf(0.975,dof_ext)*se_fit_ext
            plt.fill_between(x_fit_ext,y_fit_ext-delta_ext,y_fit_ext+delta_ext,color=settings["external_color"],alpha=0.18,label=f"{external_label} 95% CI",zorder=1); plt.plot(x_fit_ext,y_fit_ext,color=settings["external_color"],linewidth=settings["line_width"],zorder=2)
        plt.scatter(x,pred,s=settings["point_size"],alpha=settings["point_alpha"],color=settings["oof_color"],edgecolor=settings["point_edge_color"],linewidth=settings["point_edge_width"],label=training_label,zorder=4); plt.scatter(x_ext,pred_ext,s=settings["point_size"]*1.2,alpha=0.9,color=settings["external_color"],edgecolor=settings["reference_line_color"],linewidth=settings["point_edge_width"],label=external_label,zorder=5)
        all_values=np.concatenate([x,pred,x_ext,pred_ext]); plt.plot([all_values.min(),all_values.max()],[all_values.min(),all_values.max()],linewidth=settings["line_width"],color=settings["reference_line_color"]); plt.text(0.05,0.95,f"{training_label} R² = {training_r2:.3f}\n{training_label} MAE = {training_mae:.3f}\nExternal R² = {metric['R2_external']:.3f}\nExternal MAE = {metric['MAE_external']:.3f}",transform=plt.gca().transAxes,va="top",fontsize=settings["legend_size"]); plt.xlabel(f"Experimental {target_column}",fontsize=settings["label_size"]); plt.ylabel(f"Predicted {target_column}",fontsize=settings["label_size"]); plt.title(f"Ensemble {model_name}",fontsize=settings["title_size"]); plt.legend(loc="upper center",bbox_to_anchor=(0.5,-0.20),ncol=2,fontsize=settings["legend_size"],frameon=True)
        plt.subplots_adjust(bottom=0.28)
        if axis_limits is not None: plt.xlim(*axis_limits); plt.ylim(*axis_limits)
        if settings["equal_aspect"]: plt.gca().set_aspect("equal",adjustable="box")
        _style_axes(settings); plt.tight_layout(); extension=settings["file_format"]; path=Path(results_dir)/f"external_overlay_{model_name}_{output_prefix}.{extension}"; _finish_figure(path,save_figures,show_figures,settings); paths.append(path)
    return paths

def plot_virtual_screening_distribution(
    y_training: Iterable[float],
    screening_results: pd.DataFrame,
    model_names: list[str],
    results_dir: str | Path,
    output_prefix: str,
    target_label: str,
    bins: int = 20,
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
):
    """Compare training-target and screening prediction distributions."""
    settings = _resolve_plot_settings(plot_settings)
    y_training = np.asarray(list(y_training), dtype=float)
    paths = []

    for model_name in model_names:
        prediction_column = f"Predicted_{model_name}"
        if prediction_column not in screening_results.columns:
            raise KeyError( f"Column '{prediction_column}' was not found in screening results." )

        plt.figure(figsize=settings["bar_figure_size"])
        plt.hist(
            y_training,
            bins=bins,
            alpha=0.55,
            density=False,
            color=settings["oof_color"],
            edgecolor=settings["bar_edge_color"],
            linewidth=settings["bar_edge_width"],
            label="Training experimental values",
        )
        plt.hist(
            screening_results[prediction_column],
            bins=bins,
            alpha=0.55,
            density=True,
            color=settings["external_color"],
            edgecolor=settings["bar_edge_color"],
            linewidth=settings["bar_edge_width"],
            label="Virtual-screening predictions",
        )
        plt.xlabel(target_label, fontsize=settings["label_size"])
        plt.ylabel("Density", fontsize=settings["label_size"])
        plt.title( f"Virtual-screening distribution: {model_name}", fontsize=settings["title_size"], )
        plt.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=settings["legend_size"], framealpha=0.9)
        _style_axes(settings)
        plt.tight_layout()

        extension = settings["file_format"]
        path = Path(results_dir) / ( f"virtual_screening_distribution_{model_name}_{output_prefix}.{extension}" )
        _finish_figure(path, save_figures, show_figures, settings)
        paths.append(path)

    return paths

### 11. Ensemble SHAP interpretation ###
def compute_ensemble_shap(
    ensemble_model: EnsembleModel,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    model_name: str,
    dataset_labels: Iterable[str] | None = None,
    sample_ids: Iterable[Any] | None = None,
    background_size: int = 100,
    max_evals: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Explain the final ensemble mean prediction with permutation SHAP.

    The explained prediction function is ``ensemble_model.predict``. This is the
    same fold-ensemble mean used for external-test and virtual-screening
    predictions. The background is sampled only from the training descriptor
    matrix supplied through ``X_background``.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError( "SHAP is required for ensemble interpretation. Install it with " "'conda install -c conda-forge shap' or 'pip install shap'." ) from exc

    if not isinstance(X_background, pd.DataFrame) or not isinstance(X_explain, pd.DataFrame):
        raise TypeError("X_background and X_explain must be pandas DataFrames.")

    missing = [column for column in X_background.columns if column not in X_explain.columns]
    if missing:
        raise ValueError(f"X_explain is missing required feature columns: {missing}")

    X_explain = X_explain.loc[:, X_background.columns].reset_index(drop=True)
    background_size = min(max(1, int(background_size)), len(X_background))
    background = shap.sample( X_background.reset_index(drop=True), background_size, random_state=random_state, )

    if max_evals is None:
        max_evals = 2 * X_background.shape[1] + 1
    minimum_evals = 2 * X_background.shape[1] + 1
    if max_evals < minimum_evals:
        raise ValueError( f"max_evals must be at least {minimum_evals} for " f"{X_background.shape[1]} features." )

    # Explain the ensemble mean directly rather than explaining one arbitrary
    # fold model. Permutation SHAP is model-agnostic and works with mixed model
    # families and fold-specific feature-selection pipelines.
    explainer = shap.Explainer( ensemble_model.predict, background, algorithm="permutation", feature_names=list(X_background.columns), )
    explanation = explainer(X_explain, max_evals=max_evals)

    labels = ( list(dataset_labels) if dataset_labels is not None else ["Train"] * len(X_explain) )
    ids = list(sample_ids) if sample_ids is not None else list(range(len(X_explain)))
    if len(labels) != len(X_explain) or len(ids) != len(X_explain):
        raise ValueError("dataset_labels and sample_ids must match X_explain length.")

    values = pd.DataFrame(explanation.values, columns=X_background.columns)
    values.insert(0, "Ensemble_Prediction", ensemble_model.predict(X_explain))
    values.insert(0, "Base_Value", np.asarray(explanation.base_values).reshape(-1))
    values.insert(0, "Sample_ID", ids)
    values.insert(0, "Dataset", labels)
    values.insert(0, "Model", model_name)

    importance = pd.DataFrame({
        "Model": model_name,
        "Feature": X_background.columns,
        "Mean_abs_SHAP": np.abs(explanation.values).mean(axis=0),
    }).sort_values("Mean_abs_SHAP", ascending=False).reset_index(drop=True)

    return { "explanation": explanation, "values": values, "importance": importance, "background": background, "X_explain": X_explain, }


def plot_ensemble_shap(
    shap_result: dict[str, Any],
    model_name: str,
    results_dir: str | Path,
    output_prefix: str,
    max_display: int = 15,
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
    figure_size: tuple[float, float] = (5, 4),
) -> list[Path]:
    """Create beeswarm and mean-absolute-SHAP plots for one ensemble."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP is required to create SHAP plots.") from exc

    settings = _resolve_plot_settings(plot_settings)
    explanation = shap_result["explanation"]
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    extension = settings["file_format"]
    paths = []

    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title(f"Ensemble SHAP: {model_name}", fontsize=settings["title_size"])
    plt.tight_layout()
    beeswarm_path = results_dir / f"shap_beeswarm_{model_name}_{output_prefix}.{extension}"
    _finish_figure(beeswarm_path, save_figures, show_figures, settings)
    paths.append(beeswarm_path)

    shap.plots.bar(explanation, max_display=max_display, show=False)
    shap_patches = plt.gca().patches
    for patch in shap_patches:
        patch.set_edgecolor(settings["bar_edge_color"])
        patch.set_linewidth(settings["bar_edge_width"])
    plt.title(f"Mean absolute SHAP: {model_name}", fontsize=settings["title_size"])
    plt.tight_layout()
    bar_path = results_dir / f"shap_importance_{model_name}_{output_prefix}.{extension}"
    _finish_figure(bar_path, save_figures, show_figures, settings)
    paths.append(bar_path)
    return paths


def save_ensemble_shap_results( shap_results: dict[str, dict[str, Any]], output_file: str | Path, save_excel: bool = True, ) -> Path | None:
    """Save ensemble SHAP values and global importance tables to Excel."""
    if not save_excel:
        return None
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file) as writer:
        for model_name, result in shap_results.items():
            safe_name = str(model_name)[:20]
            result["values"].to_excel( writer, sheet_name=f"SHAP_{safe_name}"[:31], index=False, )
            result["importance"].to_excel( writer, sheet_name=f"Importance_{safe_name}"[:31], index=False, )
    return output_file


### 12. Descriptor exploration plots ###
def compute_feature_target_correlations( X: pd.DataFrame, y: pd.Series, method: str = "pearson", ) -> pd.DataFrame:
    """Compute univariate feature-target correlation, R2 and p-value."""
    method = method.lower()
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")
    rows = []
    y_values = np.asarray(y, dtype=float)
    for feature in X.columns:
        x_values = np.asarray(X[feature], dtype=float)
        if np.nanstd(x_values) == 0 or np.nanstd(y_values) == 0:
            corr, p_value = np.nan, np.nan
        else:
            corr, p_value = (pearsonr(x_values, y_values) if method == "pearson" else spearmanr(x_values, y_values))
        rows.append({
            "Feature": feature,
            "Correlation": corr,
            "Abs_Correlation": abs(corr) if pd.notna(corr) else np.nan,
            "Univariate_R2": corr ** 2 if pd.notna(corr) else np.nan,
            "P_Value": p_value,
            "N": len(x_values),
            "Method": method,
        })
    return pd.DataFrame(rows).sort_values("Abs_Correlation", ascending=False).reset_index(drop=True)


def _resolve_feature_list( X: pd.DataFrame, features: list[str] | None, correlations: pd.DataFrame | None = None, top_n: int | None = None, ) -> list[str]:
    if features:
        missing = [f for f in features if f not in X.columns]
        if missing:
            raise KeyError(f"Feature(s) not found: {missing}")
        selected = list(dict.fromkeys(features))
    elif correlations is not None:
        selected = correlations["Feature"].tolist()
    else:
        selected = list(X.columns)
    return selected[:top_n] if top_n is not None else selected


def plot_feature_histograms(
    X: pd.DataFrame,
    results_dir: str | Path,
    output_prefix: str,
    features: list[str] | None = None,
    top_n: int | None = None,
    bins: int = 15,
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
) -> list[Path]:
    settings = _resolve_plot_settings(plot_settings)
    selected = _resolve_feature_list(X, features, top_n=top_n)
    paths = []
    for feature in selected:
        plt.figure(figsize=settings["bar_figure_size"])
        plt.hist(X[feature], bins=bins, color=settings["bar_color"], edgecolor=settings["bar_edge_color"], linewidth=settings["bar_edge_width"])
        plt.xlabel(feature, fontsize=settings["label_size"])
        plt.ylabel("Frequency", fontsize=settings["label_size"])
        plt.title(f"Distribution: {feature}", fontsize=settings["title_size"])
        _style_axes(settings); plt.tight_layout()
        path=Path(results_dir)/f"feature_histogram_{_safe_filename(feature)}_{output_prefix}.{settings['file_format']}"
        _finish_figure(path,save_figures,show_figures,settings); paths.append(path)
    return paths


def plot_feature_target_correlations(
    correlation_table: pd.DataFrame,
    results_dir: str | Path,
    output_prefix: str,
    top_n: int = 15,
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
) -> Path:
    settings=_resolve_plot_settings(plot_settings)
    subset=correlation_table.head(top_n).sort_values("Abs_Correlation")
    height=max(3.0,0.28*len(subset))
    plt.figure(figsize=(settings["bar_figure_size"][0],height))
    bars = plt.barh( subset["Feature"], subset["Correlation"], color=settings["bar_color"], edgecolor=settings["bar_edge_color"], linewidth=settings["bar_edge_width"], )
    plt.axvline(0,color=settings["reference_line_color"],linewidth=settings["line_width"])
    plt.xlabel(f"{subset['Method'].iloc[0].capitalize()} correlation with target",fontsize=settings["label_size"])
    plt.ylabel("Feature",fontsize=settings["label_size"])
    plt.title("Univariate feature-target correlations",fontsize=settings["title_size"])
    _annotate_bar_values( bars, subset["Correlation"], horizontal=True, fmt=".2f", fontsize=settings["tick_size"], )
    _style_axes(settings); plt.tight_layout()
    path=Path(results_dir)/f"feature_target_correlations_{output_prefix}.{settings['file_format']}"
    _finish_figure(path,save_figures,show_figures,settings)
    return path



def plot_feature_target_scatter(
    X: pd.DataFrame,
    y: pd.Series,
    correlation_table: pd.DataFrame,
    results_dir: str | Path,
    output_prefix: str,
    top_n: int = 5,
    features: list[str] | None = None,
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
) -> list[Path]:
    """Plot one descriptor-versus-target scatter per selected feature."""
    settings = _resolve_plot_settings(plot_settings)
    selected = ( _resolve_feature_list(X, features) if features else correlation_table["Feature"].head(top_n).tolist() )
    y_values = np.asarray(y, dtype=float)
    paths = []

    for feature in selected:
        x_values = np.asarray(X[feature], dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        x_plot = x_values[valid]
        y_plot = y_values[valid]

        row = correlation_table[correlation_table["Feature"] == feature]
        if row.empty:
            corr, p_value = pearsonr(x_plot, y_plot)
            univariate_r2 = corr ** 2
        else:
            corr = float(row.iloc[0]["Correlation"])
            p_value = float(row.iloc[0]["P_Value"])
            univariate_r2 = float(row.iloc[0]["Univariate_R2"])

        plt.figure(figsize=settings["parity_figure_size"])
        plt.scatter(
            x_plot, y_plot,
            s=settings["point_size"],
            alpha=settings["point_alpha"],
            color=settings["oof_color"],
            edgecolor=settings["point_edge_color"],
            linewidth=settings["point_edge_width"],
        )

        if len(np.unique(x_plot)) > 1:
            slope, intercept = np.polyfit(x_plot, y_plot, 1)
            x_line = np.linspace(x_plot.min(), x_plot.max(), 100)
            plt.plot( x_line, slope * x_line + intercept, color=settings["reference_line_color"], linewidth=settings["line_width"], )

        plt.text(
            0.05, 0.95,
            f"r = {corr:.3f}\nR² = {univariate_r2:.3f}\np = {p_value:.2g}",
            transform=plt.gca().transAxes,
            va="top",
            fontsize=settings["legend_size"],
        )
        plt.xlabel(feature, fontsize=settings["label_size"])
        plt.ylabel(getattr(y, "name", None) or "Target", fontsize=settings["label_size"])
        plt.title(f"{feature} vs target", fontsize=settings["title_size"])
        _style_axes(settings)
        plt.tight_layout()

        path = Path(results_dir) / ( f"feature_vs_target_{_safe_filename(feature)}_" f"{output_prefix}.{settings['file_format']}" )
        _finish_figure(path, save_figures, show_figures, settings)
        paths.append(path)

    return paths


def _safe_filename(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:120]


def make_feature_pairs( features: list[str], max_pairs: int | None = None, adjacent_only: bool = False, ) -> list[tuple[str,str]]:
    if adjacent_only:
        pairs=list(zip(features[:-1],features[1:]))
    else:
        pairs=[(features[i],features[j]) for i in range(len(features)) for j in range(i+1,len(features))]
    return pairs[:max_pairs] if max_pairs is not None else pairs


def plot_feature_pairs(
    X: pd.DataFrame,
    y: pd.Series,
    feature_pairs: list[tuple[str,str]],
    results_dir: str | Path,
    output_prefix: str,
    bins: int = 15,
    color_by_target: bool = True,
    annotate_ids: Iterable[Any] | None = None,
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
) -> list[Path]:
    settings=_resolve_plot_settings(plot_settings)
    ids=list(annotate_ids) if annotate_ids is not None else None
    paths = []
    for feature_x,feature_y in feature_pairs:
        if feature_x not in X.columns or feature_y not in X.columns:
            raise KeyError(f"Unknown feature pair: {(feature_x,feature_y)}")
        fig,axes=plt.subplots(1,3,figsize=(9,3))
        axes[0].hist(X[feature_x],bins=bins,color=settings["bar_color"],edgecolor=settings["bar_edge_color"],linewidth=settings["bar_edge_width"])
        axes[0].set_xlabel(feature_x,fontsize=settings["label_size"]); axes[0].set_ylabel("Frequency",fontsize=settings["label_size"])
        axes[1].hist(X[feature_y],bins=bins,color=settings["bar_color"],edgecolor=settings["bar_edge_color"],linewidth=settings["bar_edge_width"])
        axes[1].set_xlabel(feature_y,fontsize=settings["label_size"]); axes[1].set_ylabel("Frequency",fontsize=settings["label_size"])
        scatter_kwargs=dict(s=settings["point_size"],alpha=settings["point_alpha"],edgecolor=settings["bar_edge_color"],linewidth=settings["bar_edge_width"])
        if color_by_target:
            sc=axes[2].scatter(X[feature_x],X[feature_y],c=y,cmap="viridis",**scatter_kwargs)
            fig.colorbar(sc,ax=axes[2],label=getattr(y,"name",None) or "Target")
        else:
            axes[2].scatter(X[feature_x],X[feature_y],color=settings["oof_color"],**scatter_kwargs)
        corr,p_value=pearsonr(np.asarray(X[feature_x],float),np.asarray(X[feature_y],float))
        axes[2].text(0.04,0.96,f"r = {corr:.3f}\nR² = {corr**2:.3f}\np = {p_value:.2g}",transform=axes[2].transAxes,va="top",fontsize=settings["legend_size"])
        axes[2].set_xlabel(feature_x,fontsize=settings["label_size"]); axes[2].set_ylabel(feature_y,fontsize=settings["label_size"])
        if ids is not None:
            for i,label in enumerate(ids):
                axes[2].annotate(str(label),(X.iloc[i][feature_x],X.iloc[i][feature_y]),fontsize=max(5,settings["tick_size"]-1))
        fig.suptitle(f"Feature relationship: {feature_x} vs {feature_y}",fontsize=settings["title_size"])
        fig.tight_layout()
        path=Path(results_dir)/f"feature_pair_{_safe_filename(feature_x)}_vs_{_safe_filename(feature_y)}_{output_prefix}.{settings['file_format']}"
        if save_figures:
            path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=settings["dpi"],bbox_inches="tight")
        if show_figures: plt.show()
        plt.close(fig); paths.append(path)
    return paths


def plot_full_ensemble_parity(
    results: dict[str,Any],
    results_dir: str | Path,
    output_prefix: str,
    target_label: str,
    model_names: list[str] | None = None,
    save_figures: bool = True,
    show_figures: bool = True,
    axis_limits: tuple[float,float] | None = None,
    show_prediction_errorbars: bool = True,
    plot_settings: dict[str,Any] | None = None,
) -> list[Path]:
    settings=_resolve_plot_settings(plot_settings); dataframe=results["predictions_full_ensemble"]; model_names=model_names or dataframe["Model"].unique().tolist(); paths=[]
    for model_name in model_names:
        df=dataframe[dataframe["Model"]==model_name].copy().sort_values("True"); r2=r2_score(df["True"],df["Predicted"]); rmse=safe_rmse(df["True"],df["Predicted"]); mae=mean_absolute_error(df["True"],df["Predicted"]); x=df["True"].to_numpy(float); pred=df["Predicted"].to_numpy(float); sd=df["Prediction_STD"].to_numpy(float)
        plt.figure(figsize=settings["parity_figure_size"])
        if show_prediction_errorbars: plt.errorbar(x,pred,yerr=sd,fmt="none",ecolor="#666666",elinewidth=1.5,alpha=0.95,capsize=2,zorder=2,label="±1 SD")
        if len(x)>=3 and np.ptp(x)>0:
            slope,intercept=np.polyfit(x,pred,1); x_fit=np.linspace(x.min(),x.max(),200); y_fit=intercept+slope*x_fit; residuals=pred-(intercept+slope*x); dof=len(x)-2; s_err=np.sqrt(np.sum(residuals**2)/dof); sxx=np.sum((x-x.mean())**2); se_fit=s_err*np.sqrt(1/len(x)+(x_fit-x.mean())**2/sxx); delta=t.ppf(0.975,dof)*se_fit
            plt.fill_between(x_fit,y_fit-delta,y_fit+delta,color=settings["oof_color"],alpha=0.18,label="95% confidence interval",zorder=1); plt.plot(x_fit,y_fit,color=settings["oof_color"],linewidth=settings["line_width"],zorder=2,label="Regression fit")
        plt.scatter(x,pred,s=settings["point_size"],alpha=settings["point_alpha"],color=settings["oof_color"],edgecolor=settings["bar_edge_color"],linewidth=settings["bar_edge_width"],label="Full-data ensemble",zorder=3); low=min(x.min(),pred.min()); high=max(x.max(),pred.max()); plt.plot([low,high],[low,high],color=settings["reference_line_color"],linewidth=settings["line_width"]); plt.xlabel(f"Experimental {target_label}",fontsize=settings["label_size"]); plt.ylabel(f"Predicted {target_label}",fontsize=settings["label_size"]); plt.title(f"Ensemble {model_name}: full training dataset",fontsize=settings["title_size"]); plt.legend(loc="upper center",bbox_to_anchor=(0.5,-0.20),ncol=2,fontsize=settings["legend_size"],frameon=True)
        plt.subplots_adjust(bottom=0.28)
        if axis_limits is not None: plt.xlim(*axis_limits); plt.ylim(*axis_limits)
        if settings["equal_aspect"]: plt.gca().set_aspect("equal",adjustable="box")
        _style_axes(settings); plt.tight_layout(); path=Path(results_dir)/f"full_ensemble_parity_{model_name}_{output_prefix}.{settings['file_format']}"; _finish_figure(path,save_figures,show_figures,settings); paths.append(path)
    return paths

