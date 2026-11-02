#!/usr/bin/env python3
# coding: utf-8

"""Nested cross-validation classification workflow with Optuna.

This module contains the complete classification workflow used in
bo_classification (.ipynb or .py). It deliberately keeps feature filtering,
scaling, feature selection, class-threshold optimization, hyperparameter
optimization and model fitting inside each cross-validation training fold to
avoid data leakage.

Search for the section banners beginning with ``###`` to jump directly to:
1. Data loading and validation
2. Leakage-safe feature filtering
3. Model and feature-selector factories
4. Split-generation strategies and class thresholds
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

from itertools import combinations
from pathlib import Path
import re
from joblib import Parallel, delayed, parallel_backend
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from scipy.stats import pearsonr, spearmanr

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ( accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score, )
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import warnings
import logging
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)

### 1. Data loading and validation ###
def evaluate_classification( y_true: Iterable[int], y_pred: Iterable[int], probabilities: np.ndarray | None = None, n_classes: int = 2, ) -> dict[str, float]:
    """Return the common classification metrics used throughout the workflow."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        try:
            if n_classes == 2:
                metrics["roc_auc"] = float( roc_auc_score(y_true, probabilities[:, 1]) )
            else:
                metrics["roc_auc"] = float( roc_auc_score( y_true, probabilities, multi_class="ovr", average="macro", ) )
        except ValueError:
            metrics["roc_auc"] = np.nan

    return metrics


def load_classification_data(
    excel_file: str | Path,
    sheet_name: str,
    target_column: str,
    id_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
    metadata_columns: list[str] | None = None,
    round_digits: int = 15,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load numeric classification data and retain identifiers/split metadata."""
    id_columns = id_columns or []
    drop_columns = drop_columns or []
    metadata_columns = metadata_columns or []
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' was not found.")
    retained_metadata = []
    for column in [*id_columns, *metadata_columns]:
        if not isinstance(column, str):
            raise TypeError( f"Metadata entries must be column-name strings, got {column!r}." )
        if column in df.columns and column not in retained_metadata:
            retained_metadata.append(column)
    metadata = df[retained_metadata].copy()
    df = df.drop(columns=drop_columns, errors="ignore").dropna( subset=[target_column] )
    X = df.drop( columns=[target_column] + retained_metadata, errors="ignore", ).apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_column], errors="coerce")
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    metadata = metadata.loc[valid].reset_index(drop=True)
    X = X.round(round_digits)
    y = y.round(round_digits)
    if X.empty:
        raise ValueError("No complete numeric rows remain after cleaning.")
    return X, y, metadata

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
def build_model_from_trial(model_name: str, trial: optuna.Trial, n_classes: int, random_state: int = 42, n_jobs: int = 1):
    if model_name == "LogReg":
        return LogisticRegression( C=trial.suggest_float("C", 1e-3, 100, log=True), class_weight="balanced", max_iter=5000, random_state=random_state, )
    if model_name == "SVC":
        return SVC(
            C=trial.suggest_float("C", 1e-3, 100, log=True),
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
            kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]),
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )
    if model_name == "KNN":
        return KNeighborsClassifier( n_neighbors=trial.suggest_int("n_neighbors", 2, 5), weights=trial.suggest_categorical( "weights", ["uniform", "distance"], ), )
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 20, 200),
            max_depth=trial.suggest_int("max_depth", 1, 5),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 8),
            min_samples_split=trial.suggest_int("min_samples_split", 4, 12),
            max_features=trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2"],
            ),
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if model_name == "HistGB":
        return HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float(
                "learning_rate",
                0.01,
                0.5,
                log=True,
            ),
            max_iter=trial.suggest_int("max_iter", 50, 300),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 5, 50),
            l2_regularization=trial.suggest_float(
                "l2_regularization",
                1e-6,
                10,
                log=True,
            ),
            random_state=random_state,
        )
    if model_name == "XGB":
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        model_parameters = {
            "objective": objective,
            "n_estimators": trial.suggest_int("n_estimators", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 1, 5),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.2,
                log=True,
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-4,
                10,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-2,
                100,
                log=True,
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight",
                1,
                20,
            ),
            "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        if n_classes > 2:
            model_parameters["num_class"] = n_classes
        return XGBClassifier(**model_parameters)
    if model_name == "NN":
        return MLPClassifier(
            hidden_layer_sizes=trial.suggest_categorical(
                "hidden_layer_sizes",
                [(5,), (10,), (5, 5), (10, 5)],
            ),
            activation=trial.suggest_categorical(
                "activation",
                ["relu", "tanh"],
            ),
            alpha=trial.suggest_float("alpha", 1e-3, 1, log=True),
            learning_rate_init=trial.suggest_float(
                "learning_rate_init",
                1e-4,
                1e-2,
                log=True,
            ),
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=30,
            max_iter=3000,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model: {model_name}")


def remove_feature_selection_params(params: dict[str, Any]) -> dict[str, Any]:
    fs_params = {
        "feature_selector",
        "selectkbest_k",
        "C_feature",
        "logreg_threshold_feature",
        "rf_threshold_feature",
        "rf_n_estimators_feature",
        "rf_max_depth_feature",
        "rfe_n_features",
        "rfe_C",
    }
    return {key: value for key, value in params.items() if key not in fs_params}


def build_final_model_from_params(model_name: str, params: dict[str, Any], n_classes: int, random_state: int = 42, n_jobs: int = 1):
    params = remove_feature_selection_params(dict(params))
    if model_name == "LogReg":
        return LogisticRegression( **params, class_weight="balanced", max_iter=5000, random_state=random_state, )
    if model_name == "SVC":
        return SVC( **params, probability=True, class_weight="balanced", random_state=random_state, )
    if model_name == "KNN":
        return KNeighborsClassifier(**params)
    if model_name == "RF":
        return RandomForestClassifier( **params, class_weight="balanced", random_state=random_state, n_jobs=n_jobs, )
    if model_name == "HistGB":
        return HistGradientBoostingClassifier( **params, random_state=random_state, )
    if model_name == "XGB":
        model_parameters = {
            **params,
            "objective": (
                "binary:logistic" if n_classes == 2 else "multi:softprob"
            ),
            "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        if n_classes > 2:
            model_parameters["num_class"] = n_classes
        return XGBClassifier(**model_parameters)
    if model_name == "NN":
        return MLPClassifier( **params, early_stopping=True, validation_fraction=0.2, n_iter_no_change=30, max_iter=3000, random_state=random_state, )
    raise ValueError(f"Unknown model: {model_name}")


def build_feature_selector_from_trial(trial: optuna.Trial, max_features_cap: int, min_features: int = 1, random_state: int = 42, rfe_step: int | float = 1, n_jobs: int = 1):
    min_features = max(1, min(int(min_features), int(max_features_cap)))
    selector_name = trial.suggest_categorical( "feature_selector", ["selectkbest", "logreg_feature", "rf_feature", "rfe_feature"], )

    if selector_name == "selectkbest":
        selector = SafeSelectKBest( score_func=f_classif, k=trial.suggest_int( "selectkbest_k", min_features, max_features_cap, ), )
    elif selector_name == "logreg_feature":
        selector = SelectFromModel(
            estimator=LogisticRegression(
                penalty="l1",
                solver="saga",
                C=trial.suggest_float("C_feature", 1e-3, 100, log=True),
                class_weight="balanced",
                max_iter=5000,
                random_state=random_state,
            ),
            threshold=trial.suggest_categorical(
                "logreg_threshold_feature",
                ["mean", "median"],
            ),
            max_features=max_features_cap,
        )
    elif selector_name == "rf_feature":
        selector = SelectFromModel(
            estimator=RandomForestClassifier(
                n_estimators=trial.suggest_int(
                    "rf_n_estimators_feature",
                    20,
                    200,
                ),
                max_depth=trial.suggest_int(
                    "rf_max_depth_feature",
                    1,
                    5,
                ),
                class_weight="balanced",
                random_state=random_state,
                n_jobs=n_jobs,
            ),
            threshold=trial.suggest_categorical(
                "rf_threshold_feature",
                ["mean", "median"],
            ),
            max_features=max_features_cap,
        )
    elif selector_name == "rfe_feature":
        selector = SafeRFE(
            estimator=LogisticRegression(
                C=trial.suggest_float("rfe_C", 1e-3, 100, log=True),
                class_weight="balanced",
                max_iter=5000,
                random_state=random_state,
            ),
            n_features_to_select=trial.suggest_int(
                "rfe_n_features",
                min_features,
                max_features_cap,
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


def get_model_probabilities( fitted_model: Pipeline, X_input: pd.DataFrame, n_classes: int, ) -> np.ndarray:
    """Return a complete probability matrix for any fitted classifier."""
    if hasattr(fitted_model, "predict_proba"):
        probabilities = np.asarray( fitted_model.predict_proba(X_input), dtype=float, )
        if probabilities.shape[1] == n_classes:
            return probabilities

    predictions = np.asarray(fitted_model.predict(X_input), dtype=int)
    probabilities = np.zeros((len(predictions), n_classes), dtype=float)
    probabilities[np.arange(len(predictions)), predictions] = 1.0
    return probabilities


### 4. Split-generation strategies and class thresholds ###

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

def class_labels(n_classes: int) -> list[str]:
    """Return ordered labels for binary, ternary or higher classification."""
    if n_classes == 2:
        return ["Low", "High"]
    if n_classes == 3:
        return ["Low", "Medium", "High"]
    return [f"Class_{index + 1}" for index in range(n_classes)]


def classes_from_thresholds( y_continuous: Iterable[float], thresholds: Iterable[float], ) -> pd.Series:
    """Convert a continuous response into ordered integer classes."""
    y_values = np.asarray(pd.Series(y_continuous), dtype=float)
    sorted_thresholds = np.asarray( sorted(float(value) for value in thresholds), dtype=float, )
    return pd.Series( np.digitize(y_values, sorted_thresholds, right=True), dtype=int, )


def candidate_threshold_sets(
    threshold_mode: str,
    threshold_candidates: list[float] | None,
    threshold_quantiles: list[float] | None,
    n_classes: int,
    min_quantile_gap: float = 0.05,
) -> list[tuple[float, ...]]:
    """Generate valid numerical-threshold or quantile combinations."""
    threshold_mode = threshold_mode.lower()
    if threshold_mode not in {"fixed-grid", "quantiles"}:
        raise ValueError( "threshold_mode must be either 'fixed-grid' or 'quantiles'." )

    n_thresholds = int(n_classes) - 1
    pool = ( threshold_candidates if threshold_mode == "fixed-grid" else threshold_quantiles )
    if not pool:
        raise ValueError( f"Provide candidates for threshold_mode={threshold_mode!r}." )

    pool = sorted(set(float(value) for value in pool))
    if threshold_mode == "quantiles":
        invalid_quantiles = [value for value in pool if not 0 < value < 1]
        if invalid_quantiles:
            raise ValueError("Quantiles must be strictly between 0 and 1.")

    candidate_sets = []
    for values in combinations(pool, n_thresholds):
        if ( threshold_mode == "quantiles" and len(values) > 1 and np.min(np.diff(values)) < min_quantile_gap ):
            continue
        candidate_sets.append(tuple(values))

    if not candidate_sets:
        raise ValueError("No valid threshold combination could be generated.")
    return candidate_sets


def valid_class_counts( y_classes: Iterable[int], n_classes: int, min_class_count: int, ) -> bool:
    """Check that every ordered class contains enough training observations."""
    counts = ( pd.Series(y_classes) .value_counts() .reindex(range(n_classes), fill_value=0) )
    return bool((counts >= int(min_class_count)).all())

def make_classification_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_classes: int,
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
    """Build configurable outer or inner classification splits."""
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
            raise ValueError("A group split requires group labels.")
        if pd.Series(groups).isna().any():
            raise ValueError("Group labels contain missing values.")
        return list( _repeated_group_splits( groups=groups, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state, ) )

    if strategy == "leave_group_out":
        if groups is None or not held_out_groups:
            raise ValueError( "leave_group_out requires split_group_columns and " "held_out_groups." )
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
    model_name, X_train_outer, y_train_cont_outer, X_test_outer, y_test_cont_outer,
    threshold_mode, threshold_candidates, threshold_quantiles, n_classes,
    inner_cv_splits, inner_cv_repeats, max_features_cap, min_features,
    corr_threshold, n_trials, overfit_penalty_weight, optimize_metric, random_state,
    inner_split_strategy="y_stratified", inner_groups=None, y_stratify_bins=5,
    min_class_count=5, min_quantile_gap=0.05, rfe_step=1, one_se=False, n_jobs=1,
):
    # The inner splits are created only from the current outer-training set.
    # They are used exclusively for Optuna model/selector/threshold selection.
    X_train_outer = X_train_outer.reset_index(drop=True)
    X_test_outer = X_test_outer.reset_index(drop=True)
    y_train_cont_outer = pd.Series(y_train_cont_outer).reset_index(drop=True).astype(float)
    y_test_cont_outer = pd.Series(y_test_cont_outer).reset_index(drop=True).astype(float)
    candidates = candidate_threshold_sets( threshold_mode, threshold_candidates, threshold_quantiles, n_classes, min_quantile_gap )
    inner_splits = make_classification_splits(
        X=X_train_outer, y=y_train_cont_outer, n_classes=n_classes,
        n_splits=inner_cv_splits, n_repeats=inner_cv_repeats,
        random_state=random_state, strategy=inner_split_strategy,
        groups=inner_groups, y_stratify_bins=y_stratify_bins,
    )
    candidate_results = []
    for candidate_index, candidate in enumerate(candidates):
        def objective(trial):
            # One Optuna trial defines both a feature-selection strategy and the
            # model hyperparameters. Thresholds are evaluated inside every inner
            # training fold so preprocessing never sees validation observations.
            selector, _ = build_feature_selector_from_trial(trial, max_features_cap, min_features, random_state, rfe_step, n_jobs)
            model = build_model_from_trial(model_name, trial, n_classes, random_state)
            pipe = Pipeline([ ("correlation_filter", CorrelationFilter(corr_threshold)), ("scaler", StandardScaler()), ("selector", selector), ("model", model), ])
            train_scores, validation_scores, feature_counts = [], [], []
            for train_index, validation_index in inner_splits:
                X_train = X_train_outer.iloc[train_index]
                X_validation = X_train_outer.iloc[validation_index]
                y_train_cont = y_train_cont_outer.iloc[train_index]
                y_validation_cont = y_train_cont_outer.iloc[validation_index]
                thresholds = ( list(candidate) if threshold_mode == "fixed-grid" else [float(y_train_cont.quantile(q)) for q in candidate] )
                if len(set(np.round(thresholds, 12))) != len(thresholds):
                    return -1e9
                y_train = classes_from_thresholds(y_train_cont, thresholds)
                y_validation = classes_from_thresholds(y_validation_cont, thresholds)
                try:
                    pipe.fit(X_train, y_train)
                    feature_counts.append(len(get_selected_feature_names(pipe)))
                    train_scores.append( evaluate_classification(y_train, pipe.predict(X_train))[optimize_metric] )
                    validation_scores.append( evaluate_classification(y_validation, pipe.predict(X_validation))[optimize_metric] )
                except Exception:
                    return -1e9
            mean_train = float(np.nanmean(train_scores))
            mean_validation = float(np.nanmean(validation_scores))
            if np.isnan(mean_validation):
                return -1e9
            overfit_penalty = max(0.0, mean_train - mean_validation)
            trial.set_user_attr("n_features", float(np.mean(feature_counts)))
            trial.set_user_attr("validation_mean", mean_validation)
            trial.set_user_attr("validation_se", float(np.std(validation_scores, ddof=1) / np.sqrt(len(validation_scores))) if len(validation_scores) > 1 else 0.0)
            return mean_validation - overfit_penalty_weight * overfit_penalty
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state + candidate_index))
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
        if study.best_value > -1e10:
            chosen_trial = study.best_trial
            if one_se:
                valid_trials = [t for t in study.trials if "validation_mean" in t.user_attrs and "n_features" in t.user_attrs]
                if valid_trials:
                    best_validation_trial = max(valid_trials, key=lambda t: t.user_attrs["validation_mean"])
                    cutoff = best_validation_trial.user_attrs["validation_mean"] - best_validation_trial.user_attrs.get("validation_se", 0.0)
                    eligible = [t for t in valid_trials if t.user_attrs["validation_mean"] >= cutoff]
                    chosen_trial = min(eligible, key=lambda t: (t.user_attrs["n_features"], -t.user_attrs["validation_mean"]))
            candidate_results.append({
                "candidate_values": tuple(candidate),
                "best_inner_score": float(chosen_trial.value),
                "best_params": dict(chosen_trial.params),
                "best_trial_features": chosen_trial.user_attrs.get("n_features"),
            })
    best = max(candidate_results, key=lambda row: row["best_inner_score"])
    best_quantiles = list(best["candidate_values"]) if threshold_mode == "quantiles" else []
    best_thresholds = ( list(best["candidate_values"]) if threshold_mode == "fixed-grid" else [float(y_train_cont_outer.quantile(q)) for q in best_quantiles] )
    y_train = classes_from_thresholds(y_train_cont_outer, best_thresholds)
    y_test = classes_from_thresholds(y_test_cont_outer, best_thresholds)
    fixed_trial = optuna.trial.FixedTrial(best["best_params"])
    final_selector, selector_name = build_feature_selector_from_trial(fixed_trial, max_features_cap, min_features, random_state, rfe_step, n_jobs)
    final_model = build_final_model_from_params(model_name, best["best_params"], n_classes, random_state, n_jobs)
    final_pipe = Pipeline([ ("correlation_filter", CorrelationFilter(corr_threshold)), ("scaler", StandardScaler()), ("selector", final_selector), ("model", final_model), ])
    # After Optuna chooses the best trial, refit that complete pipeline on all
    # observations available in the current outer-training fold.
    final_pipe.fit(X_train_outer, y_train)
    pred_train = final_pipe.predict(X_train_outer)
    pred_test = final_pipe.predict(X_test_outer)
    proba_train = get_model_probabilities(final_pipe, X_train_outer, n_classes)
    proba_test = get_model_probabilities(final_pipe, X_test_outer, n_classes)
    return {
        "model": final_pipe, "best_thresholds": best_thresholds,
        "best_threshold_quantiles": best_quantiles,
        "best_inner_score": best["best_inner_score"],
        "candidate_scores": candidate_results, "best_params": best["best_params"],
        "feature_selection_method": selector_name,
        "selected_features": get_selected_feature_names(final_pipe),
        "n_features_after_correlation_filter": len(
            final_pipe.named_steps["correlation_filter"].selected_features_
        ),
        "pred_train": pred_train, "pred_test": pred_test,
        "proba_train": proba_train, "proba_test": proba_test,
        "y_train_classes": y_train.to_numpy(), "y_test_classes": y_test.to_numpy(),
        "metrics_train": evaluate_classification(y_train, pred_train, proba_train, n_classes),
        "metrics_test": evaluate_classification(y_test, pred_test, proba_test, n_classes),
    }

### 6. Repeated nested-CV orchestration ###
class EnsembleModel:
    """Average class probabilities from all outer-fold pipelines of one model family."""
    def __init__(self, members, class_names):
        self.members = members
        self.models = [member["pipeline"] for member in members]
        self.class_names = class_names
    def predict_proba_members(self, X_input):
        return np.stack([ get_model_probabilities(member["pipeline"], X_input, len(self.class_names)) for member in self.members ], axis=0)
    def predict_proba(self, X_input):
        return self.predict_proba_members(X_input).mean(axis=0)
    def predict(self, X_input):
        return self.predict_proba(X_input).argmax(axis=1)



def refit_full_training_ensembles( cv_models_dict: dict[str, EnsembleModel], X: pd.DataFrame, y: pd.Series | Iterable[float], ) -> dict[str, EnsembleModel]:
    """Refit every selected outer-fold pipeline on the complete dataset.

    Nested-CV models remain reserved for unbiased OOF evaluation. The returned
    ensembles are the deployment models used for full-dataset visualization,
    external testing and virtual screening. Each member retains the threshold
    set and hyperparameters selected in its corresponding outer fold, but its
    pipeline is refitted using every available training observation.
    """
    X_full = X.reset_index(drop=True)
    y_full = pd.Series(y).reset_index(drop=True).astype(float)
    full_models: dict[str, EnsembleModel] = {}

    for model_name, cv_ensemble in cv_models_dict.items():
        full_members = []
        for member in cv_ensemble.members:
            thresholds = list(member["thresholds"])
            y_classes = classes_from_thresholds(y_full, thresholds)
            pipeline = clone(member["pipeline"])
            pipeline.fit(X_full, y_classes)
            full_members.append({ **member, "pipeline": pipeline, "training_scope": "full_dataset", })
        full_models[model_name] = EnsembleModel( full_members, cv_ensemble.class_names, )

    return full_models

def _majority_vote(values: Iterable[int]) -> int:
    """Return the smallest class in case of an exact voting tie."""
    counts = np.bincount(np.asarray(list(values), dtype=int))
    return int(np.flatnonzero(counts == counts.max())[0])


def build_composite_groups( metadata: pd.DataFrame, group_columns: list[str] | None, separator: str = " | ", ) -> pd.Series | None:
    """Build one group label from one or several metadata columns."""
    if not group_columns:
        return None
    missing = [column for column in group_columns if column not in metadata.columns]
    if missing:
        raise KeyError(f"Group column(s) not found in metadata: {missing}")
    if len(group_columns) == 1:
        return metadata[group_columns[0]].astype(str)
    return metadata[group_columns].astype(str).agg(separator.join, axis=1)


def run_nested_classification(
    X, y, model_names, n_classes, threshold_mode, threshold_candidates,
    threshold_quantiles, outer_n_splits, outer_n_repeats, inner_cv_splits,
    inner_cv_repeats, max_features_cap, min_features, corr_threshold, n_trials,
    overfit_penalty_weight, optimize_metric="f1_macro", min_class_count=5,
    min_quantile_gap=0.05, random_state=42, metadata=None, verbose=True,
    outer_split_strategy="y_stratified", inner_split_strategy="y_stratified",
    split_group_columns=None, y_stratify_bins=5, ks_test_fraction=0.20,
    held_out_groups=None, predefined_split_column=None,
    predefined_train_label="train", export_split_indices=False,
    export_test_groups=True, max_test_groups_to_export=50, rfe_step=1, n_jobs=1, one_se=False,
):
    """Run nested CV with configurable outer and inner split strategies."""
    max_features_cap = min(max_features_cap, X.shape[1])
    min_features = max(1, min(int(min_features), max_features_cap))
    metadata = (metadata.reset_index(drop=True) if metadata is not None else pd.DataFrame(index=np.arange(len(X))))
    groups = build_composite_groups(metadata, split_group_columns)
    predefined_labels = None
    if predefined_split_column is not None:
        if predefined_split_column not in metadata.columns:
            raise KeyError(f"Predefined split column {predefined_split_column!r} was not retained.")
        predefined_labels = metadata[predefined_split_column].reset_index(drop=True)
    # Outer splits estimate generalization performance. Every observation in an
    # outer validation set remains unseen by tuning and fitting for that fold.
    outer_splits = make_classification_splits(
        X=X, y=y, n_classes=n_classes, n_splits=outer_n_splits,
        n_repeats=outer_n_repeats, random_state=random_state,
        strategy=outer_split_strategy, groups=groups,
        y_stratify_bins=y_stratify_bins, ks_test_fraction=ks_test_fraction,
        held_out_groups=held_out_groups, predefined_labels=predefined_labels,
        predefined_train_label=predefined_train_label,
    )
    predictions, params, selected, candidates, fold_rows = [], [], [], [], []
    split_manifest, split_assignments, models_dict = [], [], {}
    for fold_number, (train_index, test_index) in enumerate(outer_splits, 1):
        for sample_index in train_index:
            split_assignments.append({ "Fold": fold_number, "Sample_ID": int(sample_index), "Set": "Training", "Target": float(y.iloc[int(sample_index)]), })
        for sample_index in test_index:
            split_assignments.append({ "Fold": fold_number, "Sample_ID": int(sample_index), "Set": "Validation", "Target": float(y.iloc[int(sample_index)]), })
        row = { "Fold": fold_number, "Outer_Split_Strategy": outer_split_strategy, "N_Train": len(train_index), "N_Test": len(test_index), }
        if export_split_indices:
            row["Train_Indices"] = ", ".join(map(str, train_index))
            row["Test_Indices"] = ", ".join(map(str, test_index))
        if groups is not None:
            train_groups = pd.unique(groups.iloc[train_index])
            test_groups = pd.unique(groups.iloc[test_index])
            row["N_Train_Groups"] = len(train_groups)
            row["N_Test_Groups"] = len(test_groups)
            if export_test_groups:
                row["Test_Groups"] = ", ".join(map(str, test_groups[:max_test_groups_to_export]))
                row["Test_Groups_Truncated"] = len(test_groups) > max_test_groups_to_export
        if predefined_labels is not None:
            row["Test_Predefined_Labels"] = ", ".join( map(str, pd.unique(predefined_labels.iloc[test_index])) )
        split_manifest.append(row)
    labels = class_labels(n_classes)
    def _fit_outer(model_name, fold, train_index, test_index):
        inner_groups = (groups.iloc[train_index].reset_index(drop=True) if groups is not None and inner_split_strategy in {"group", "cluster"} else None)
        if inner_split_strategy in {"ks", "ks_holdout", "predefined", "leave_group_out"}:
            raise ValueError(f"{inner_split_strategy!r} is not supported for inner CV")
        return train_outer_fold(
            model_name=model_name,
            X_train_outer=X.iloc[train_index].reset_index(drop=True),
            y_train_cont_outer=y.iloc[train_index].reset_index(drop=True),
            X_test_outer=X.iloc[test_index].reset_index(drop=True),
            y_test_cont_outer=y.iloc[test_index].reset_index(drop=True),
            threshold_mode=threshold_mode, threshold_candidates=threshold_candidates,
            threshold_quantiles=threshold_quantiles, n_classes=n_classes,
            inner_cv_splits=inner_cv_splits, inner_cv_repeats=inner_cv_repeats,
            max_features_cap=max_features_cap, min_features=min_features,
            corr_threshold=corr_threshold, n_trials=n_trials,
            overfit_penalty_weight=overfit_penalty_weight,
            optimize_metric=optimize_metric, random_state=random_state + fold,
            inner_split_strategy=inner_split_strategy, inner_groups=inner_groups,
            y_stratify_bins=y_stratify_bins, min_class_count=min_class_count,
            min_quantile_gap=min_quantile_gap, rfe_step=rfe_step, one_se=one_se, n_jobs=1)

    for model_name in model_names:
        if verbose:
            print(f"\
Starting {model_name}")
        members = []
        if n_jobs > 1:
            with parallel_backend("loky", inner_max_num_threads=1):
                fold_results = Parallel(n_jobs=min(n_jobs, len(outer_splits)))( delayed(_fit_outer)(model_name, fold, tr, te) for fold, (tr, te) in enumerate(outer_splits, 1))
        else:
            fold_results = None
        for fold, (train_index, test_index) in enumerate(outer_splits, 1):
            if verbose:
                print(f"  Fold {fold}/{len(outer_splits)}")
            inner_groups = ( groups.iloc[train_index].reset_index(drop=True) if groups is not None and inner_split_strategy in {"group", "cluster"} else None )
            if inner_split_strategy in {"ks", "ks_holdout", "predefined", "leave_group_out"}:
                raise ValueError(f"{inner_split_strategy!r} is not supported for inner CV.")
            # Tune on the inner CV and evaluate once on this outer validation
            # fold. The returned pipeline becomes one member of the ensemble.
            result = (fold_results[fold - 1] if fold_results is not None else _fit_outer(model_name, fold, train_index, test_index))
            for local_index, original_index in enumerate(test_index):
                row = {
                    "Model": model_name, "Fold": fold,
                    "Sample_ID": int(original_index),
                    "Target": float(y.iloc[original_index]),
                    "True_encoded": int(result["y_test_classes"][local_index]),
                    "Predicted_encoded": int(result["pred_test"][local_index]),
                    "True_label": labels[int(result["y_test_classes"][local_index])],
                    "Predicted_label": labels[int(result["pred_test"][local_index])],
                    "Correct": int(result["y_test_classes"][local_index] == result["pred_test"][local_index]),
                    "Thresholds": ", ".join(map(str, result["best_thresholds"])),
                    "Threshold_quantiles": ", ".join(map(str, result["best_threshold_quantiles"])),
                }
                for class_index, label in enumerate(labels):
                    row[f"Prob_{label}"] = float(result["proba_test"][local_index, class_index])
                for column in metadata.columns:
                    row[column] = metadata.iloc[original_index][column]
                predictions.append(row)
            metric_row = {
                "Model": model_name, "Fold": fold,
                "Outer_Split_Strategy": outer_split_strategy,
                "Inner_Split_Strategy": inner_split_strategy,
                "Optimize_Metric": optimize_metric,
                "Best_inner_score": result["best_inner_score"],
                "Thresholds": ", ".join(map(str, result["best_thresholds"])),
                "Threshold_quantiles": ", ".join(map(str, result["best_threshold_quantiles"])),
                "Feature_Selection_Method": result["feature_selection_method"],
                "N_Features_After_Correlation_Filter": result["n_features_after_correlation_filter"],
                "Params": str(result["best_params"]),
                "Features": ", ".join(result["selected_features"]),
            }
            metric_row.update({f"train_{key}": value for key, value in result["metrics_train"].items()})
            metric_row.update({f"test_{key}": value for key, value in result["metrics_test"].items()})
            params.append(metric_row)
            fold_rows.append({"Model": model_name, "Fold": fold, **result["metrics_test"]})
            for feature in result["selected_features"]:
                selected.append({ "Model": model_name, "Fold": fold, "Feature_Selection_Method": result["feature_selection_method"], "Feature": feature, })
            for candidate in result["candidate_scores"]:
                candidates.append({ "Model": model_name, "Fold": fold, "Candidate": str(candidate["candidate_values"]), "Best_inner_score": candidate["best_inner_score"], })
            members.append({ "pipeline": result["model"], "thresholds": result["best_thresholds"], "quantiles": result["best_threshold_quantiles"], })
        models_dict[model_name] = EnsembleModel(members, labels)
    results = _build_result_tables( models_dict, predictions, params, selected, candidates, fold_rows, n_classes, optimize_metric )
    results["split_manifest"] = pd.DataFrame(split_manifest)
    results["split_assignments"] = pd.DataFrame(split_assignments)
    results["configuration_labels"] = labels
    full_models_dict = refit_full_training_ensembles(models_dict, X, y)
    results["full_models_dict"] = full_models_dict
    full_rows = []
    for model_name, ensemble in full_models_dict.items():
        member_probabilities = ensemble.predict_proba_members(X)
        mean_probabilities = member_probabilities.mean(axis=0)
        std_probabilities = member_probabilities.std(axis=0)
        predictions_full = mean_probabilities.argmax(axis=1)
        member_true_classes = np.column_stack([ classes_from_thresholds(y, member["thresholds"]).to_numpy() for member in ensemble.members ])
        true_full = np.apply_along_axis(_majority_vote, 1, member_true_classes)
        for sample_id in range(len(X)):
            row = {
                "Model": model_name, "Sample_ID": sample_id,
                "Target": float(y.iloc[sample_id]),
                "True_encoded": int(true_full[sample_id]),
                "Predicted_encoded": int(predictions_full[sample_id]),
                "True_label": labels[int(true_full[sample_id])],
                "Predicted_label": labels[int(predictions_full[sample_id])],
                "Correct": int(true_full[sample_id] == predictions_full[sample_id]),
                "Prediction_STD": float(std_probabilities[sample_id].max()),
            }
            for class_index, label in enumerate(labels):
                row[f"Prob_{label}"] = float(mean_probabilities[sample_id, class_index])
                row[f"Prob_STD_{label}"] = float(std_probabilities[sample_id, class_index])
            for column in metadata.columns:
                row[column] = metadata.iloc[sample_id][column]
            full_rows.append(row)
    results["predictions_full_ensemble"] = pd.DataFrame(full_rows)
    return results

### 7. Result aggregation and Excel export ###
def _build_result_tables( models_dict, predictions, params, selected, candidates, folds, n_classes, optimize_metric ):
    df_predictions_all = pd.DataFrame(predictions)
    df_params = pd.DataFrame(params)
    df_folds = pd.DataFrame(folds)
    metric_columns = [column for column in ["accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc"] if column in df_folds.columns]
    nested_rows = []
    for model_name, subset in df_folds.groupby("Model"):
        row = {"Model": model_name}
        for metric in metric_columns:
            row[f"Mean_{metric}_CV"] = subset[metric].mean()
            row[f"Std_{metric}_CV"] = subset[metric].std()
        nested_rows.append(row)
    optimize_metric = str(optimize_metric).lower()
    nested_sort_column = f"Mean_{optimize_metric}_CV"
    if nested_sort_column not in pd.DataFrame(nested_rows).columns:
        raise KeyError(f"Optimization metric is unavailable in summary: {optimize_metric}")
    df_summary_nested = pd.DataFrame(nested_rows).sort_values( nested_sort_column, ascending=False )
    df_summary_nested.insert(1, "Model_selection_metric", optimize_metric)
    labels = class_labels(n_classes)
    mean_rows = []
    for (model_name, sample_id), subset in df_predictions_all.groupby(["Model", "Sample_ID"]):
        row = { "Model": model_name, "Sample_ID": sample_id, "Target": subset["Target"].mean(), "True_encoded": _majority_vote(subset["True_encoded"]), }
        probabilities = np.asarray([subset[f"Prob_{label}"].mean() for label in labels])
        row["Predicted_encoded"] = int(np.argmax(probabilities))
        row["True_label"] = labels[row["True_encoded"]]
        row["Predicted_label"] = labels[row["Predicted_encoded"]]
        row["Correct"] = int(row["True_encoded"] == row["Predicted_encoded"])
        for index, label in enumerate(labels):
            row[f"Prob_{label}"] = float(probabilities[index])
        mean_rows.append(row)
    df_predictions_mean = pd.DataFrame(mean_rows)
    summary_mean_rows = []
    for model_name, subset in df_predictions_mean.groupby("Model"):
        probability_columns = [f"Prob_{label}" for label in labels]
        metrics = evaluate_classification( subset["True_encoded"], subset["Predicted_encoded"], subset[probability_columns].to_numpy(), n_classes )
        summary_mean_rows.append({"Model": model_name, **{f"{key}_mean_prediction": value for key, value in metrics.items()}})
    mean_sort_column = f"{optimize_metric}_mean_prediction"
    df_summary_mean = pd.DataFrame(summary_mean_rows)
    if mean_sort_column not in df_summary_mean.columns:
        raise KeyError(f"Optimization metric is unavailable in mean summary: {optimize_metric}")
    df_summary_mean = df_summary_mean.sort_values( mean_sort_column, ascending=False )
    df_summary_mean.insert(1, "Model_selection_metric", optimize_metric)
    train_columns = [column for column in df_params.columns if column.startswith("train_")]
    test_columns = [column for column in df_params.columns if column.startswith("test_")]
    df_train_mean = df_params.groupby("Model")[train_columns].mean().reset_index()
    df_test_mean = df_params.groupby("Model")[test_columns].mean().reset_index()
    df_selected = pd.DataFrame(selected)
    n_folds = max(1, df_params["Fold"].nunique())
    if df_selected.empty:
        feature_frequency = pd.DataFrame()
        feature_frequency_by_selector = pd.DataFrame()
    else:
        feature_frequency = (df_selected.groupby(["Model", "Feature"]).size() .reset_index(name="Count") .sort_values(["Model", "Count"], ascending=[True, False]))
        feature_frequency["Selection_Frequency"] = feature_frequency["Count"] / n_folds
        feature_frequency_by_selector = (
            df_selected.groupby(["Model", "Feature_Selection_Method", "Feature"]).size()
            .reset_index(name="Count")
            .sort_values(["Model", "Feature_Selection_Method", "Count"], ascending=[True, True, False])
        )
        feature_frequency_by_selector["Selection_Frequency"] = feature_frequency_by_selector["Count"] / n_folds
    selector_frequency = (df_params.groupby(["Model", "Feature_Selection_Method"]).size() .reset_index(name="Count"))
    selector_frequency["Selection_Method_Frequency"] = selector_frequency["Count"] / n_folds
    representative_features = {}
    if not feature_frequency.empty:
        for model_name in feature_frequency["Model"].unique():
            representative_features[model_name] = (
                feature_frequency[feature_frequency["Model"] == model_name]
                .sort_values(["Count", "Feature"], ascending=[False, True])["Feature"].tolist()
            )
    threshold_columns = [ column for column in [ "Model", "Fold", "Thresholds", "Threshold_quantiles", "Best_inner_score" ] if column in df_params.columns ]
    threshold_summary = df_params[threshold_columns].copy()

    return {
        "models_dict": models_dict, "summary_nested": df_summary_nested,
        "summary_mean": df_summary_mean, "summary": df_summary_mean,
        "fold_summary": df_folds, "predictions_all": df_predictions_all,
        "predictions_mean": df_predictions_mean, "params": df_params,
        "train_mean": df_train_mean, "test_mean": df_test_mean,
        "candidate_inner_scores": pd.DataFrame(candidates),
        "threshold_summary": threshold_summary,
        "selected_features": df_selected, "feature_frequency": feature_frequency,
        "feature_frequency_by_selector": feature_frequency_by_selector,
        "selector_frequency": selector_frequency,
        "representative_features": representative_features,
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
            "candidate_inner_scores": "Candidate_scores",
            "threshold_summary": "Threshold_summary",
            "feature_frequency": "Feature_frequency",
            "feature_frequency_by_selector": "Feature_freq_by_selector",
            "selector_frequency": "Selector_frequency",
            "split_manifest": "Split_manifest",
            "split_assignments": "Split_assignments",
            "predictions_full_ensemble": "Predictions_full_ensemble",
        }

        for result_key, sheet_name in sheet_map.items():
            dataframe = results.get(result_key)
            if isinstance(dataframe, pd.DataFrame):
                dataframe.to_excel( writer, sheet_name=sheet_name, index=False, )

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
    output_prefix: str = "classification",
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

def plot_model_metric_comparison(
    results: dict[str, Any],
    results_dir: str | Path,
    output_prefix: str,
    metric: str = "f1_macro",
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
):
    """Compare model families using averaged OOF classification metrics."""
    settings = _resolve_plot_settings(plot_settings)
    metric = metric.lower()
    allowed_metrics = { "accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc", }
    if metric not in allowed_metrics:
        raise ValueError( "metric must be one of: accuracy, balanced_accuracy, " "f1_macro, mcc, roc_auc." )

    metric_column = f"{metric}_mean_prediction"
    if metric_column not in results["summary_mean"].columns:
        raise KeyError( f"Metric column '{metric_column}' is not available." )
    summary = results["summary_mean"].sort_values( metric_column, ascending=False, )

    plt.figure(figsize=settings["bar_figure_size"])
    bars = plt.bar( summary["Model"], summary[metric_column], color=settings["bar_color"], edgecolor=settings["bar_edge_color"], linewidth=settings["bar_edge_width"], )
    metric_label = metric.replace("_", " ").title()
    plt.ylabel(f"OOF {metric_label}", fontsize=settings["label_size"])
    plt.xlabel("Model", fontsize=settings["label_size"])
    plt.title( f"Model comparison: OOF {metric_label}", fontsize=settings["title_size"], )
    plt.xticks( rotation=45, ha="right", fontsize=settings["tick_size"], )

    _annotate_bar_values( bars, summary[metric_column], fmt=".2f", fontsize=settings["tick_size"], )

    _style_axes(settings)
    plt.tight_layout()
    extension = settings["file_format"]
    path = Path(results_dir) / ( f"model_comparison_{metric}_{output_prefix}.{extension}" )
    _finish_figure( path, save_figures, show_figures, plot_settings=settings, )
    return path

def plot_overfitting_gap(
    results: dict[str, Any],
    results_dir: str | Path,
    output_prefix: str,
    metric: str = "f1_macro",
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
):
    """Plot the mean outer-fold training-to-test performance gap."""
    settings = _resolve_plot_settings(plot_settings)
    metric = metric.lower()
    allowed_metrics = { "accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc", }
    if metric not in allowed_metrics:
        raise ValueError( "metric must be one of: accuracy, balanced_accuracy, " "f1_macro, mcc, roc_auc." )

    train_column = f"train_{metric}"
    test_column = f"test_{metric}"
    train = results["train_mean"][["Model", train_column]]
    test = results["test_mean"][["Model", test_column]]
    comparison = train.merge(test, on="Model", how="inner")
    comparison["Metric_gap"] = ( comparison[train_column] - comparison[test_column] )
    comparison = comparison.sort_values("Metric_gap", ascending=False)

    plt.figure(figsize=settings["bar_figure_size"])
    bars = plt.bar( comparison["Model"], comparison["Metric_gap"], color=settings["bar_color"], edgecolor=settings["bar_edge_color"], linewidth=settings["bar_edge_width"], )
    plt.axhline( 0, linewidth=settings["line_width"], color=settings["reference_line_color"], )
    metric_label = metric.replace("_", " ").title()
    plt.ylabel( f"Training-to-test degradation in {metric_label}", fontsize=settings["label_size"], )
    plt.xlabel("Model", fontsize=settings["label_size"])
    plt.title("Train–test performance gap", fontsize=settings["title_size"])
    plt.xticks( rotation=45, ha="right", fontsize=settings["tick_size"], )

    _annotate_bar_values( bars, comparison["Metric_gap"], fmt=".2f", fontsize=settings["tick_size"], )

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
    """Compare averaged OOF high-class probabilities from two models."""
    settings = _resolve_plot_settings(plot_settings)
    highest_class = results["configuration_labels"][-1]
    probability_column = f"Prob_{highest_class}"
    predictions = results["predictions_mean"]

    x_values = predictions[predictions["Model"] == model_x][ ["Sample_ID", probability_column] ].rename(columns={probability_column: f"Probability_{model_x}"})
    y_values = predictions[predictions["Model"] == model_y][ ["Sample_ID", probability_column] ].rename(columns={probability_column: f"Probability_{model_y}"})

    merged = x_values.merge(y_values, on="Sample_ID", how="inner")
    if merged.empty:
        raise ValueError( f"No common predictions found for {model_x} and {model_y}." )

    x_column = f"Probability_{model_x}"
    y_column = f"Probability_{model_y}"

    plt.figure(figsize=settings["parity_figure_size"])
    plt.scatter(
        merged[x_column],
        merged[y_column],
        s=settings["point_size"],
        alpha=settings["point_alpha"],
        color=settings["oof_color"],
        edgecolor=settings["point_edge_color"],
        linewidth=settings["point_edge_width"],
    )
    plt.plot( [0, 1], [0, 1], linewidth=settings["line_width"], color=settings["reference_line_color"], )
    plt.xlabel( f"{model_x} OOF probability: {highest_class}", fontsize=settings["label_size"], )
    plt.ylabel( f"{model_y} OOF probability: {highest_class}", fontsize=settings["label_size"], )
    plt.title( f"{model_x} vs {model_y}", fontsize=settings["title_size"], )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    if settings["equal_aspect"]:
        plt.gca().set_aspect("equal", adjustable="box")
    _style_axes(settings)
    plt.tight_layout()

    extension = settings["file_format"]
    path = Path(results_dir) / ( f"model_prediction_comparison_{model_x}_vs_{model_y}_" f"{output_prefix}.{extension}" )
    _finish_figure(path, save_figures, show_figures, settings)
    return path


def plot_threshold_distribution(results, results_dir, output_prefix, save_figures=True, show_figures=True, plot_settings=None):
    settings=_resolve_plot_settings(plot_settings); rows=[]
    for _,r in results["params"].iterrows():
        for i,t in enumerate(str(r["Thresholds"]).split(","),1):
            rows.append({"Model":r["Model"],"Threshold_number":i,"Threshold":float(t)})
    long=pd.DataFrame(rows); plt.figure(figsize=settings["bar_figure_size"])
    groups=[g["Threshold"].values for _,g in long.groupby("Model")]; labels=[k for k,_ in long.groupby("Model")]
    plt.boxplot(groups,tick_labels=labels); plt.ylabel("Selected numerical threshold",fontsize=settings["label_size"])
    plt.title("Threshold stability across outer folds",fontsize=settings["title_size"]); plt.xticks(rotation=45,ha="right"); plt.tight_layout()
    path=Path(results_dir)/f"threshold_distribution_{output_prefix}.{settings['file_format']}"
    _finish_figure(path,save_figures,show_figures,settings); return path


def plot_confusion_matrices(results, results_dir, output_prefix, normalize=True, save_figures=True, show_figures=True, plot_settings=None):
    settings=_resolve_plot_settings(plot_settings); paths=[]; labels=results["configuration_labels"]
    for model in results["predictions_all"]["Model"].unique():
        df=results["predictions_all"][results["predictions_all"]["Model"]==model]
        cm=confusion_matrix(df["True_encoded"],df["Predicted_encoded"],labels=range(len(labels)),normalize="true" if normalize else None)
        if normalize:
            cm_to_plot = cm * 100
            annot_fmt = "{:.1f}%"
        else:
            cm_to_plot = cm
            annot_fmt = "{:d}"
        plt.figure(figsize=settings["parity_figure_size"]); im=plt.imshow(cm_to_plot,cmap="Blues", vmin=0, vmax=100 if normalize else None); plt.colorbar(im)
        plt.xticks(range(len(labels)),labels,rotation=45,ha="right"); plt.yticks(range(len(labels)),labels)
        plt.xlabel("Predicted class",fontsize=settings["label_size"]); plt.ylabel("Observed class",fontsize=settings["label_size"])
        plt.title(f"OOF confusion matrix: {model}",fontsize=settings["title_size"]); fmt=".2f" if normalize else "d"
        for i in range(cm_to_plot.shape[0]):
            for j in range(cm_to_plot.shape[1]): plt.text(j,i,annot_fmt.format(cm_to_plot[i,j],fmt),ha="center",va="center")
        plt.tight_layout(); path=Path(results_dir)/f"confusion_matrix_{model}_{output_prefix}.{settings['file_format']}"
        _finish_figure(path,save_figures,show_figures,settings); paths.append(path)
    return paths

### 10. External-test and virtual-screening utilities ###
def evaluate_external_dataset( df_external, training_columns, models_dict, target_column, id_columns=None, drop_columns=None, model_names=None, include_prediction_std=True, ):
    """Predict external or screening data with full-training ensemble members."""
    X_external = prepare_external_features( df_external, training_columns, target_column, id_columns, drop_columns )
    model_names = model_names or list(models_dict)
    output = df_external.reset_index(drop=True).copy()
    metrics_rows = []
    has_target = target_column in output.columns and output[target_column].notna().all()
    y_cont_external = pd.to_numeric(output[target_column], errors="raise") if has_target else None
    for model_name in model_names:
        ensemble = models_dict[model_name]
        member_probabilities = ensemble.predict_proba_members(X_external)
        mean_probabilities = member_probabilities.mean(axis=0)
        std_probabilities = member_probabilities.std(axis=0)
        predicted = mean_probabilities.argmax(axis=1)
        output[f"Predicted_encoded_{model_name}"] = predicted
        output[f"Predicted_label_{model_name}"] = [ensemble.class_names[index] for index in predicted]
        for class_index, label in enumerate(ensemble.class_names):
            output[f"Prob_{label}_{model_name}"] = mean_probabilities[:, class_index]
            if include_prediction_std:
                output[f"Prob_STD_{label}_{model_name}"] = std_probabilities[:, class_index]
        if include_prediction_std:
            output[f"Prediction_STD_{model_name}"] = std_probabilities.max(axis=1)
        if has_target:
            true_members = np.column_stack([ classes_from_thresholds(y_cont_external, member["thresholds"]).to_numpy() for member in ensemble.members ])
            true_classes = np.apply_along_axis(_majority_vote, 1, true_members)
            output[f"True_encoded_{model_name}"] = true_classes
            output[f"True_label_{model_name}"] = [ensemble.class_names[index] for index in true_classes]
            metrics = evaluate_classification(true_classes, predicted, mean_probabilities, len(ensemble.class_names))
            metrics_rows.append({ "Model": model_name, "N_Ensemble_Members": len(ensemble.members), **{f"{key}_external": value for key, value in metrics.items()}, })
    return output, pd.DataFrame(metrics_rows)


def combine_training_and_external_predictions( results: dict[str, Any], external_results: pd.DataFrame, target_column: str, model_names: list[str] | None = None, ) -> pd.DataFrame:
    """Combine mean OOF training and external classification predictions."""
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
                    "Target": row["Target"],
                    "True": row["True_encoded"],
                    "Predicted": row["Predicted_encoded"],
                    "Correct": row["Correct"],
                    "Prediction_STD": np.nan,
                }
            )

        prediction_column = f"Predicted_encoded_{model_name}"
        true_column = f"True_encoded_{model_name}"
        std_column = f"Prediction_STD_{model_name}"

        for external_index, external_row in external_results.iterrows():
            true_value = ( external_row[true_column] if true_column in external_results.columns else np.nan )
            predicted_value = external_row[prediction_column]
            combined_rows.append(
                {
                    "Dataset": "External",
                    "Model": model_name,
                    "Sample_ID": external_index,
                    "Target": (
                        external_row[target_column]
                        if target_column in external_results.columns
                        else np.nan
                    ),
                    "True": true_value,
                    "Predicted": predicted_value,
                    "Correct": (
                        int(true_value == predicted_value)
                        if pd.notna(true_value)
                        else np.nan
                    ),
                    "Prediction_STD": (
                        external_row[std_column]
                        if std_column in external_results.columns
                        else np.nan
                    ),
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

def plot_external_confusion_matrice(
    external_results: pd.DataFrame,
    results: dict[str, Any],
    results_dir: str | Path,
    output_prefix: str,
    model_names: list[str] | None = None,
    external_label: str = "External test",
    normalize: bool = False,
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
):
    """Plot one confusion matrix per model for the external test dataset."""
    settings = _resolve_plot_settings(plot_settings)
    labels = results["configuration_labels"]
    model_names = model_names or list(results["models_dict"])
    paths = []

    for model_name in model_names:
        true_column = f"True_encoded_{model_name}"
        predicted_column = f"Predicted_encoded_{model_name}"

        missing_columns = [ column for column in [true_column, predicted_column] if column not in external_results.columns ]
        if missing_columns:
            raise KeyError( "External prediction columns were not found: " f"{missing_columns}" )

        y_true = external_results[true_column].astype(int)
        y_predicted = external_results[predicted_column].astype(int)

        matrix = confusion_matrix( y_true, y_predicted, labels=range(len(labels)), normalize="true" if normalize else None, )

        plt.figure(figsize=settings["parity_figure_size"])
        image = plt.imshow( matrix, cmap="Blues", vmin=0, vmax=1 if normalize else None, )
        plt.colorbar(image)

        plt.xticks( range(len(labels)), labels, rotation=45, ha="right", fontsize=settings["tick_size"], )
        plt.yticks( range(len(labels)), labels, fontsize=settings["tick_size"], )

        plt.xlabel( "Predicted class", fontsize=settings["label_size"], )
        plt.ylabel( "Observed class", fontsize=settings["label_size"], )
        plt.title( f"{external_label}: {model_name}", fontsize=settings["title_size"], )

        number_format = ".2f" if normalize else "d"
        threshold = ( np.nanmax(matrix) / 2 if matrix.size and np.nanmax(matrix) > 0 else 0 )

        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]

                plt.text(
                    column_index,
                    row_index,
                    format(value, number_format),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if value > threshold
                        else "black"
                    ),
                    fontsize=settings["tick_size"],
                )

        plt.tight_layout()

        extension = settings["file_format"]
        path = Path(results_dir) / ( f"external_confusion_matrix_" f"{model_name}_{output_prefix}.{extension}" )

        _finish_figure( path, save_figures, show_figures, settings, )
        paths.append(path)

    return paths


def plot_virtual_screening_distribution(y_training, screening_results, model_names,
                                        results_dir, output_prefix, target_label="Class probability",
                                        bins=20, save_figures=True, show_figures=True,
                                        plot_settings=None):
    """Compare training-target and screening high-class probability distributions."""
    settings=_resolve_plot_settings(plot_settings); paths=[]
    for model in model_names:
        prob_cols=[c for c in screening_results.columns if c.startswith("Prob_") and c.endswith(f"_{model}")]
        if not prob_cols: continue
        col=prob_cols[-1]; plt.figure(figsize=settings["bar_figure_size"])
        plt.hist(screening_results[col],bins=bins,color=settings["external_color"],edgecolor=settings["bar_edge_color"],linewidth=settings["bar_edge_width"])
        plt.xlabel(target_label,fontsize=settings["label_size"]); plt.ylabel("Frequency",fontsize=settings["label_size"])
        plt.title(f"Virtual-screening distribution: {model}",fontsize=settings["title_size"]); _style_axes(settings); plt.tight_layout()
        path=Path(results_dir)/f"virtual_screening_distribution_{model}_{output_prefix}.{settings['file_format']}"
        _finish_figure(path,save_figures,show_figures,settings); paths.append(path)
    return paths



def _feature_class_threshold(x,y,target_threshold):
    x=np.asarray(x,float); y=np.asarray(y,float); valid=np.isfinite(x)&np.isfinite(y); x=x[valid]; classes=(y[valid]>target_threshold).astype(int)
    values=np.unique(x)
    if len(values)<2 or len(np.unique(classes))<2: return np.nan
    candidates=(values[:-1]+values[1:])/2; best=None
    for threshold in candidates:
        pred=(x>threshold).astype(int); score=max(np.mean(pred==classes),np.mean((1-pred)==classes)); candidate=(score,-abs(threshold-np.median(x)),threshold)
        if best is None or candidate>best: best=candidate
    return float(best[2])

def plot_classification_feature_maps(
    X_training: pd.DataFrame,
    y_training: pd.Series | Iterable[float],
    results: dict[str, Any],
    evaluated_results: pd.DataFrame,
    model_names: list[str],
    results_dir: str | Path,
    output_prefix: str,
    target_column: str,
    dataset_kind: str = "external",
    dataset_label: str | None = None,
    features: list[str] | None = None,
    top_n_features: int = 3,
    hide_training: bool = False,
    cmap: str = "viridis",
    save_figures: bool = True,
    show_figures: bool = True,
    plot_settings: dict[str, Any] | None = None,
) -> list[Path]:
    """Plot OOF, full-training, external-test or screening feature maps.

    ``oof`` shows only held-out OOF predictions as squares. ``full`` shows
    only predictions from the ensembles refitted on the complete training set
    as circles. ``external`` overlays external observations as squares on the
    full-training reference circles. ``screening`` overlays candidates as
    crosses below the observed-response range.
    """
    settings = _resolve_plot_settings(plot_settings)
    dataset_kind = dataset_kind.lower()
    allowed = {"oof", "full", "external", "screening"}
    if dataset_kind not in allowed:
        raise ValueError( "dataset_kind must be 'oof', 'full', 'external', or 'screening'." )

    y_training = pd.Series(y_training).reset_index(drop=True).astype(float)
    X_training = X_training.reset_index(drop=True)
    evaluated_results = evaluated_results.reset_index(drop=True)
    default_labels = { "oof": "OOF validation", "full": "Full-training ensemble", "external": "External test", "screening": "Virtual screening", }
    dataset_label = dataset_label or default_labels[dataset_kind]

    paths: list[Path] = []
    for model_name in model_names:
        ensemble_key = "models_dict" if dataset_kind == "oof" else "full_models_dict"
        if ensemble_key not in results:
            raise KeyError(f"{ensemble_key!r} is unavailable.")
        if model_name not in results[ensemble_key]:
            raise KeyError(f"Unknown model: {model_name}")

        ensemble = results[ensemble_key][model_name]
        highest_label = ensemble.class_names[-1]
        evaluated_probability_column = ( f"Prob_{highest_label}" if dataset_kind in {"oof", "full"} else f"Prob_{highest_label}_{model_name}" )
        if evaluated_probability_column not in evaluated_results.columns:
            raise KeyError(f"Missing probability column: {evaluated_probability_column}")

        if features:
            selected_features = list(dict.fromkeys(features))
        else:
            selected_features = results.get("representative_features", {}).get( model_name, [] )[:top_n_features]
            if not selected_features:
                selected_features = list(X_training.columns[:top_n_features])

        missing = [f for f in selected_features if f not in evaluated_results.columns]
        if missing:
            raise KeyError(f"Feature(s) missing from evaluated data: {missing}")

        member_thresholds = [member.get("thresholds", []) for member in ensemble.members]
        n_thresholds = max((len(v) for v in member_thresholds), default=0)
        threshold_medians = []
        for threshold_index in range(n_thresholds):
            values = [ float(v[threshold_index]) for v in member_thresholds if len(v) > threshold_index ]
            if values:
                threshold_medians.append(float(np.median(values)))

        # Full-training reference probabilities are used only for external and screening.
        training_probabilities = None
        if dataset_kind in {"external", "screening"} and not hide_training:
            full_predictions = results["predictions_full_ensemble"]
            full_predictions = full_predictions[ full_predictions["Model"] == model_name ].sort_values("Sample_ID")
            training_probabilities = np.full(len(X_training), np.nan, dtype=float)
            sample_ids = full_predictions["Sample_ID"].astype(int).to_numpy()
            valid_ids = (sample_ids >= 0) & (sample_ids < len(X_training))
            training_probabilities[sample_ids[valid_ids]] = full_predictions.loc[ valid_ids, f"Prob_{highest_label}" ].to_numpy(dtype=float)

        for feature in selected_features:
            x_eval = pd.to_numeric(evaluated_results[feature], errors="coerce")
            p_eval = pd.to_numeric( evaluated_results[evaluated_probability_column], errors="coerce" )
            valid_eval = x_eval.notna().to_numpy() & p_eval.notna().to_numpy()

            fig, axis = plt.subplots(figsize=settings["bar_figure_size"])
            mappable = None

            if dataset_kind in {"external", "screening"} and training_probabilities is not None:
                x_train = pd.to_numeric(X_training[feature], errors="coerce")
                valid_train = ( x_train.notna().to_numpy() & y_training.notna().to_numpy() & np.isfinite(training_probabilities) )
                if valid_train.any():
                    mappable = axis.scatter(
                        x_train.to_numpy()[valid_train],
                        y_training.to_numpy()[valid_train],
                        c=training_probabilities[valid_train],
                        cmap=cmap,
                        vmin=0,
                        vmax=1,
                        marker="o",
                        s=settings["point_size"],
                        alpha=0.55,
                        edgecolor=settings["point_edge_color"],
                        linewidth=settings["point_edge_width"],
                        label="Full-training ensemble",
                    )

            if dataset_kind in {"oof", "full", "external"}:
                observed_column = "Target" if dataset_kind in {"oof", "full"} else target_column
                if observed_column not in evaluated_results.columns:
                    raise KeyError( f"{dataset_label} plot requires {observed_column!r}." )
                y_eval = pd.to_numeric(evaluated_results[observed_column], errors="coerce")
                valid_eval &= y_eval.notna().to_numpy()
                marker = "s" if dataset_kind in {"oof", "external"} else "o"
                edgecolor = ( settings["validation_color"] if dataset_kind == "oof" else settings["test_color"] if dataset_kind == "external" else settings["point_edge_color"] )
                mappable = axis.scatter(
                    x_eval.to_numpy()[valid_eval],
                    y_eval.to_numpy()[valid_eval],
                    c=p_eval.to_numpy()[valid_eval],
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    marker=marker,
                    s=settings["point_size"] * (1.15 if marker == "s" else 1.0),
                    alpha=1.0,
                    edgecolor=edgecolor,
                    linewidth=max(1.2, settings["point_edge_width"]),
                    label=dataset_label,
                )
                y_for_limits = np.concatenate([ y_training.to_numpy(dtype=float), y_eval.dropna().to_numpy(dtype=float), ])
            else:
                y_range = float(y_training.max() - y_training.min())
                baseline = float(y_training.min() - (0.08 * y_range if y_range else 1.0))
                mappable = axis.scatter(
                    x_eval.to_numpy()[valid_eval],
                    np.full(valid_eval.sum(), baseline),
                    c=p_eval.to_numpy()[valid_eval],
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    marker="x",
                    s=settings["point_size"] * 1.25,
                    linewidth=max(1.5, settings["point_edge_width"]),
                    label=dataset_label,
                )
                y_for_limits = np.concatenate([ y_training.to_numpy(dtype=float), np.asarray([baseline]) ])

            for i, threshold in enumerate(threshold_medians, start=1):
                axis.axhline(
                    threshold,
                    color=settings["reference_line_color"],
                    linestyle="--",
                    linewidth=settings["line_width"],
                    alpha=0.8,
                    label=f"Median class threshold {i}" if i == 1 else None,
                )
                feature_threshold=_feature_class_threshold(X_training[feature],y_training,threshold)
                if np.isfinite(feature_threshold): axis.axvline(feature_threshold,color=settings["reference_line_color"],linestyle=":",linewidth=settings["line_width"],alpha=0.8,label=f"Feature threshold {i} = {feature_threshold:.3g}" if i == 1 else None)

            finite_y = y_for_limits[np.isfinite(y_for_limits)]
            if len(finite_y):
                span = float(finite_y.max() - finite_y.min())
                margin = 0.06 * span if span else 1.0
                axis.set_ylim(finite_y.min() - margin, finite_y.max() + margin)

            axis.set_xlabel(feature, fontsize=settings["label_size"])
            axis.set_ylabel(target_column, fontsize=settings["label_size"])
            axis.set_title( f"{dataset_label}: {feature} ({model_name})", fontsize=settings["title_size"], )
            axis.tick_params(axis="both", labelsize=settings["tick_size"])
            axis.legend(loc="best", fontsize=settings["legend_size"], frameon=True)

            colorbar = fig.colorbar(mappable, ax=axis)
            colorbar.set_label( f"Predicted probability: {highest_label}", fontsize=settings["label_size"], )
            colorbar.ax.tick_params(labelsize=settings["tick_size"])

            fig.tight_layout()
            path = Path(results_dir) / ( f"classification_feature_map_{dataset_kind}_{model_name}_" f"{_safe_filename(feature)}_{output_prefix}.{settings['file_format']}" )
            if save_figures:
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, dpi=settings["dpi"], bbox_inches="tight")
            if show_figures:
                plt.show()
            plt.close(fig)
            paths.append(path)

    return paths


### 11. Ensemble SHAP interpretation ###

def compute_ensemble_shap(ensemble_model, X_background, X_explain, model_name, dataset_labels=None, sample_ids=None, background_size=100, max_evals=None, random_state=42):
    """Explain the ensemble probability assigned to the highest ordered class."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP is required for ensemble interpretation.") from exc
    X_explain=X_explain.loc[:,X_background.columns].reset_index(drop=True)
    background=shap.sample(X_background.reset_index(drop=True),min(background_size,len(X_background)),random_state=random_state)
    prediction_function=lambda data: ensemble_model.predict_proba(pd.DataFrame(data,columns=X_background.columns))[:,-1]
    max_evals=max_evals or 2*X_background.shape[1]+1
    explainer=shap.Explainer(prediction_function,background,algorithm="permutation",feature_names=list(X_background.columns))
    explanation=explainer(X_explain,max_evals=max_evals)
    labels=list(dataset_labels) if dataset_labels is not None else ["Train"]*len(X_explain)
    ids=list(sample_ids) if sample_ids is not None else list(range(len(X_explain)))
    values=pd.DataFrame(explanation.values,columns=X_background.columns)
    values.insert(0,"Ensemble_Probability",prediction_function(X_explain)); values.insert(0,"Base_Value",np.asarray(explanation.base_values).reshape(-1))
    values.insert(0,"Sample_ID",ids); values.insert(0,"Dataset",labels); values.insert(0,"Model",model_name)
    importance=pd.DataFrame({"Model":model_name,"Feature":X_background.columns,"Mean_abs_SHAP":np.abs(explanation.values).mean(axis=0)}).sort_values("Mean_abs_SHAP",ascending=False)
    return {"explanation":explanation,"values":values,"importance":importance,"background":background,"X_explain":X_explain}

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
