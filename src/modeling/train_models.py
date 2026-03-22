from __future__ import annotations

from pathlib import Path
from typing import Callable
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import warnings

import config.logging as log_config

#--------------------------------------------------------
# Logistic Regression Training Functions
#--------------------------------------------------------

def scale_logistic_regression_inputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    log: Callable[[str], None] | Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale training and testing inputs for Logistic Regression using
    training-set fitted standardization.
    """
    try:
        if X_train is None or X_test is None:
            raise ValueError("X_train and X_test must not be None.")

        if X_train.empty:
            raise ValueError("X_train must not be empty.")

        if X_test.empty:
            raise ValueError("X_test must not be empty.")

        scaler = StandardScaler()

        scaled_train_array = scaler.fit_transform(X_train)
        scaled_test_array = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(
            scaled_train_array,
            columns=X_train.columns,
            index=X_train.index,
        )

        X_test_scaled = pd.DataFrame(
            scaled_test_array,
            columns=X_test.columns,
            index=X_test.index,
        )

        log_config.emit_log(
            log=log,
            message=(
                "[scale_logistic_regression_inputs] "
                f"Scaled train_shape={X_train.shape}, test_shape={X_test.shape}"
            ),
        )

        return X_train_scaled, X_test_scaled, scaler

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=f"[scale_logistic_regression_inputs][error] {type(exc).__name__}: {exc}",
        )
        raise


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    log: Callable[[str], None] | Path | str | None = None,
    random_state: int = 42,
    max_iter: int = 1000,
    class_weight: str | dict[int, float] | None = None,
    solver: str = "lbfgs",
) -> tuple[LogisticRegression, dict[str, object]]:
    """
    Train a logistic regression classifier on engineered training data.
    """
    try:
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must not be None.")

        if X_train.empty:
            raise ValueError("X_train must not be empty.")

        if y_train.empty:
            raise ValueError("y_train must not be empty.")

        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train row counts must match. "
                f"Got len(X_train)={len(X_train)} and len(y_train)={len(y_train)}."
            )

        if y_train.isna().any():
            raise ValueError("y_train contains missing values.")

        unique_target_values = sorted(y_train.dropna().unique().tolist())
        if unique_target_values != [0, 1]:
            raise ValueError(
                f"y_train must be binary with values [0, 1]. Got {unique_target_values}."
            )

        logistic_regression_model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight,
            solver=solver,
        )

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            logistic_regression_model.fit(X_train, y_train)

        warning_messages = [
            f"{captured_warning.category.__name__}: {captured_warning.message}"
            for captured_warning in captured_warnings
        ]

        convergence_warnings = [
            captured_warning
            for captured_warning in captured_warnings
            if issubclass(captured_warning.category, ConvergenceWarning)
        ]

        training_metadata = {
            "warning_count": len(captured_warnings),
            "convergence_warning_count": len(convergence_warnings),
            "warning_messages": warning_messages,
        }

        log_config.emit_log(
            log=log,
            message=(
                "[train_logistic_regression] "
                f"Trained LogisticRegression with rows={X_train.shape[0]}, "
                f"columns={X_train.shape[1]}, solver={solver}, "
                f"max_iter={max_iter}, class_weight={class_weight}"
            ),
        )

        for warning_message in warning_messages:
            log_config.emit_log(
                log=log,
                message=f"[train_logistic_regression][warning] {warning_message}",
            )

        if convergence_warnings:
            log_config.emit_log(
                log=log,
                message=(
                    "[train_logistic_regression] "
                    f"Convergence warnings captured={len(convergence_warnings)}"
                ),
            )

        return logistic_regression_model, training_metadata

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=f"[train_logistic_regression][error] {type(exc).__name__}: {exc}",
        )
        raise


#--------------------------------------------------------
# Random Forest Training Functions
#--------------------------------------------------------


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    log: Callable[[str], None] | Path | str | None = None,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str | int | float | None = "sqrt",
    class_weight: str | dict[int, float] | list[dict[int, float]] | None = None,
    n_jobs: int = -1,
) -> tuple[RandomForestClassifier, dict[str, object]]:
    """
    Train a random forest classifier on engineered training data.
    """
    try:
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must not be None.")

        if X_train.empty:
            raise ValueError("X_train must not be empty.")

        if y_train.empty:
            raise ValueError("y_train must not be empty.")

        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train row counts must match. "
                f"Got len(X_train)={len(X_train)} and len(y_train)={len(y_train)}."
            )

        if y_train.isna().any():
            raise ValueError("y_train contains missing values.")

        if X_train.isna().any().any():
            raise ValueError("X_train contains missing values. Random Forest requires imputed inputs.")

        unique_target_values = sorted(y_train.dropna().unique().tolist())
        if unique_target_values != [0, 1]:
            raise ValueError(
                f"y_train must be binary with values [0, 1]. Got {unique_target_values}."
            )

        random_forest_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            random_forest_model.fit(X_train, y_train)

        warning_messages = [
            f"{captured_warning.category.__name__}: {captured_warning.message}"
            for captured_warning in captured_warnings
        ]

        training_metadata = {
            "warning_count": len(captured_warnings),
            "warning_messages": warning_messages,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
            "class_weight": class_weight,
        }

        log_config.emit_log(
            log=log,
            message=(
                "[train_random_forest] "
                f"Trained RandomForestClassifier with rows={X_train.shape[0]}, "
                f"columns={X_train.shape[1]}, n_estimators={n_estimators}, "
                f"max_depth={max_depth}, max_features={max_features}, "
                f"class_weight={class_weight}"
            ),
        )

        for warning_message in warning_messages:
            log_config.emit_log(
                log=log,
                message=f"[train_random_forest][warning] {warning_message}",
            )

        return random_forest_model, training_metadata

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=f"[train_random_forest][error] {type(exc).__name__}: {exc}",
        )
        raise
    

#--------------------------------------------------------
# CatBoost Training Functions
#--------------------------------------------------------


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    log: Callable[[str], None] | Path | str | None = None,
    categorical_feature_names: list[str] | None = None,
    random_state: int = 42,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    loss_function: str = "Logloss",
    eval_metric: str = "AUC",
    verbose: bool | int = False,
) -> tuple[CatBoostClassifier, dict[str, object]]:
    """
    Train a CatBoost classifier on engineered training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Engineered training feature matrix.
    y_train : pd.Series
        Binary training target.
    log : Callable[[str], None] | Path | str | None, default=None
        Callable logger or log file path supported by log_config.emit_log.
    categorical_feature_names : list[str] | None, default=None
        List of categorical feature names to pass to CatBoost for native
        categorical handling. If None, CatBoost is trained without an explicit
        categorical feature list.
    random_state : int, default=42
        Random seed for reproducibility.
    iterations : int, default=500
        Number of boosting iterations.
    learning_rate : float, default=0.05
        Boosting learning rate.
    depth : int, default=6
        Tree depth.
    l2_leaf_reg : float, default=3.0
        L2 regularization term.
    loss_function : str, default="Logloss"
        Optimization objective.
    eval_metric : str, default="AUC"
        Evaluation metric tracked during training.
    verbose : bool | int, default=False
        CatBoost verbosity setting.

    Returns
    -------
    tuple[CatBoostClassifier, dict[str, object]]
        Trained CatBoost model and training metadata.

    Raises
    ------
    ValueError
        If training inputs are empty, misaligned, or invalid.
    """
    try:
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must not be None.")

        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be a pandas DataFrame.")

        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train must be a pandas Series.")

        if X_train.empty:
            raise ValueError("X_train must not be empty.")

        if y_train.empty:
            raise ValueError("y_train must not be empty.")

        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train row counts must match. "
                f"Got len(X_train)={len(X_train)} and len(y_train)={len(y_train)}."
            )

        if y_train.isna().any():
            raise ValueError("y_train contains missing values.")

        unique_target_values = sorted(y_train.dropna().unique().tolist())
        if unique_target_values != [0, 1]:
            raise ValueError(
                f"y_train must be binary with values [0, 1]. Got {unique_target_values}."
            )

        validated_categorical_feature_names: list[str] = []

        if categorical_feature_names is not None:
            if not isinstance(categorical_feature_names, list):
                raise ValueError(
                    "categorical_feature_names must be a list of column names or None."
                )

            missing_categorical_features = [
                feature_name
                for feature_name in categorical_feature_names
                if feature_name not in X_train.columns
            ]
            if missing_categorical_features:
                raise ValueError(
                    "All categorical_feature_names must exist in X_train. "
                    f"Missing columns: {missing_categorical_features}"
                )

            duplicate_categorical_features = sorted(
                {
                    feature_name
                    for feature_name in categorical_feature_names
                    if categorical_feature_names.count(feature_name) > 1
                }
            )
            if duplicate_categorical_features:
                raise ValueError(
                    "categorical_feature_names must not contain duplicates. "
                    f"Duplicate columns: {duplicate_categorical_features}"
                )

            validated_categorical_feature_names = categorical_feature_names.copy()

        catboost_model = CatBoostClassifier(
            random_seed=random_state,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            loss_function=loss_function,
            eval_metric=eval_metric,
            verbose=verbose,
        )

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            catboost_model.fit(
                X_train,
                y_train,
                cat_features=validated_categorical_feature_names or None,
            )

        warning_messages = [
            f"{captured_warning.category.__name__}: {captured_warning.message}"
            for captured_warning in captured_warnings
        ]

        training_metadata = {
            "warning_count": len(captured_warnings),
            "warning_messages": warning_messages,
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "loss_function": loss_function,
            "eval_metric": eval_metric,
            "categorical_feature_count": len(validated_categorical_feature_names),
            "categorical_feature_names": validated_categorical_feature_names,
        }

        log_config.emit_log(
            log=log,
            message=(
                "[train_catboost] "
                f"Trained CatBoostClassifier with rows={X_train.shape[0]}, "
                f"columns={X_train.shape[1]}, iterations={iterations}, "
                f"learning_rate={learning_rate}, depth={depth}, "
                f"l2_leaf_reg={l2_leaf_reg}, loss_function={loss_function}, "
                f"eval_metric={eval_metric}, "
                f"categorical_feature_count={len(validated_categorical_feature_names)}"
            ),
        )

        if validated_categorical_feature_names:
            log_config.emit_log(
                log=log,
                message=(
                    "[train_catboost] "
                    f"categorical_feature_names={validated_categorical_feature_names}"
                ),
            )

        for warning_message in warning_messages:
            log_config.emit_log(
                log=log,
                message=f"[train_catboost][warning] {warning_message}",
            )

        return catboost_model, training_metadata

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=f"[train_catboost][error] {type(exc).__name__}: {exc}",
        )
        raise