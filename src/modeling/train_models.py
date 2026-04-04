from __future__ import annotations

from pathlib import Path
from typing import Callable
import warnings

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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


def tune_catboost_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    log: Callable[[str], None] | Path | str | None = None,
    categorical_feature_names: list[str] | None = None,
    sample_weight: pd.Series | np.ndarray | None = None,
    random_state: int = 42,
    n_trials: int = 25,
    validation_size: float = 0.2,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    """
    Tune CatBoost hyperparameters using a lightweight random search on a
    validation split drawn from the training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Binary training target.
    log : Callable[[str], None] | Path | str | None, default=None
        Callable logger or log file path supported by log_config.emit_log.
    categorical_feature_names : list[str] | None, default=None
        Optional list of categorical feature names for native CatBoost handling.
    sample_weight : pd.Series | np.ndarray | None, default=None
        Optional sample weights aligned to X_train and y_train.
    random_state : int, default=42
        Random seed for reproducibility.
    n_trials : int, default=25
        Number of random parameter combinations to evaluate.
    validation_size : float, default=0.2
        Fraction of training data reserved for validation.

    Returns
    -------
    tuple[dict[str, object], pd.DataFrame, dict[str, object]]
        Best parameter dictionary, full tuning results table, and train metadata.

    Raises
    ------
    ValueError
        If inputs are empty, misaligned, or invalid.
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

        if not isinstance(n_trials, int) or n_trials <= 0:
            raise ValueError("n_trials must be a positive integer.")

        if not (0 < validation_size < 1):
            raise ValueError("validation_size must be between 0 and 1.")

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

        validated_sample_weight: pd.Series | np.ndarray | None = None

        if sample_weight is not None:
            if isinstance(sample_weight, pd.Series):
                if sample_weight.empty:
                    raise ValueError("sample_weight must not be empty.")

                if len(sample_weight) != len(X_train):
                    raise ValueError(
                        f"sample_weight length must match X_train. "
                        f"Got len(sample_weight)={len(sample_weight)} and len(X_train)={len(X_train)}."
                    )

                if sample_weight.isna().any():
                    raise ValueError("sample_weight contains missing values.")

                if (sample_weight < 0).any():
                    raise ValueError("sample_weight must not contain negative values.")

                if not sample_weight.index.equals(X_train.index):
                    raise ValueError("sample_weight index must match X_train index.")

                validated_sample_weight = sample_weight.copy()

            elif isinstance(sample_weight, np.ndarray):
                if sample_weight.size == 0:
                    raise ValueError("sample_weight must not be empty.")

                if len(sample_weight) != len(X_train):
                    raise ValueError(
                        f"sample_weight length must match X_train. "
                        f"Got len(sample_weight)={len(sample_weight)} and len(X_train)={len(X_train)}."
                    )

                if np.isnan(sample_weight).any():
                    raise ValueError("sample_weight contains missing values.")

                if (sample_weight < 0).any():
                    raise ValueError("sample_weight must not contain negative values.")

                validated_sample_weight = sample_weight.copy()

            else:
                raise ValueError("sample_weight must be a pandas Series, numpy array, or None.")

        split_inputs: list[object] = [X_train, y_train]
        split_kwargs: dict[str, object] = {
            "test_size": validation_size,
            "random_state": random_state,
            "stratify": y_train,
        }

        if validated_sample_weight is not None:
            split_inputs.append(validated_sample_weight)

        split_outputs = train_test_split(*split_inputs, **split_kwargs)

        if validated_sample_weight is None:
            X_train_sub, X_valid, y_train_sub, y_valid = split_outputs
            sample_weight_train_sub = None
            sample_weight_valid = None
        else:
            (
                X_train_sub,
                X_valid,
                y_train_sub,
                y_valid,
                sample_weight_train_sub,
                sample_weight_valid,
            ) = split_outputs

        random_number_generator = np.random.default_rng(seed=random_state)

        trial_results: list[dict[str, object]] = []

        log_config.emit_log(
            log=log,
            message=(
                "[tune_catboost_hyperparameters] "
                f"Starting tuning with n_trials={n_trials}, "
                f"train_rows={X_train_sub.shape[0]}, valid_rows={X_valid.shape[0]}, "
                f"categorical_feature_count={len(validated_categorical_feature_names)}, "
                f"weighted={validated_sample_weight is not None}"
            ),
        )

        for trial_index in range(n_trials):
            trial_params: dict[str, object] = {
                "iterations": int(random_number_generator.integers(300, 801)),
                "learning_rate": float(random_number_generator.uniform(0.02, 0.15)),
                "depth": int(random_number_generator.integers(4, 9)),
                "l2_leaf_reg": float(random_number_generator.uniform(1.0, 10.0)),
                "border_count": int(random_number_generator.integers(32, 129)),
                "random_strength": float(random_number_generator.uniform(0.5, 2.0)),
            }

            trial_model = CatBoostClassifier(
                random_seed=random_state,
                loss_function="Logloss",
                eval_metric="AUC",
                verbose=False,
                **trial_params,
            )

            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")
                trial_model.fit(
                    X_train_sub,
                    y_train_sub,
                    cat_features=validated_categorical_feature_names or None,
                    sample_weight=sample_weight_train_sub,
                )

            y_valid_probability = trial_model.predict_proba(X_valid)[:, 1]
            validation_roc_auc = float(roc_auc_score(y_valid, y_valid_probability))

            warning_messages = [
                f"{captured_warning.category.__name__}: {captured_warning.message}"
                for captured_warning in captured_warnings
            ]

            trial_result = {
                "trial_index": trial_index,
                "roc_auc": validation_roc_auc,
                "warning_count": len(captured_warnings),
                "iterations": trial_params["iterations"],
                "learning_rate": trial_params["learning_rate"],
                "depth": trial_params["depth"],
                "l2_leaf_reg": trial_params["l2_leaf_reg"],
                "border_count": trial_params["border_count"],
                "random_strength": trial_params["random_strength"],
                "warning_messages": " | ".join(warning_messages),
            }

            trial_results.append(trial_result)

            log_config.emit_log(
                log=log,
                message=(
                    "[tune_catboost_hyperparameters] "
                    f"trial_index={trial_index} roc_auc={validation_roc_auc:.6f} "
                    f"params={trial_params} warning_count={len(captured_warnings)}"
                ),
            )

            for warning_message in warning_messages:
                log_config.emit_log(
                    log=log,
                    message=(
                        "[tune_catboost_hyperparameters][warning] "
                        f"trial_index={trial_index} {warning_message}"
                    ),
                )

        tuning_results_df = pd.DataFrame(trial_results).sort_values(
            by="roc_auc",
            ascending=False,
        ).reset_index(drop=True)

        if tuning_results_df.empty:
            raise ValueError("CatBoost tuning produced no trial results.")

        best_row = tuning_results_df.iloc[0]

        best_params = {
            "iterations": int(best_row["iterations"]),
            "learning_rate": float(best_row["learning_rate"]),
            "depth": int(best_row["depth"]),
            "l2_leaf_reg": float(best_row["l2_leaf_reg"]),
            "border_count": int(best_row["border_count"]),
            "random_strength": float(best_row["random_strength"]),
        }

        train_metadata = {
            "n_trials": n_trials,
            "validation_size": validation_size,
            "best_roc_auc": float(best_row["roc_auc"]),
            "best_params": best_params,
            "categorical_feature_count": len(validated_categorical_feature_names),
            "categorical_feature_names": validated_categorical_feature_names,
            "weighted": validated_sample_weight is not None,
        }

        log_config.emit_log(
            log=log,
            message=(
                "[tune_catboost_hyperparameters] "
                f"Best roc_auc={best_row['roc_auc']:.6f} best_params={best_params}"
            ),
        )

        return best_params, tuning_results_df, train_metadata

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=(
                "[tune_catboost_hyperparameters][error] "
                f"{type(exc).__name__}: {exc}"
            ),
        )
        raise


def build_loan_amount_weights(
    loan_amounts: pd.Series,
    method: str = "sqrt",
    log=None,
) -> pd.Series:
    """
    Construct sample weights based on raw loan amounts.

    Parameters
    ----------
    loan_amounts : pd.Series
        Raw loan amounts aligned with training data index.
    method : str
        Weighting method: "sqrt", "log", or "raw".
    log : Callable | Path | str | None
        Logging target.

    Returns
    -------
    pd.Series
        Sample weights aligned with input index.
    """

    try:
        if loan_amounts is None:
            raise ValueError("loan_amounts must not be None.")

        if not isinstance(loan_amounts, pd.Series):
            raise TypeError("loan_amounts must be a pandas Series.")

        if loan_amounts.empty:
            raise ValueError("loan_amounts must not be empty.")

        if loan_amounts.isna().any():
            raise ValueError("loan_amounts contains missing values.")

        if (loan_amounts < 0).any():
            raise ValueError("loan_amounts must be non-negative.")

        if method == "sqrt":
            weights = np.sqrt(loan_amounts)
        elif method == "log":
            weights = np.log1p(loan_amounts)
        elif method == "raw":
            weights = loan_amounts.copy()
        else:
            raise ValueError(f"Unknown weighting method: {method}")

        weights = pd.Series(weights, index=loan_amounts.index)

        log_config.emit_log(
            log=log,
            message=(
                "[build_loan_amount_weights] "
                f"method={method} "
                f"min={weights.min():.4f} "
                f"max={weights.max():.4f} "
                f"mean={weights.mean():.4f}"
            ),
        )

        return weights

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=f"[build_loan_amount_weights][error] {type(exc).__name__}: {exc}",
        )
        raise


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    log: Callable[[str], None] | Path | str | None = None,
    categorical_feature_names: list[str] | None = None,
    sample_weight: pd.Series | np.ndarray | None = None,
    extra_params: dict[str, object] | None = None,
    random_state: int = 42,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    loss_function: str = "Logloss",
    eval_metric: str = "AUC",
    allow_writing_files: bool = False,
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
    sample_weight : pd.Series | np.ndarray | None, default=None
        Optional sample weights aligned to X_train and y_train.
    extra_params : dict[str, object] | None, default=None
        Optional parameter overrides, typically from hyperparameter tuning.
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
        Trained CatBoost model and train metadata.

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

        validated_sample_weight: pd.Series | np.ndarray | None = None

        if sample_weight is not None:
            if isinstance(sample_weight, pd.Series):
                if sample_weight.empty:
                    raise ValueError("sample_weight must not be empty.")

                if len(sample_weight) != len(X_train):
                    raise ValueError(
                        f"sample_weight length must match X_train. "
                        f"Got len(sample_weight)={len(sample_weight)} and len(X_train)={len(X_train)}."
                    )

                if sample_weight.isna().any():
                    raise ValueError("sample_weight contains missing values.")

                if (sample_weight < 0).any():
                    raise ValueError("sample_weight must not contain negative values.")

                if not sample_weight.index.equals(X_train.index):
                    raise ValueError("sample_weight index must match X_train index.")

                validated_sample_weight = sample_weight.copy()

            elif isinstance(sample_weight, np.ndarray):
                if sample_weight.size == 0:
                    raise ValueError("sample_weight must not be empty.")

                if len(sample_weight) != len(X_train):
                    raise ValueError(
                        f"sample_weight length must match X_train. "
                        f"Got len(sample_weight)={len(sample_weight)} and len(X_train)={len(X_train)}."
                    )

                if np.isnan(sample_weight).any():
                    raise ValueError("sample_weight contains missing values.")

                if (sample_weight < 0).any():
                    raise ValueError("sample_weight must not contain negative values.")

                validated_sample_weight = sample_weight.copy()

            else:
                raise ValueError("sample_weight must be a pandas Series, numpy array, or None.")

        model_params: dict[str, object] = {
            "random_seed": random_state,
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "loss_function": loss_function,
            "eval_metric": eval_metric,
            "allow_writing_files": allow_writing_files,
            "verbose": verbose,
        }

        if extra_params is not None:
            if not isinstance(extra_params, dict):
                raise ValueError("extra_params must be a dictionary or None.")

            model_params.update(extra_params)

        catboost_model = CatBoostClassifier(**model_params)

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            catboost_model.fit(
                X_train,
                y_train,
                cat_features=validated_categorical_feature_names or None,
                sample_weight=validated_sample_weight,
            )

        warning_messages = [
            f"{captured_warning.category.__name__}: {captured_warning.message}"
            for captured_warning in captured_warnings
        ]

        train_metadata = {
            "warning_count": len(captured_warnings),
            "warning_messages": warning_messages,
            "model_params": model_params,
            "categorical_feature_count": len(validated_categorical_feature_names),
            "categorical_feature_names": validated_categorical_feature_names,
            "weighted": validated_sample_weight is not None,
        }

        log_config.emit_log(
            log=log,
            message=(
                "[train_catboost] "
                f"Trained CatBoostClassifier with rows={X_train.shape[0]}, "
                f"columns={X_train.shape[1]}, "
                f"categorical_feature_count={len(validated_categorical_feature_names)}, "
                f"weighted={validated_sample_weight is not None}, "
                f"model_params={model_params}"
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

        if validated_sample_weight is not None:
            if isinstance(validated_sample_weight, pd.Series):
                weight_min = validated_sample_weight.min()
                weight_max = validated_sample_weight.max()
                weight_mean = validated_sample_weight.mean()
            else:
                weight_min = float(validated_sample_weight.min())
                weight_max = float(validated_sample_weight.max())
                weight_mean = float(validated_sample_weight.mean())

            log_config.emit_log(
                log=log,
                message=(
                    "[train_catboost] "
                    f"sample_weight_summary=min={weight_min:.4f}, "
                    f"max={weight_max:.4f}, mean={weight_mean:.4f}"
                ),
            )

        for warning_message in warning_messages:
            log_config.emit_log(
                log=log,
                message=f"[train_catboost][warning] {warning_message}",
            )

        return catboost_model, train_metadata

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=f"[train_catboost][error] {type(exc).__name__}: {exc}",
        )
        raise