from __future__ import annotations

from typing import Any, Callable
from pathlib import Path

import pandas as pd
import numpy as np

import config.logging as log_config
import features.feature_utils as fu


def build_feature_group_audit(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_column: str,
    log: Any = None,
) -> pd.DataFrame:
    """
    Build a feature audit table describing how each non-target column is
    currently represented and how it should be reviewed for transformation.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataset for the modeling stage.
    df_test : pd.DataFrame
        Testing dataset for the modeling stage.
    target_column : str
        Name of the binary prediction target.
    log : Any, default=None
        Callable logger or log file path supported by log_config.emit_log.

    Returns
    -------
    pd.DataFrame
        Feature audit table for all non-target columns.
    """
    try:
        if target_column not in df_train.columns:
            raise KeyError(f"'{target_column}' not found in training data.")

        if target_column not in df_test.columns:
            raise KeyError(f"'{target_column}' not found in testing data.")

        train_feature_columns = [column for column in df_train.columns if column != target_column]
        test_feature_columns = [column for column in df_test.columns if column != target_column]

        if train_feature_columns != test_feature_columns:
            raise ValueError("Training and testing feature columns are not aligned.")

        feature_audit_records: list[dict[str, Any]] = []

        for column_name in train_feature_columns:
            train_series = df_train[column_name]
            test_series = df_test[column_name]

            train_dtype = str(train_series.dtype)
            test_dtype = str(test_series.dtype)

            non_null_train_unique_values = train_series.dropna().nunique()
            non_null_test_unique_values = test_series.dropna().nunique()

            is_numeric = pd.api.types.is_numeric_dtype(train_series)
            is_binary = False

            combined_non_null_values = pd.concat(
                [train_series, test_series],
                axis=0
            ).dropna()

            if is_numeric and combined_non_null_values.nunique() <= 2:
                is_binary = True

            if pd.api.types.is_bool_dtype(train_series):
                feature_group = "boolean"
            elif is_numeric:
                feature_group = "numeric"
            else:
                feature_group = "categorical"

            requires_manual_review = False

            if train_dtype != test_dtype:
                requires_manual_review = True

            feature_audit_records.append(
                {
                    "feature_name": column_name,
                    "train_dtype": train_dtype,
                    "test_dtype": test_dtype,
                    "feature_group": feature_group,
                    "is_binary": is_binary,
                    "train_non_null_unique_values": non_null_train_unique_values,
                    "test_non_null_unique_values": non_null_test_unique_values,
                    "requires_manual_review": requires_manual_review,
                }
            )

        feature_group_audit_df = pd.DataFrame(feature_audit_records).sort_values(
            by=["feature_group", "feature_name"]
        ).reset_index(drop=True)

        log_config.emit_log(
            log,
            f"Built feature group audit for {feature_group_audit_df.shape[0]} features."
        )

        return feature_group_audit_df

    except Exception as exc:
        log_config.emit_log(
            log,
            f"Failed to build feature group audit: {exc}"
        )
        raise


def apply_numeric_log_transformation(
    df: pd.DataFrame,
    feature_names: list[str],
    dataset_name: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Apply log1p transformation to selected numeric features in a DataFrame.
    """
    try:
        if df is None:
            raise ValueError("df must not be None")

        if not dataset_name or not str(dataset_name).strip():
            raise ValueError("dataset_name must be a non-empty string")

        requested_features = list(dict.fromkeys(feature_names))

        if not requested_features:
            raise ValueError("feature_names must contain at least one feature")

        existing_columns = set(df.columns)

        missing_features = [
            feature_name
            for feature_name in requested_features
            if feature_name not in existing_columns
        ]
        if missing_features:
            raise KeyError(
                f"Numeric log-transform features not found in dataframe: {missing_features}"
            )

        negative_features = [
            feature_name
            for feature_name in requested_features
            if (df[feature_name].dropna() < 0).any()
        ]
        if negative_features:
            raise ValueError(
                f"Negative values found in log-transform features: {negative_features}"
            )

        transformed_dataframe = df.copy()
        shape_before = transformed_dataframe.shape

        for feature_name in requested_features:
            transformed_dataframe[feature_name] = np.log1p(
                transformed_dataframe[feature_name]
            ).astype(float)

        shape_after = transformed_dataframe.shape

        log_config.emit_log(log, f"[numeric_log_transform] dataset={dataset_name}")
        log_config.emit_log(
            log,
            f"[numeric_log_transform] shape_before={shape_before} shape_after={shape_after}",
        )
        log_config.emit_log(
            log,
            f"[numeric_log_transform] transformed={len(requested_features)}",
        )
        log_config.emit_log(
            log,
            "[numeric_log_transform] transformed_features=" + fu._format_column_list(requested_features),
        )

        return transformed_dataframe

    except Exception as exc:
        try:
            log_config.emit_log(
                log,
                f"[apply_numeric_log_transformation][error] dataset={dataset_name} "
                f"Error={type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise


def apply_one_hot_encoding(
    df: pd.DataFrame,
    feature_names: list[str],
    dataset_name: str,
    category_mapping: dict[str, list[str]],
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Apply one-hot encoding to selected categorical features using a predefined
    category mapping.
    """
    try:
        if df is None:
            raise ValueError("df must not be None")

        if not dataset_name or not str(dataset_name).strip():
            raise ValueError("dataset_name must be a non-empty string")

        requested_features = list(dict.fromkeys(feature_names))

        if not requested_features:
            raise ValueError("feature_names must contain at least one feature")

        existing_columns = set(df.columns)

        missing_features = [
            feature_name
            for feature_name in requested_features
            if feature_name not in existing_columns
        ]
        if missing_features:
            raise KeyError(
                f"Categorical features not found in dataframe: {missing_features}"
            )

        missing_category_mappings = [
            feature_name
            for feature_name in requested_features
            if feature_name not in category_mapping
        ]
        if missing_category_mappings:
            raise KeyError(
                f"Category mapping missing for features: {missing_category_mappings}"
            )

        transformed_dataframe = df.copy()
        shape_before = transformed_dataframe.shape

        for feature_name in requested_features:
            transformed_dataframe[feature_name] = pd.Categorical(
                transformed_dataframe[feature_name],
                categories=category_mapping[feature_name],
            )

        transformed_dataframe = pd.get_dummies(
            transformed_dataframe,
            columns=requested_features,
            dummy_na=False,
        )

        shape_after = transformed_dataframe.shape

        log_config.emit_log(log, f"[one_hot_encoding] dataset={dataset_name}")
        log_config.emit_log(
            log,
            f"[one_hot_encoding] shape_before={shape_before} shape_after={shape_after}",
        )
        log_config.emit_log(
            log,
            f"[one_hot_encoding] encoded={len(requested_features)}",
        )
        log_config.emit_log(
            log,
            "[one_hot_encoding] encoded_features=" + fu._format_column_list(requested_features),
        )

        return transformed_dataframe

    except Exception as exc:
        try:
            log_config.emit_log(
                log,
                f"[apply_one_hot_encoding][error] dataset={dataset_name} "
                f"Error={type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise


def apply_median_imputation(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_names: list[str],
    log: Callable[[str], None] | Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values in selected features using training-set medians.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataframe used to derive imputation values.
    df_test : pd.DataFrame
        Testing dataframe that receives the same train-derived imputations.
    feature_names : list[str]
        Features to impute.
    log : Callable[[str], None] | Path | str | None, default=None
        Callable logger or log file path supported by log_config.emit_log.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Imputed training and testing dataframes.

    Raises
    ------
    ValueError
        If inputs are invalid.
    KeyError
        If requested features are missing.
    """
    try:
        if df_train is None or df_test is None:
            raise ValueError("df_train and df_test must not be None.")

        requested_features = list(dict.fromkeys(feature_names))

        if not requested_features:
            raise ValueError("feature_names must contain at least one feature.")

        missing_train_features = [
            feature_name
            for feature_name in requested_features
            if feature_name not in df_train.columns
        ]
        if missing_train_features:
            raise KeyError(
                f"Imputation features not found in training data: {missing_train_features}"
            )

        missing_test_features = [
            feature_name
            for feature_name in requested_features
            if feature_name not in df_test.columns
        ]
        if missing_test_features:
            raise KeyError(
                f"Imputation features not found in testing data: {missing_test_features}"
            )

        transformed_train = df_train.copy()
        transformed_test = df_test.copy()

        train_missing_before = int(transformed_train[requested_features].isna().sum().sum())
        test_missing_before = int(transformed_test[requested_features].isna().sum().sum())

        median_mapping: dict[str, float] = {}
        all_missing_features: list[str] = []

        for feature_name in requested_features:
            if transformed_train[feature_name].dropna().empty:
                all_missing_features.append(feature_name)
            else:
                median_mapping[feature_name] = float(transformed_train[feature_name].median())

        if all_missing_features:
            raise ValueError(
                "Cannot compute training-set median for features with all values missing: "
                f"{all_missing_features}"
            )

        for feature_name in requested_features:
            transformed_train[feature_name] = transformed_train[feature_name].fillna(
                median_mapping[feature_name]
            )
            transformed_test[feature_name] = transformed_test[feature_name].fillna(
                median_mapping[feature_name]
            )

        train_missing_after = int(transformed_train[requested_features].isna().sum().sum())
        test_missing_after = int(transformed_test[requested_features].isna().sum().sum())

        log_config.emit_log(
            log=log,
            message="[median_imputation] completed",
        )
        log_config.emit_log(
            log=log,
            message=(
                f"[median_imputation] features={len(requested_features)} "
                f"train_missing_before={train_missing_before} "
                f"train_missing_after={train_missing_after} "
                f"test_missing_before={test_missing_before} "
                f"test_missing_after={test_missing_after}"
            ),
        )
        log_config.emit_log(
            log=log,
            message=(
                "[median_imputation] imputed_features="
                + fu._format_column_list(requested_features)
            ),
        )

        return transformed_train, transformed_test

    except Exception as exc:
        try:
            log_config.emit_log(
                log=log,
                message=f"[apply_median_imputation][error] {type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise


def build_category_mapping(
    df_train: pd.DataFrame,
    feature_names: list[str],
    log: Callable[[str], None] | Path | str | None = None,
) -> dict[str, list[str]]:
    """
    Build category mappings for selected categorical features from training data.
    """
    try:
        if df_train is None:
            raise ValueError("df_train must not be None")

        requested_features = list(dict.fromkeys(feature_names))

        if not requested_features:
            raise ValueError("feature_names must contain at least one feature")

        missing_features = [
            feature_name
            for feature_name in requested_features
            if feature_name not in df_train.columns
        ]
        if missing_features:
            raise KeyError(
                f"Categorical features not found in training data: {missing_features}"
            )

        category_mapping = {
            feature_name: (
                df_train[feature_name]
                .dropna()
                .astype("category")
                .cat.categories
                .tolist()
            )
            for feature_name in requested_features
        }

        log_config.emit_log(
            log,
            "[build_category_mapping] features=" + fu._format_column_list(requested_features),
        )

        return category_mapping

    except Exception as exc:
        try:
            log_config.emit_log(
                log,
                f"[build_category_mapping][error] Error={type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise