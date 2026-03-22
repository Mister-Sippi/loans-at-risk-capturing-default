from __future__ import annotations

from typing import Callable, Sequence
from pathlib import Path

import pandas as pd

import config.logging as log_config
import modeling.evaluate_models as em


def build_modeling_population_summary_table(
    df_feature_base_train: pd.DataFrame,
    df_feature_base_test: pd.DataFrame,
    df_model_train: pd.DataFrame,
    df_model_test: pd.DataFrame,
    *,
    train_label: str = "train",
    test_label: str = "test",
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a comparison table showing dataset size before and after
    terminal-cohort filtering.
    """

    try:
        rows = [
            {
                "population_stage": "feature_base",
                "dataset_split": train_label,
                "rows": int(df_feature_base_train.shape[0]),
                "columns": int(df_feature_base_train.shape[1]),
            },
            {
                "population_stage": "feature_base",
                "dataset_split": test_label,
                "rows": int(df_feature_base_test.shape[0]),
                "columns": int(df_feature_base_test.shape[1]),
            },
            {
                "population_stage": "terminal_cohort",
                "dataset_split": train_label,
                "rows": int(df_model_train.shape[0]),
                "columns": int(df_model_train.shape[1]),
            },
            {
                "population_stage": "terminal_cohort",
                "dataset_split": test_label,
                "rows": int(df_model_test.shape[0]),
                "columns": int(df_model_test.shape[1]),
            },
        ]

        summary_table = pd.DataFrame(rows)

        log_config.emit_log(
            log,
            "[build_modeling_population_summary_table] completed | "
            f"rows={summary_table.shape[0]}",
        )

        return summary_table

    except Exception as exc:
        try:
            log_config.emit_log(
                log,
                f"[build_modeling_population_summary_table][error] "
                f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise


def build_target_distribution_table(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    target_column: str = "target_default",
    train_label: str = "train",
    test_label: str = "test",
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a target distribution summary for train and test splits.
    """
    try:
        required_inputs = {
            train_label: df_train,
            test_label: df_test,
        }

        summary_rows: list[dict[str, int | float | str]] = []

        for dataset_split, df_split in required_inputs.items():
            if df_split is None:
                raise ValueError(f"build_target_distribution_table: {dataset_split} dataframe must not be None")

            if target_column not in df_split.columns:
                raise KeyError(
                    "build_target_distribution_table: missing required column "
                    f"'{target_column}' in {dataset_split} dataframe"
                )

            positive_count = int(df_split[target_column].sum())
            negative_count = int((df_split[target_column] == 0).sum())
            default_rate_percent = round(float(df_split[target_column].mean()) * 100, 2)

            summary_rows.append(
                {
                    "dataset_split": dataset_split,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "default_rate_percent": default_rate_percent,
                }
            )

        target_distribution_table = pd.DataFrame(summary_rows)

        log_config.emit_log(
            log,
            "[build_target_distribution_table] completed | "
            f"rows={target_distribution_table.shape[0]}",
        )

        return target_distribution_table

    except Exception as exc:
        try:
            log_config.emit_log(
                log,
                f"[build_target_distribution_table][error] "
                f"Error={type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise
    

def build_numeric_skewness_summary(
    df_train: pd.DataFrame,
    numeric_features: list[str],
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a summary table for numeric feature distributions to support
    log-transformation decisions.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataset containing numeric model features.
    numeric_features : list[str]
        Numeric feature columns to summarize.
    log : Callable[[str], None] | Path | str | None, default=None
        Callable logger or log file path supported by log_config.emit_log.

    Returns
    -------
    pd.DataFrame
        Summary table describing skewness and scale for numeric features.
    """
    try:
        missing_features = [
            feature_name
            for feature_name in numeric_features
            if feature_name not in df_train.columns
        ]
        if missing_features:
            raise KeyError(
                "Numeric features not found in training data: "
                f"{missing_features}"
            )

        summary_records: list[dict[str, int | float | str]] = []

        for feature_name in numeric_features:
            feature_series = df_train[feature_name].dropna().copy()

            if feature_series.empty:
                summary_records.append(
                    {
                        "feature_name": feature_name,
                        "non_null_count": 0,
                        "skewness": float("nan"),
                        "min_value": float("nan"),
                        "median_value": float("nan"),
                        "mean_value": float("nan"),
                        "max_value": float("nan"),
                    }
                )
                continue

            summary_records.append(
                {
                    "feature_name": feature_name,
                    "non_null_count": int(feature_series.shape[0]),
                    "skewness": float(feature_series.skew()),
                    "min_value": float(feature_series.min()),
                    "median_value": float(feature_series.median()),
                    "mean_value": float(feature_series.mean()),
                    "max_value": float(feature_series.max()),
                }
            )

        numeric_skewness_summary_df = (
            pd.DataFrame(summary_records)
            .sort_values(by="skewness", ascending=False)
            .reset_index(drop=True)
            .copy()
        )

        log_config.emit_log(
            log=log,
            message=(
                "[numeric_skewness_summary] "
                f"Built skewness summary for {numeric_skewness_summary_df.shape[0]} numeric features."
            ),
        )

        return numeric_skewness_summary_df

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=f"[numeric_skewness_summary] Failed to build summary: {exc}",
        )
        raise
    

def build_threshold_metrics_artifact(
    model,
    X_data: pd.DataFrame,
    y_true: pd.Series,
    model_name: str,
    dataset_name: str,
    thresholds: list[float],
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a threshold comparison artifact containing classification metrics
    for a fitted binary classification model.
    """
    try:
        if model is None:
            raise ValueError("model must not be None.")

        if X_data is None:
            raise ValueError("X_data must not be None.")

        if y_true is None:
            raise ValueError("y_true must not be None.")

        if not isinstance(X_data, pd.DataFrame):
            raise ValueError("X_data must be a pandas DataFrame.")

        if not isinstance(y_true, pd.Series):
            raise ValueError("y_true must be a pandas Series.")

        if X_data.empty:
            raise ValueError("X_data must not be empty.")

        if y_true.empty:
            raise ValueError("y_true must not be empty.")

        if len(X_data) != len(y_true):
            raise ValueError(
                f"X_data and y_true row counts must match. "
                f"Got len(X_data)={len(X_data)} and len(y_true)={len(y_true)}."
            )

        if not X_data.index.equals(y_true.index):
            raise ValueError("X_data and y_true indices must match.")

        if not model_name or not str(model_name).strip():
            raise ValueError("model_name must be a non-empty string.")

        if not dataset_name or not str(dataset_name).strip():
            raise ValueError("dataset_name must be a non-empty string.")

        if not isinstance(thresholds, list) or not thresholds:
            raise ValueError("thresholds must be a non-empty list of numeric values.")

        validated_thresholds = []
        for threshold_value in thresholds:
            if not isinstance(threshold_value, (int, float)):
                raise ValueError("All thresholds must be numeric values.")
            threshold_float = float(threshold_value)
            if not 0.0 <= threshold_float <= 1.0:
                raise ValueError("All thresholds must lie between 0.0 and 1.0.")
            validated_thresholds.append(threshold_float)

        if len(validated_thresholds) != len(set(validated_thresholds)):
            raise ValueError("thresholds must not contain duplicate values.")

        validated_thresholds = sorted(validated_thresholds)

        metrics_tables: list[pd.DataFrame] = []

        for threshold_value in validated_thresholds:
            prediction_dataframe = em.generate_model_predictions(
                model=model,
                X_data=X_data,
                dataset_name=dataset_name,
                model_name=model_name,
                log=log,
                threshold=threshold_value,
            )

            metrics_dataframe = em.build_classification_metrics_table(
                y_true=y_true,
                prediction_dataframe=prediction_dataframe,
                model_name=model_name,
                dataset_name=dataset_name,
                log=log,
            ).copy()

            metrics_dataframe["model_name"] = model_name
            metrics_dataframe["dataset_name"] = dataset_name
            metrics_dataframe["threshold"] = threshold_value

            metrics_dataframe = metrics_dataframe[
                [
                    "model_name",
                    "dataset_name",
                    "threshold",
                    "roc_auc",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "brier_score",
                    "true_negative",
                    "false_positive",
                    "false_negative",
                    "true_positive",
                ]
            ]

            metrics_tables.append(metrics_dataframe)

        threshold_metrics_df = pd.concat(
            metrics_tables,
            axis=0,
            ignore_index=True,
        ).sort_values(by=["model_name", "dataset_name", "threshold"]).reset_index(drop=True)

        log_config.emit_log(
            log=log,
            message=(
                "[build_threshold_metrics_artifact] "
                f"model={model_name} dataset={dataset_name} "
                f"threshold_count={len(validated_thresholds)} rows={threshold_metrics_df.shape[0]}"
            ),
        )

        return threshold_metrics_df

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=(
                "[build_threshold_metrics_artifact][error] "
                f"model={model_name if 'model_name' in locals() else 'unknown'} "
                f"dataset={dataset_name if 'dataset_name' in locals() else 'unknown'} "
                f"Error={type(exc).__name__}: {exc}"
            ),
        )
        raise
    

def build_threshold_loan_outcomes_artifact(
    model,
    X_data: pd.DataFrame,
    y_true: pd.Series,
    loan_amounts: pd.Series,
    model_name: str,
    dataset_name: str,
    thresholds: list[float],
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a threshold comparison artifact containing loan outcome summaries
    for a fitted binary classification model.
    """
    try:
        if model is None:
            raise ValueError("model must not be None.")

        if X_data is None:
            raise ValueError("X_data must not be None.")

        if y_true is None:
            raise ValueError("y_true must not be None.")

        if loan_amounts is None:
            raise ValueError("loan_amounts must not be None.")

        if not isinstance(X_data, pd.DataFrame):
            raise ValueError("X_data must be a pandas DataFrame.")

        if not isinstance(y_true, pd.Series):
            raise ValueError("y_true must be a pandas Series.")

        if not isinstance(loan_amounts, pd.Series):
            raise ValueError("loan_amounts must be a pandas Series.")

        if X_data.empty:
            raise ValueError("X_data must not be empty.")

        if y_true.empty:
            raise ValueError("y_true must not be empty.")

        if loan_amounts.empty:
            raise ValueError("loan_amounts must not be empty.")

        if len(X_data) != len(y_true):
            raise ValueError(
                f"X_data and y_true row counts must match. "
                f"Got len(X_data)={len(X_data)} and len(y_true)={len(y_true)}."
            )

        if len(y_true) != len(loan_amounts):
            raise ValueError(
                f"y_true and loan_amounts row counts must match. "
                f"Got len(y_true)={len(y_true)} and len(loan_amounts)={len(loan_amounts)}."
            )

        if not X_data.index.equals(y_true.index):
            raise ValueError("X_data and y_true indices must match.")

        if not y_true.index.equals(loan_amounts.index):
            raise ValueError("y_true and loan_amounts indices must match.")

        if not model_name or not str(model_name).strip():
            raise ValueError("model_name must be a non-empty string.")

        if not dataset_name or not str(dataset_name).strip():
            raise ValueError("dataset_name must be a non-empty string.")

        if not isinstance(thresholds, list) or not thresholds:
            raise ValueError("thresholds must be a non-empty list of numeric values.")

        validated_thresholds = []
        for threshold_value in thresholds:
            if not isinstance(threshold_value, (int, float)):
                raise ValueError("All thresholds must be numeric values.")
            threshold_float = float(threshold_value)
            if not 0.0 <= threshold_float <= 1.0:
                raise ValueError("All thresholds must lie between 0.0 and 1.0.")
            validated_thresholds.append(threshold_float)

        if len(validated_thresholds) != len(set(validated_thresholds)):
            raise ValueError("thresholds must not contain duplicate values.")

        validated_thresholds = sorted(validated_thresholds)

        loan_outcome_tables: list[pd.DataFrame] = []

        for threshold_value in validated_thresholds:
            prediction_dataframe = em.generate_model_predictions(
                model=model,
                X_data=X_data,
                dataset_name=dataset_name,
                model_name=model_name,
                log=log,
                threshold=threshold_value,
            )

            outcome_dataframe = em.build_prediction_outcome_table(
                y_true=y_true,
                prediction_dataframe=prediction_dataframe,
                loan_amounts=loan_amounts,
                model_name=model_name,
                dataset_name=dataset_name,
                log=log,
                threshold=threshold_value,
            )

            loan_outcome_summary_dataframe = em.summarize_outcomes_by_loan_amount(
                outcome_dataframe=outcome_dataframe,
                log=log,
            ).copy()

            loan_outcome_summary_dataframe["model_name"] = model_name
            loan_outcome_summary_dataframe["dataset_name"] = dataset_name
            loan_outcome_summary_dataframe["threshold"] = threshold_value

            loan_outcome_summary_dataframe = loan_outcome_summary_dataframe[
                [
                    "model_name",
                    "dataset_name",
                    "threshold",
                    "outcome_type",
                    "row_count",
                    "total_loan_amnt",
                    "average_loan_amnt",
                    "median_loan_amnt",
                ]
            ]

            loan_outcome_tables.append(loan_outcome_summary_dataframe)

        threshold_loan_outcomes_df = pd.concat(
            loan_outcome_tables,
            axis=0,
            ignore_index=True,
        ).sort_values(
            by=["model_name", "dataset_name", "threshold", "outcome_type"]
        ).reset_index(drop=True)

        log_config.emit_log(
            log=log,
            message=(
                "[build_threshold_loan_outcomes_artifact] "
                f"model={model_name} dataset={dataset_name} "
                f"threshold_count={len(validated_thresholds)} rows={threshold_loan_outcomes_df.shape[0]}"
            ),
        )

        return threshold_loan_outcomes_df

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=(
                "[build_threshold_loan_outcomes_artifact][error] "
                f"model={model_name if 'model_name' in locals() else 'unknown'} "
                f"dataset={dataset_name if 'dataset_name' in locals() else 'unknown'} "
                f"Error={type(exc).__name__}: {exc}"
            ),
        )
        raise
    

def reshape_threshold_loan_outcomes_wide(
    threshold_loan_outcomes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reshape threshold loan outcomes from long format to wide format.
    """
    if threshold_loan_outcomes_df is None:
        raise ValueError("threshold_loan_outcomes_df must not be None.")

    if threshold_loan_outcomes_df.empty:
        raise ValueError("threshold_loan_outcomes_df must not be empty.")

    required_columns = {
        "model_name",
        "dataset_name",
        "threshold",
        "outcome_type",
        "row_count",
        "total_loan_amnt",
        "average_loan_amnt",
        "median_loan_amnt",
    }
    missing_columns = required_columns - set(threshold_loan_outcomes_df.columns)
    if missing_columns:
        raise KeyError(
            f"threshold_loan_outcomes_df missing required columns: {sorted(missing_columns)}"
        )

    threshold_loan_outcomes_wide_df = (
        threshold_loan_outcomes_df
        .pivot_table(
            index=["model_name", "dataset_name", "threshold"],
            columns="outcome_type",
            values=["row_count", "total_loan_amnt", "average_loan_amnt", "median_loan_amnt"],
            aggfunc="first",
        )
    )

    threshold_loan_outcomes_wide_df.columns = [
        f"{metric_name}_{outcome_type}"
        for metric_name, outcome_type in threshold_loan_outcomes_wide_df.columns
    ]

    return (
        threshold_loan_outcomes_wide_df
        .reset_index()
        .sort_values(by=["model_name", "dataset_name", "threshold"])
        .reset_index(drop=True)
    )
