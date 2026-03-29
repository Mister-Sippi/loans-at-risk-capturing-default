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
