from __future__ import annotations

from typing import Callable, Sequence
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_curve, auc

import config.logging as log_config
import modeling.evaluate_models as em
import validation.metrics as vm


def build_subgrade_distribution_table(
    df_subgrade_distribution: pd.DataFrame,
    split_column: str,
    grade_column: str,
    subgrade_column: str,
    count_column: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a decision-readable subgrade distribution table by pivoting subgrades
    into columns.

    Parameters
    ----------
    df_subgrade_distribution : pd.DataFrame
        Long-form subgrade distribution table containing split, grade, subgrade,
        and loan count.
    split_column : str
        Name of the split column (e.g., train/test).
    grade_column : str
        Name of the grade column.
    subgrade_column : str
        Name of the subgrade column.
    count_column : str
        Name of the count column.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by ``log_config.emit_log``.

    Returns
    -------
    pd.DataFrame
        Pivoted subgrade distribution table with one row per split-grade
        combination and one column per subgrade.

    Raises
    ------
    ValueError
        If required columns are missing or duplicate split-grade-subgrade
        combinations are present.
    """
    try:
        log_config.emit_log(log, "Building subgrade distribution table")

        required_columns = {
            split_column,
            grade_column,
            subgrade_column,
            count_column,
        }
        missing_columns = required_columns - set(df_subgrade_distribution.columns)

        if missing_columns:
            raise ValueError(
                {
                    "stage": "build_subgrade_distribution_table",
                    "error": "missing_required_columns",
                    "missing_columns": sorted(missing_columns),
                }
            )

        df_subgrade_distribution_local = df_subgrade_distribution.copy()

        duplicate_mask = df_subgrade_distribution_local.duplicated(
            subset=[split_column, grade_column, subgrade_column]
        )
        if duplicate_mask.any():
            raise ValueError(
                {
                    "stage": "build_subgrade_distribution_table",
                    "error": "duplicate_split_grade_subgrade_combinations",
                    "duplicate_rows": int(duplicate_mask.sum()),
                }
            )

        df_subgrade_distribution_table = (
            df_subgrade_distribution_local.pivot(
                index=[split_column, grade_column],
                columns=subgrade_column,
                values=count_column,
            )
            .fillna(0)
            .reset_index()
        )

        subgrade_columns = [
            column
            for column in df_subgrade_distribution_table.columns
            if column not in {split_column, grade_column}
        ]

        df_subgrade_distribution_table[subgrade_columns] = (
            df_subgrade_distribution_table[subgrade_columns].astype(int)
        )

        df_subgrade_distribution_table.columns.name = None

        log_config.emit_log(
            log,
            {
                "stage": "build_subgrade_distribution_table_complete",
                "rows": df_subgrade_distribution_table.shape[0],
                "columns": df_subgrade_distribution_table.shape[1],
            },
        )

        return df_subgrade_distribution_table

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_subgrade_distribution_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_normalized_subgrade_distribution_table(
    df_subgrade_distribution_table: pd.DataFrame,
    split_column: str,
    grade_column: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Normalize subgrade distribution table by row (within each grade).

    Returns proportions per subgrade.
    """

    try:
        log_config.emit_log(log, "Building normalized subgrade distribution table")

        required_columns = {split_column, grade_column}
        missing_columns = required_columns - set(df_subgrade_distribution_table.columns)

        if missing_columns:
            raise ValueError(
                {
                    "stage": "build_normalized_subgrade_distribution_table",
                    "error": "missing_required_columns",
                    "missing_columns": sorted(missing_columns),
                }
            )

        df_local = df_subgrade_distribution_table.copy()

        subgrade_columns = [
            column
            for column in df_local.columns
            if column not in {split_column, grade_column}
        ]

        row_totals = df_local[subgrade_columns].sum(axis=1)

        df_local[subgrade_columns] = df_local[subgrade_columns].div(row_totals, axis=0)

        log_config.emit_log(
            log,
            {
                "stage": "build_normalized_subgrade_distribution_table_complete",
                "rows": df_local.shape[0],
                "columns": df_local.shape[1],
            },
        )

        return df_local

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_normalized_subgrade_distribution_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_default_rate_by_subgrade_table(
    df_default_rate_by_subgrade: pd.DataFrame,
    split_column: str,
    subgrade_column: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a decision-readable default-rate-by-subgrade artifact table.

    Parameters
    ----------
    df_default_rate_by_subgrade : pd.DataFrame
        Combined default-rate table containing split, subgrade, counts,
        and default rate.
    split_column : str
        Name of the split column (e.g., train/test).
    subgrade_column : str
        Name of the subgrade column.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by ``log_config.emit_log``.

    Returns
    -------
    pd.DataFrame
        Sorted artifact table with train displayed before test and subgrades
        ordered from a1 to g5.

    Raises
    ------
    ValueError
        If required columns are missing or duplicate split-subgrade rows exist.
    """
    try:
        log_config.emit_log(log, "Building default rate by subgrade artifact table")

        required_columns = {
            split_column,
            subgrade_column,
            "loan_count",
            "default_count",
            "non_default_count",
            "default_rate",
        }
        missing_columns = required_columns - set(df_default_rate_by_subgrade.columns)

        if missing_columns:
            raise ValueError(
                {
                    "stage": "build_default_rate_by_subgrade_table",
                    "error": "missing_required_columns",
                    "missing_columns": sorted(missing_columns),
                }
            )

        df_default_rate_local = df_default_rate_by_subgrade.copy()

        duplicate_mask = df_default_rate_local.duplicated(
            subset=[split_column, subgrade_column]
        )

        if duplicate_mask.any():
            raise ValueError(
                {
                    "stage": "build_default_rate_by_subgrade_table",
                    "error": "duplicate_split_subgrade_rows",
                    "duplicate_rows": int(duplicate_mask.sum()),
                }
            )

        ordered_subgrades = [
            f"{grade}{subgrade_number}"
            for grade in ["a", "b", "c", "d", "e", "f", "g"]
            for subgrade_number in [1, 2, 3, 4, 5]
        ]

        df_default_rate_local[split_column] = pd.Categorical(
            df_default_rate_local[split_column],
            categories=["train", "test"],
            ordered=True,
        )

        df_default_rate_local[subgrade_column] = pd.Categorical(
            df_default_rate_local[subgrade_column],
            categories=ordered_subgrades,
            ordered=True,
        )

        df_default_rate_table = df_default_rate_local.sort_values(
            [split_column, subgrade_column]
        ).reset_index(drop=True)

        log_config.emit_log(
            log,
            {
                "stage": "build_default_rate_by_subgrade_table_complete",
                "rows": df_default_rate_table.shape[0],
                "columns": df_default_rate_table.shape[1],
            },
        )

        return df_default_rate_table

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_default_rate_by_subgrade_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def compute_baseline_roc_auc_from_subgrade(
    y_true: pd.Series,
    subgrade_series: pd.Series,
    system_name: str,
    dataset_name: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Compute ROC curve points and AUC for the subgrade-based baseline system.

    Subgrades are first converted to a numeric risk rank, which is then used
    as the baseline score.
    """

    try:
        log_config.emit_log(
            log,
            f"Computing baseline ROC AUC from subgrade for system={system_name}, dataset={dataset_name}",
        )

        if y_true is None:
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "y_true_is_none",
                }
            )

        if subgrade_series is None:
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "subgrade_series_is_none",
                }
            )

        if not isinstance(y_true, pd.Series):
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "y_true_is_not_series",
                    "input_type": type(y_true).__name__,
                }
            )

        if not isinstance(subgrade_series, pd.Series):
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "subgrade_series_is_not_series",
                    "input_type": type(subgrade_series).__name__,
                }
            )

        if y_true.empty:
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "y_true_is_empty",
                }
            )

        if subgrade_series.empty:
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "subgrade_series_is_empty",
                }
            )

        if len(y_true) != len(subgrade_series):
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "length_mismatch",
                    "y_true_rows": len(y_true),
                    "subgrade_rows": len(subgrade_series),
                }
            )

        if not y_true.index.equals(subgrade_series.index):
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "index_mismatch",
                }
            )

        if y_true.isna().any():
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "missing_y_true_values",
                    "missing_rows": int(y_true.isna().sum()),
                }
            )

        observed_target_values = set(y_true.unique().tolist())
        if not observed_target_values.issubset({0, 1}):
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "invalid_target_values",
                    "observed_values": sorted(observed_target_values),
                }
            )

        subgrade_rank_series = vm.map_subgrade_to_rank(
            subgrade_series=subgrade_series,
            log=log,
        )

        false_positive_rate, true_positive_rate, threshold_values = roc_curve(
            y_true=y_true,
            y_score=subgrade_rank_series,
        )

        auc_value = auc(false_positive_rate, true_positive_rate)

        roc_auc_dataframe = pd.DataFrame(
            {
                "system_name": system_name,
                "dataset_name": dataset_name,
                "false_positive_rate": false_positive_rate,
                "true_positive_rate": true_positive_rate,
                "threshold": threshold_values,
                "auc": auc_value,
            }
        )

        log_config.emit_log(
            log,
            {
                "stage": "compute_baseline_roc_auc_from_subgrade_complete",
                "system_name": system_name,
                "dataset_name": dataset_name,
                "rows": roc_auc_dataframe.shape[0],
                "auc": float(auc_value),
            },
        )

        return roc_auc_dataframe

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "compute_baseline_roc_auc_from_subgrade_failed",
                "system_name": system_name if "system_name" in locals() else "unknown",
                "dataset_name": dataset_name if "dataset_name" in locals() else "unknown",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_risk_separation_summary_table(
    roc_auc_artifact_df: pd.DataFrame,
    system_name_column: str,
    dataset_name_column: str,
    auc_column: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a compact summary table for risk separation results.

    The input is expected to contain ROC curve points with the AUC value
    repeated across rows for each system-dataset combination.
    """

    try:
        log_config.emit_log(log, "Building risk separation summary table")

        if roc_auc_artifact_df is None:
            raise ValueError(
                {
                    "stage": "build_risk_separation_summary_table",
                    "error": "input_is_none",
                }
            )

        if roc_auc_artifact_df.empty:
            raise ValueError(
                {
                    "stage": "build_risk_separation_summary_table",
                    "error": "input_is_empty",
                }
            )

        required_columns = {
            system_name_column,
            dataset_name_column,
            auc_column,
        }
        missing_columns = required_columns - set(roc_auc_artifact_df.columns)

        if missing_columns:
            raise KeyError(
                {
                    "stage": "build_risk_separation_summary_table",
                    "error": "missing_required_columns",
                    "missing_columns": sorted(missing_columns),
                }
            )

        df_risk_separation_summary = (
            roc_auc_artifact_df[
                [system_name_column, dataset_name_column, auc_column]
            ]
            .drop_duplicates()
            .sort_values(by=[system_name_column, dataset_name_column])
            .reset_index(drop=True)
        )

        duplicate_mask = df_risk_separation_summary.duplicated(
            subset=[system_name_column, dataset_name_column]
        )

        if duplicate_mask.any():
            raise ValueError(
                {
                    "stage": "build_risk_separation_summary_table",
                    "error": "duplicate_system_dataset_rows",
                    "duplicate_rows": int(duplicate_mask.sum()),
                }
            )

        df_risk_separation_summary[dataset_name_column] = pd.Categorical(
            df_risk_separation_summary[dataset_name_column],
            categories=["train", "test"],
            ordered=True,
        )

        df_risk_separation_summary = (
            df_risk_separation_summary
            .sort_values(by=[dataset_name_column, system_name_column])
            .reset_index(drop=True)
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_risk_separation_summary_table_complete",
                "rows": df_risk_separation_summary.shape[0],
                "columns": df_risk_separation_summary.shape[1],
            },
        )

        return df_risk_separation_summary

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_risk_separation_summary_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_roc_curve_artifact_table(
    roc_auc_artifact_frames: list[pd.DataFrame],
    system_name_column: str,
    dataset_name_column: str,
    false_positive_rate_column: str,
    true_positive_rate_column: str,
    threshold_column: str,
    auc_column: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a combined ROC curve artifact table from multiple ROC/AUC result frames.
    """

    try:
        log_config.emit_log(log, "Building ROC curve artifact table")

        if roc_auc_artifact_frames is None:
            raise ValueError(
                {
                    "stage": "build_roc_curve_artifact_table",
                    "error": "input_is_none",
                }
            )

        if not isinstance(roc_auc_artifact_frames, list):
            raise ValueError(
                {
                    "stage": "build_roc_curve_artifact_table",
                    "error": "input_is_not_list",
                    "input_type": type(roc_auc_artifact_frames).__name__,
                }
            )

        if not roc_auc_artifact_frames:
            raise ValueError(
                {
                    "stage": "build_roc_curve_artifact_table",
                    "error": "input_is_empty_list",
                }
            )

        for frame_index, roc_auc_frame in enumerate(roc_auc_artifact_frames):
            if roc_auc_frame is None:
                raise ValueError(
                    {
                        "stage": "build_roc_curve_artifact_table",
                        "error": "frame_is_none",
                        "frame_index": frame_index,
                    }
                )

            if not isinstance(roc_auc_frame, pd.DataFrame):
                raise ValueError(
                    {
                        "stage": "build_roc_curve_artifact_table",
                        "error": "frame_is_not_dataframe",
                        "frame_index": frame_index,
                        "input_type": type(roc_auc_frame).__name__,
                    }
                )

            if roc_auc_frame.empty:
                raise ValueError(
                    {
                        "stage": "build_roc_curve_artifact_table",
                        "error": "frame_is_empty",
                        "frame_index": frame_index,
                    }
                )

        df_roc_curve_artifact = pd.concat(
            roc_auc_artifact_frames,
            axis=0,
            ignore_index=True,
        )

        required_columns = {
            system_name_column,
            dataset_name_column,
            false_positive_rate_column,
            true_positive_rate_column,
            threshold_column,
            auc_column,
        }
        missing_columns = required_columns - set(df_roc_curve_artifact.columns)

        if missing_columns:
            raise KeyError(
                {
                    "stage": "build_roc_curve_artifact_table",
                    "error": "missing_required_columns",
                    "missing_columns": sorted(missing_columns),
                }
            )

        df_roc_curve_artifact[dataset_name_column] = pd.Categorical(
            df_roc_curve_artifact[dataset_name_column],
            categories=["train", "test"],
            ordered=True,
        )

        df_roc_curve_artifact = (
            df_roc_curve_artifact
            .sort_values(
                by=[
                    dataset_name_column,
                    system_name_column,
                    false_positive_rate_column,
                    true_positive_rate_column,
                ]
            )
            .reset_index(drop=True)
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_roc_curve_artifact_table_complete",
                "rows": df_roc_curve_artifact.shape[0],
                "columns": df_roc_curve_artifact.shape[1],
            },
        )

        return df_roc_curve_artifact

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_roc_curve_artifact_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_calibration_artifact_table(
    calibration_tables: list[pd.DataFrame],
    system_name_column: str,
    dataset_name_column: str,
    bin_order_column: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a combined calibration artifact table from one or more calibration tables.
    """

    try:
        log_config.emit_log(log, "Building calibration artifact table")

        if calibration_tables is None:
            raise ValueError(
                {
                    "stage": "build_calibration_artifact_table",
                    "error": "input_is_none",
                }
            )

        if not isinstance(calibration_tables, list):
            raise ValueError(
                {
                    "stage": "build_calibration_artifact_table",
                    "error": "input_is_not_list",
                    "input_type": type(calibration_tables).__name__,
                }
            )

        if not calibration_tables:
            raise ValueError(
                {
                    "stage": "build_calibration_artifact_table",
                    "error": "input_is_empty_list",
                }
            )

        for table_index, calibration_table in enumerate(calibration_tables):
            if calibration_table is None:
                raise ValueError(
                    {
                        "stage": "build_calibration_artifact_table",
                        "error": "table_is_none",
                        "table_index": table_index,
                    }
                )

            if not isinstance(calibration_table, pd.DataFrame):
                raise ValueError(
                    {
                        "stage": "build_calibration_artifact_table",
                        "error": "table_is_not_dataframe",
                        "table_index": table_index,
                        "input_type": type(calibration_table).__name__,
                    }
                )

            if calibration_table.empty:
                raise ValueError(
                    {
                        "stage": "build_calibration_artifact_table",
                        "error": "table_is_empty",
                        "table_index": table_index,
                    }
                )

        df_calibration_artifact = pd.concat(
            calibration_tables,
            axis=0,
            ignore_index=True,
        )

        required_columns = {
            system_name_column,
            dataset_name_column,
            bin_order_column,
            "row_count",
            "predicted_probability_mean",
            "observed_default_rate",
        }
        missing_columns = required_columns - set(df_calibration_artifact.columns)

        if missing_columns:
            raise KeyError(
                {
                    "stage": "build_calibration_artifact_table",
                    "error": "missing_required_columns",
                    "missing_columns": sorted(missing_columns),
                }
            )

        df_calibration_artifact[dataset_name_column] = pd.Categorical(
            df_calibration_artifact[dataset_name_column],
            categories=["train", "test"],
            ordered=True,
        )

        df_calibration_artifact = (
            df_calibration_artifact
            .sort_values(by=[dataset_name_column, system_name_column, bin_order_column])
            .reset_index(drop=True)
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_calibration_artifact_table_complete",
                "rows": df_calibration_artifact.shape[0],
                "columns": df_calibration_artifact.shape[1],
            },
        )

        return df_calibration_artifact

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_calibration_artifact_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise