from __future__ import annotations

from typing import Callable
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_curve, auc

import config.logging as log_config
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
    split_column: str = "split",
    grade_column: str = "grade",
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a normalized subgrade distribution table for validation artifacts.

    This function standardizes the construction of the normalized subgrade
    distribution artifact by applying consistent split ordering and stable
    sorting after normalization. It ensures that the returned table is directly
    usable for validation display, comparison, and artifact export without
    additional preparation in the notebook.

    Parameters
    ----------
    df_subgrade_distribution_table : pd.DataFrame
        Combined subgrade distribution table containing train/test splits.
    split_column : str, default="split"
        Name of the split indicator column.
    grade_column : str, default="grade"
        Name of the grade column used for stable sorting.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Normalized subgrade distribution table with enforced training → testing
        ordering and stable sorting by split and grade.

    Notes
    -----
    This function is used in the Validation notebook (Baseline Definition section)
    to ensure consistent normalized subgrade distribution artifact construction.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "build_normalized_subgrade_distribution_table_started",
                "input_rows": df_subgrade_distribution_table.shape[0],
                "input_columns": df_subgrade_distribution_table.shape[1],
                "split_column": split_column,
                "grade_column": grade_column,
            },
        )

        df_subgrade_distribution_normalized = df_subgrade_distribution_table.copy()

        numeric_value_columns = [
            column_name
            for column_name in df_subgrade_distribution_normalized.columns
            if column_name not in {split_column, grade_column}
        ]

        if not numeric_value_columns:
            raise ValueError("No subgrade columns were found for normalization.")

        row_totals = df_subgrade_distribution_normalized[numeric_value_columns].sum(axis=1)

        if (row_totals == 0).any():
            zero_total_rows = df_subgrade_distribution_normalized.loc[
                row_totals == 0, [split_column, grade_column]
            ].to_dict(orient="records")
            raise ValueError(
                f"Cannot normalize subgrade distribution because one or more rows sum to zero: {zero_total_rows}"
            )

        df_subgrade_distribution_normalized[numeric_value_columns] = (
            df_subgrade_distribution_normalized[numeric_value_columns]
            .div(row_totals, axis=0)
        )

        df_subgrade_distribution_normalized[split_column] = pd.Categorical(
            df_subgrade_distribution_normalized[split_column],
            categories=["train", "test"],
            ordered=True,
        )

        df_subgrade_distribution_normalized = (
            df_subgrade_distribution_normalized
            .sort_values([split_column, grade_column])
            .reset_index(drop=True)
            .copy()
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_normalized_subgrade_distribution_table_completed",
                "rows": df_subgrade_distribution_normalized.shape[0],
                "columns": df_subgrade_distribution_normalized.shape[1],
                "normalized_columns": numeric_value_columns,
            },
        )

        return df_subgrade_distribution_normalized

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


def build_grade_structure_table(
    df_grade_structure_train: pd.DataFrame,
    df_grade_structure_test: pd.DataFrame,
    split_column: str = "split",
    split_order: tuple[str, str] = ("train", "test"),
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a combined grade structure table for training and testing datasets.

    This function standardizes the construction of a cross-split grade structure
    artifact by applying consistent split labeling, concatenation, and ordering.
    It ensures that the returned table is directly usable for validation display
    and comparison without additional preparation in the notebook.

    Parameters
    ----------
    df_grade_structure_train : pd.DataFrame
        Grade structure summary for the training dataset.
    df_grade_structure_test : pd.DataFrame
        Grade structure summary for the testing dataset.
    split_column : str, default="split"
        Name of the split indicator column to add.
    split_order : tuple[str, str], default=("train", "test")
        Ordered labels used to identify and enforce training → testing ordering.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Combined grade structure table with explicit split labeling and
        enforced training → testing ordering.

    Notes
    -----
    This function is used in the Validation notebook (Baseline Definition section)
    to ensure consistent grade structure artifact construction across train/test splits.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "build_grade_structure_table_started",
                "train_rows": df_grade_structure_train.shape[0],
                "train_columns": df_grade_structure_train.shape[1],
                "test_rows": df_grade_structure_test.shape[0],
                "test_columns": df_grade_structure_test.shape[1],
                "split_column": split_column,
                "split_order": list(split_order),
            },
        )

        if len(split_order) != 2:
            raise ValueError(
                "split_order must contain exactly two labels for training and testing."
            )

        df_grade_structure_train_labeled = df_grade_structure_train.copy()
        df_grade_structure_train_labeled[split_column] = split_order[0]

        df_grade_structure_test_labeled = df_grade_structure_test.copy()
        df_grade_structure_test_labeled[split_column] = split_order[1]

        df_grade_structure_combined = pd.concat(
            [df_grade_structure_train_labeled, df_grade_structure_test_labeled],
            axis=0,
            ignore_index=True,
        ).copy()

        df_grade_structure_combined[split_column] = pd.Categorical(
            df_grade_structure_combined[split_column],
            categories=list(split_order),
            ordered=True,
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_grade_structure_table_completed",
                "rows": df_grade_structure_combined.shape[0],
                "columns": df_grade_structure_combined.shape[1],
                "splits_present": (
                    df_grade_structure_combined[split_column]
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                ),
            },
        )

        return df_grade_structure_combined

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_grade_structure_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_subgrade_distribution_artifact_table(
    df_subgrade_distribution_train: pd.DataFrame,
    df_subgrade_distribution_test: pd.DataFrame,
    split_column: str = "split",
    grade_column: str = "grade",
    subgrade_column: str = "sub_grade",
    count_column: str = "loan_count",
    split_order: tuple[str, str] = ("train", "test"),
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a combined subgrade distribution artifact table for training and testing datasets.

    This function standardizes the construction of a cross-split subgrade
    distribution artifact by applying consistent split labeling, concatenation,
    and ordering before passing the result into the artifact table builder.
    It ensures that the returned table is directly usable for validation display,
    comparison, and artifact export without additional preparation in the notebook.

    Parameters
    ----------
    df_subgrade_distribution_train : pd.DataFrame
        Subgrade distribution summary for the training dataset.
    df_subgrade_distribution_test : pd.DataFrame
        Subgrade distribution summary for the testing dataset.
    split_column : str, default="split"
        Name of the split indicator column to add.
    grade_column : str, default="grade"
        Name of the grade column.
    subgrade_column : str, default="sub_grade"
        Name of the subgrade column.
    count_column : str, default="loan_count"
        Name of the loan count column.
    split_order : tuple[str, str], default=("train", "test")
        Ordered labels used to identify and enforce training → testing ordering.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Combined subgrade distribution artifact table with explicit split labeling
        and enforced training → testing ordering.

    Notes
    -----
    This function is used in the Validation notebook (Baseline Definition section)
    to ensure consistent subgrade distribution artifact construction across
    train/test splits.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "build_subgrade_distribution_artifact_table_started",
                "train_rows": df_subgrade_distribution_train.shape[0],
                "train_columns": df_subgrade_distribution_train.shape[1],
                "test_rows": df_subgrade_distribution_test.shape[0],
                "test_columns": df_subgrade_distribution_test.shape[1],
                "split_column": split_column,
                "grade_column": grade_column,
                "subgrade_column": subgrade_column,
                "count_column": count_column,
                "split_order": list(split_order),
            },
        )

        if len(split_order) != 2:
            raise ValueError(
                "split_order must contain exactly two labels for training and testing."
            )

        df_subgrade_distribution_train_labeled = df_subgrade_distribution_train.copy()
        df_subgrade_distribution_train_labeled[split_column] = split_order[0]

        df_subgrade_distribution_test_labeled = df_subgrade_distribution_test.copy()
        df_subgrade_distribution_test_labeled[split_column] = split_order[1]

        df_subgrade_distribution_combined = pd.concat(
            [
                df_subgrade_distribution_train_labeled,
                df_subgrade_distribution_test_labeled,
            ],
            axis=0,
            ignore_index=True,
        ).copy()

        df_subgrade_distribution_artifact_table = build_subgrade_distribution_table(
            df_subgrade_distribution=df_subgrade_distribution_combined,
            split_column=split_column,
            grade_column=grade_column,
            subgrade_column=subgrade_column,
            count_column=count_column,
            log=log,
        ).copy()

        df_subgrade_distribution_artifact_table[split_column] = pd.Categorical(
            df_subgrade_distribution_artifact_table[split_column],
            categories=list(split_order),
            ordered=True,
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_subgrade_distribution_artifact_table_completed",
                "rows": df_subgrade_distribution_artifact_table.shape[0],
                "columns": df_subgrade_distribution_artifact_table.shape[1],
                "splits_present": (
                    df_subgrade_distribution_artifact_table[split_column]
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                ),
            },
        )

        return df_subgrade_distribution_artifact_table

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_subgrade_distribution_artifact_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_default_rate_by_subgrade_artifact_table(
    df_default_rate_by_subgrade_train: pd.DataFrame,
    df_default_rate_by_subgrade_test: pd.DataFrame,
    split_column: str = "split",
    subgrade_column: str = "sub_grade",
    split_order: tuple[str, str] = ("train", "test"),
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a combined default-rate-by-subgrade artifact table for training and testing datasets.

    This function standardizes the construction of a cross-split default-rate
    artifact by applying consistent split labeling, concatenation, and ordering
    before passing the result into the default-rate artifact table builder.
    It ensures that the returned table is directly usable for validation display,
    comparison, and artifact export without additional preparation in the notebook.

    Parameters
    ----------
    df_default_rate_by_subgrade_train : pd.DataFrame
        Default-rate-by-subgrade summary for the training dataset.
    df_default_rate_by_subgrade_test : pd.DataFrame
        Default-rate-by-subgrade summary for the testing dataset.
    split_column : str, default="split"
        Name of the split indicator column to add.
    subgrade_column : str, default="sub_grade"
        Name of the subgrade column.
    split_order : tuple[str, str], default=("train", "test")
        Ordered labels used to identify and enforce training → testing ordering.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Combined default-rate-by-subgrade artifact table with explicit split
        labeling and enforced training → testing ordering.

    Notes
    -----
    This function is used in the Validation notebook (Baseline Definition section)
    to ensure consistent default-rate-by-subgrade artifact construction across
    train/test splits.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "build_default_rate_by_subgrade_artifact_table_started",
                "train_rows": df_default_rate_by_subgrade_train.shape[0],
                "train_columns": df_default_rate_by_subgrade_train.shape[1],
                "test_rows": df_default_rate_by_subgrade_test.shape[0],
                "test_columns": df_default_rate_by_subgrade_test.shape[1],
                "split_column": split_column,
                "subgrade_column": subgrade_column,
                "split_order": list(split_order),
            },
        )

        if len(split_order) != 2:
            raise ValueError(
                "split_order must contain exactly two labels for training and testing."
            )

        df_default_rate_by_subgrade_train_labeled = (
            df_default_rate_by_subgrade_train.copy()
        )
        df_default_rate_by_subgrade_train_labeled[split_column] = split_order[0]

        df_default_rate_by_subgrade_test_labeled = (
            df_default_rate_by_subgrade_test.copy()
        )
        df_default_rate_by_subgrade_test_labeled[split_column] = split_order[1]

        df_default_rate_by_subgrade_combined = pd.concat(
            [
                df_default_rate_by_subgrade_train_labeled,
                df_default_rate_by_subgrade_test_labeled,
            ],
            axis=0,
            ignore_index=True,
        ).copy()

        df_default_rate_by_subgrade_table = build_default_rate_by_subgrade_table(
            df_default_rate_by_subgrade=df_default_rate_by_subgrade_combined,
            split_column=split_column,
            subgrade_column=subgrade_column,
            log=log,
        ).copy()

        df_default_rate_by_subgrade_table[split_column] = pd.Categorical(
            df_default_rate_by_subgrade_table[split_column],
            categories=list(split_order),
            ordered=True,
        )

        df_default_rate_by_subgrade_table = (
            df_default_rate_by_subgrade_table
            .sort_values([split_column, subgrade_column])
            .reset_index(drop=True)
            .copy()
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_default_rate_by_subgrade_artifact_table_completed",
                "rows": df_default_rate_by_subgrade_table.shape[0],
                "columns": df_default_rate_by_subgrade_table.shape[1],
                "splits_present": (
                    df_default_rate_by_subgrade_table[split_column]
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                ),
            },
        )

        return df_default_rate_by_subgrade_table

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_default_rate_by_subgrade_artifact_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def summarize_default_rate_by_subgrade_table(
    df_default_rate_by_subgrade_table: pd.DataFrame,
    split_column: str = "split",
    default_rate_column: str = "default_rate",
    split_order: tuple[str, str] = ("train", "test"),
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a descriptive summary of default rates across train/test splits.

    This function standardizes the construction of a split-level descriptive
    summary for the default-rate-by-subgrade artifact table. It ensures that the
    returned summary is directly usable for validation display without additional
    grouping or ordering logic in the notebook.

    Parameters
    ----------
    df_default_rate_by_subgrade_table : pd.DataFrame
        Default-rate-by-subgrade artifact table containing train/test splits.
    split_column : str, default="split"
        Name of the split indicator column.
    default_rate_column : str, default="default_rate"
        Name of the default rate column to summarize.
    split_order : tuple[str, str], default=("train", "test")
        Ordered labels used to identify and enforce training → testing ordering.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Descriptive summary table of default rates by split, with enforced
        training → testing ordering.

    Notes
    -----
    This function is used in the Validation notebook (Baseline Definition section)
    to ensure consistent summary construction for default-rate-by-subgrade artifacts.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "summarize_default_rate_by_subgrade_table_started",
                "input_rows": df_default_rate_by_subgrade_table.shape[0],
                "input_columns": df_default_rate_by_subgrade_table.shape[1],
                "split_column": split_column,
                "default_rate_column": default_rate_column,
                "split_order": list(split_order),
            },
        )

        if len(split_order) != 2:
            raise ValueError(
                "split_order must contain exactly two labels for training and testing."
            )

        if default_rate_column not in df_default_rate_by_subgrade_table.columns:
            raise ValueError(
                f"Column '{default_rate_column}' not found in default-rate-by-subgrade table."
            )

        df_default_rate_by_subgrade_summary = (
            df_default_rate_by_subgrade_table
            .groupby(split_column)[default_rate_column]
            .describe()
            .reset_index()
            .copy()
        )

        df_default_rate_by_subgrade_summary[split_column] = pd.Categorical(
            df_default_rate_by_subgrade_summary[split_column],
            categories=list(split_order),
            ordered=True,
        )

        df_default_rate_by_subgrade_summary = (
            df_default_rate_by_subgrade_summary
            .sort_values(split_column)
            .reset_index(drop=True)
            .copy()
        )

        log_config.emit_log(
            log,
            {
                "stage": "summarize_default_rate_by_subgrade_table_completed",
                "rows": df_default_rate_by_subgrade_summary.shape[0],
                "columns": df_default_rate_by_subgrade_summary.shape[1],
            },
        )

        return df_default_rate_by_subgrade_summary

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "summarize_default_rate_by_subgrade_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_auc_comparison_table(
    df_roc_model_train: pd.DataFrame,
    df_roc_model_test: pd.DataFrame,
    df_roc_baseline_train: pd.DataFrame,
    df_roc_baseline_test: pd.DataFrame,
    split_column: str = "split",
    system_column: str = "system",
    auc_column: str = "auc",
    split_order: tuple[str, str] = ("train", "test"),
    system_order: tuple[str, str] = ("model", "baseline_subgrade"),
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a compact AUC comparison table for model and baseline systems across
    training and testing splits.

    This function standardizes the construction of a display-ready AUC summary
    by extracting scalar AUC values from ROC artifact frames, applying
    consistent split and system labeling, and enforcing stable ordering for
    validation display.

    Parameters
    ----------
    df_roc_model_train : pd.DataFrame
        ROC/AUC artifact frame for the model on the training split.
    df_roc_model_test : pd.DataFrame
        ROC/AUC artifact frame for the model on the testing split.
    df_roc_baseline_train : pd.DataFrame
        ROC/AUC artifact frame for the baseline on the training split.
    df_roc_baseline_test : pd.DataFrame
        ROC/AUC artifact frame for the baseline on the testing split.
    split_column : str, default="split"
        Name of the split column in the returned table.
    system_column : str, default="system"
        Name of the system label column in the returned table.
    auc_column : str, default="auc"
        Name of the AUC value column to extract and return.
    split_order : tuple[str, str], default=("train", "test")
        Ordered split labels used to enforce training → testing ordering.
    system_order : tuple[str, str], default=("model", "baseline_subgrade")
        Ordered system labels used to enforce stable display ordering.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Display-ready AUC comparison table with stable split and system ordering.

    Notes
    -----
    This function is used in the Validation notebook (Risk Evaluation section)
    to support quick verification of model and baseline AUC across train/test
    splits without inline summary assembly in the notebook.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "build_auc_comparison_table_started",
                "split_order": list(split_order),
                "system_order": list(system_order),
                "auc_column": auc_column,
            },
        )

        if len(split_order) != 2:
            raise ValueError(
                "split_order must contain exactly two labels for training and testing."
            )

        auc_input_frames = {
            ("model", "train"): df_roc_model_train,
            ("model", "test"): df_roc_model_test,
            ("baseline_subgrade", "train"): df_roc_baseline_train,
            ("baseline_subgrade", "test"): df_roc_baseline_test,
        }

        auc_records: list[dict[str, float | str]] = []

        for (system_name, split_name), df_auc_source in auc_input_frames.items():
            if auc_column not in df_auc_source.columns:
                raise ValueError(
                    f"Column '{auc_column}' not found for system='{system_name}', split='{split_name}'."
                )

            auc_records.append(
                {
                    system_column: system_name,
                    split_column: split_name,
                    auc_column: float(df_auc_source[auc_column].iloc[0]),
                }
            )

        df_auc_comparison = pd.DataFrame(auc_records).copy()

        df_auc_comparison[split_column] = pd.Categorical(
            df_auc_comparison[split_column],
            categories=list(split_order),
            ordered=True,
        )

        df_auc_comparison[system_column] = pd.Categorical(
            df_auc_comparison[system_column],
            categories=list(system_order),
            ordered=True,
        )

        df_auc_comparison = (
            df_auc_comparison
            .sort_values([split_column, system_column])
            .reset_index(drop=True)
            .copy()
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_auc_comparison_table_completed",
                "rows": df_auc_comparison.shape[0],
                "columns": df_auc_comparison.shape[1],
            },
        )

        return df_auc_comparison

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_auc_comparison_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_model_policy_outcomes_artifact_table(
    model_policy_outcome_tables: list[pd.DataFrame],
    dataset_name_column: str = "dataset_name",
    policy_value_column: str = "policy_value",
    dataset_order: tuple[str, str] = ("train", "test"),
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a combined model policy outcomes artifact table across training and testing splits.

    This function standardizes the construction of the cross-split model policy
    outcomes artifact by concatenating evaluation outputs, enforcing consistent
    dataset ordering, and applying stable sorting. It ensures that the returned
    table is directly usable for validation display, comparison, and artifact
    export without additional preparation in the notebook.

    Parameters
    ----------
    model_policy_outcome_tables : list[pd.DataFrame]
        List of model policy outcome tables generated across thresholds and splits.
    dataset_name_column : str, default="dataset_name"
        Name of the dataset split column.
    policy_value_column : str, default="policy_value"
        Name of the policy value column used for stable sorting.
    dataset_order : tuple[str, str], default=("train", "test")
        Ordered dataset labels used to enforce training → testing ordering.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Combined model policy outcomes artifact table with enforced dataset
        ordering and stable sorting by dataset and policy value.

    Notes
    -----
    This function is used in the Validation notebook (Policy Simulation section)
    to ensure consistent artifact construction across model threshold policies.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "build_model_policy_outcomes_artifact_table_started",
                "table_count": len(model_policy_outcome_tables),
                "dataset_name_column": dataset_name_column,
                "policy_value_column": policy_value_column,
                "dataset_order": list(dataset_order),
            },
        )

        if not model_policy_outcome_tables:
            raise ValueError("model_policy_outcome_tables must contain at least one DataFrame.")

        if len(dataset_order) != 2:
            raise ValueError(
                "dataset_order must contain exactly two labels for training and testing."
            )

        df_model_policy_outcomes = pd.concat(
            model_policy_outcome_tables,
            axis=0,
            ignore_index=True,
        ).copy()

        if dataset_name_column not in df_model_policy_outcomes.columns:
            raise ValueError(
                f"Column '{dataset_name_column}' not found in model policy outcomes table."
            )

        if policy_value_column not in df_model_policy_outcomes.columns:
            raise ValueError(
                f"Column '{policy_value_column}' not found in model policy outcomes table."
            )

        df_model_policy_outcomes[dataset_name_column] = pd.Categorical(
            df_model_policy_outcomes[dataset_name_column],
            categories=list(dataset_order),
            ordered=True,
        )

        df_model_policy_outcomes = (
            df_model_policy_outcomes
            .sort_values([dataset_name_column, policy_value_column])
            .reset_index(drop=True)
            .copy()
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_model_policy_outcomes_artifact_table_completed",
                "rows": df_model_policy_outcomes.shape[0],
                "columns": df_model_policy_outcomes.shape[1],
            },
        )

        return df_model_policy_outcomes

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_model_policy_outcomes_artifact_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_model_policy_outcomes_display_table(
    df_model_policy_outcomes: pd.DataFrame,
    accepted_non_default_loan_amount_column: str = "accepted_non_default_loan_amnt",
    accepted_default_loan_amount_column: str = "accepted_default_loan_amnt",
    rejected_non_default_loan_amount_column: str = "rejected_non_default_loan_amnt",
    net_value_proxy_column: str = "net_value_proxy",
    net_value_proxy_with_opportunity_cost_column: str = "net_value_proxy_with_opportunity_cost",
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a display-ready model policy outcomes table with derived proxy value columns.

    This function standardizes the enrichment of the model policy outcomes
    artifact by adding derived proxy value measures used for validation display
    and economic interpretation. It ensures that the returned table is ready for
    final formatting without additional calculation in the notebook.

    Parameters
    ----------
    df_model_policy_outcomes : pd.DataFrame
        Combined model policy outcomes artifact table.
    accepted_non_default_loan_amount_column : str, default="accepted_non_default_loan_amnt"
        Name of the accepted non-default loan amount column.
    accepted_default_loan_amount_column : str, default="accepted_default_loan_amnt"
        Name of the accepted default loan amount column.
    rejected_non_default_loan_amount_column : str, default="rejected_non_default_loan_amnt"
        Name of the rejected non-default loan amount column.
    net_value_proxy_column : str, default="net_value_proxy"
        Name of the derived net value proxy column to create.
    net_value_proxy_with_opportunity_cost_column : str, default="net_value_proxy_with_opportunity_cost"
        Name of the derived proxy column including opportunity cost.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Display-ready model policy outcomes table with derived proxy value columns.

    Notes
    -----
    This function is used in the Validation notebook (Policy Simulation section)
    to separate display-oriented proxy calculations from notebook orchestration.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "build_model_policy_outcomes_display_table_started",
                "input_rows": df_model_policy_outcomes.shape[0],
                "input_columns": df_model_policy_outcomes.shape[1],
            },
        )

        required_columns = [
            accepted_non_default_loan_amount_column,
            accepted_default_loan_amount_column,
            rejected_non_default_loan_amount_column,
        ]

        missing_columns = [
            column_name
            for column_name in required_columns
            if column_name not in df_model_policy_outcomes.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns for model policy display table: {missing_columns}"
            )

        df_model_policy_outcomes_display = df_model_policy_outcomes.copy()

        df_model_policy_outcomes_display[net_value_proxy_column] = (
            df_model_policy_outcomes_display[accepted_non_default_loan_amount_column]
            - df_model_policy_outcomes_display[accepted_default_loan_amount_column]
        )

        df_model_policy_outcomes_display[net_value_proxy_with_opportunity_cost_column] = (
            df_model_policy_outcomes_display[net_value_proxy_column]
            - df_model_policy_outcomes_display[rejected_non_default_loan_amount_column]
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_model_policy_outcomes_display_table_completed",
                "rows": df_model_policy_outcomes_display.shape[0],
                "columns": df_model_policy_outcomes_display.shape[1],
                "derived_columns": [
                    net_value_proxy_column,
                    net_value_proxy_with_opportunity_cost_column,
                ],
            },
        )

        return df_model_policy_outcomes_display

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_model_policy_outcomes_display_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def format_model_policy_outcomes_display_table(
    df_model_policy_outcomes_display: pd.DataFrame,
    percentage_columns: list[str] | None = None,
    money_columns: list[str] | None = None,
    display_columns: list[str] | None = None,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Format a model policy outcomes display table for notebook presentation.

    This function standardizes the final presentation formatting of the model
    policy outcomes display table by applying percentage and currency formatting
    and restricting the output to a stable set of display columns. It ensures
    that the returned table is directly usable for notebook display without
    additional presentation logic inline.

    Parameters
    ----------
    df_model_policy_outcomes_display : pd.DataFrame
        Model policy outcomes display table containing derived proxy columns.
    percentage_columns : list[str] | None, default=None
        Columns to format as percentages. If None, project-default percentage
        columns are used.
    money_columns : list[str] | None, default=None
        Columns to format as whole-dollar currency values. If None,
        project-default money columns are used.
    display_columns : list[str] | None, default=None
        Columns to retain in the returned display table. If None,
        project-default display columns are used.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Formatted model policy outcomes display table ready for notebook display.

    Notes
    -----
    This function is used in the Validation notebook (Policy Simulation section)
    to separate presentation formatting from analytical orchestration.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "format_model_policy_outcomes_display_table_started",
                "input_rows": df_model_policy_outcomes_display.shape[0],
                "input_columns": df_model_policy_outcomes_display.shape[1],
            },
        )

        if percentage_columns is None:
            percentage_columns = [
                "acceptance_rate",
                "default_rate_among_accepted",
            ]

        if money_columns is None:
            money_columns = [
                "accepted_loan_amnt",
                "accepted_default_loan_amnt",
                "rejected_non_default_loan_amnt",
                "net_value_proxy",
                "net_value_proxy_with_opportunity_cost",
            ]

        if display_columns is None:
            display_columns = [
                "policy_name",
                "policy_value",
                "dataset_name",
                "accepted_count",
                "rejected_count",
                "acceptance_rate",
                "default_rate_among_accepted",
                "accepted_loan_amnt",
                "accepted_default_loan_amnt",
                "rejected_non_default_loan_amnt",
                "net_value_proxy",
                "net_value_proxy_with_opportunity_cost",
            ]

        required_columns = list(
            {
                *percentage_columns,
                *money_columns,
                *display_columns,
            }
        )

        missing_columns = [
            column_name
            for column_name in required_columns
            if column_name not in df_model_policy_outcomes_display.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns for formatted model policy display table: {missing_columns}"
            )

        df_model_policy_outcomes_formatted = df_model_policy_outcomes_display.copy()

        for column_name in percentage_columns:
            df_model_policy_outcomes_formatted[column_name] = (
                df_model_policy_outcomes_formatted[column_name]
                .map(lambda value: f"{value:.2%}")
            )

        for column_name in money_columns:
            df_model_policy_outcomes_formatted[column_name] = (
                df_model_policy_outcomes_formatted[column_name]
                .map(lambda value: f"${value:,.0f}")
            )

        df_model_policy_outcomes_formatted = (
            df_model_policy_outcomes_formatted[display_columns]
            .copy()
        )

        log_config.emit_log(
            log,
            {
                "stage": "format_model_policy_outcomes_display_table_completed",
                "rows": df_model_policy_outcomes_formatted.shape[0],
                "columns": df_model_policy_outcomes_formatted.shape[1],
            },
        )

        return df_model_policy_outcomes_formatted

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "format_model_policy_outcomes_display_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_baseline_policy_outcomes_artifact_table(
    baseline_policy_outcome_tables: list[pd.DataFrame],
    dataset_name_column: str = "dataset_name",
    policy_value_column: str = "policy_value",
    dataset_order: tuple[str, str] = ("train", "test"),
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a combined baseline policy outcomes artifact table across training and testing splits.

    This function standardizes the construction of the cross-split baseline policy
    outcomes artifact by concatenating evaluation outputs, enforcing consistent
    dataset ordering, and applying stable sorting. It ensures that the returned
    table is directly usable for validation display, comparison, and artifact
    export without additional preparation in the notebook.

    Parameters
    ----------
    baseline_policy_outcome_tables : list[pd.DataFrame]
        List of baseline policy outcome tables generated across cutoffs and splits.
    dataset_name_column : str, default="dataset_name"
        Name of the dataset split column.
    policy_value_column : str, default="policy_value"
        Name of the policy value column used for stable sorting.
    dataset_order : tuple[str, str], default=("train", "test")
        Ordered dataset labels used to enforce training → testing ordering.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Combined baseline policy outcomes artifact table with enforced dataset
        ordering and stable sorting by dataset and policy value.

    Notes
    -----
    This function is used in the Validation notebook (Policy Simulation section)
    to ensure consistent artifact construction across baseline subgrade policies.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "build_baseline_policy_outcomes_artifact_table_started",
                "table_count": len(baseline_policy_outcome_tables),
                "dataset_name_column": dataset_name_column,
                "policy_value_column": policy_value_column,
                "dataset_order": list(dataset_order),
            },
        )

        if not baseline_policy_outcome_tables:
            raise ValueError("baseline_policy_outcome_tables must contain at least one DataFrame.")

        if len(dataset_order) != 2:
            raise ValueError(
                "dataset_order must contain exactly two labels for training and testing."
            )

        df_baseline_policy_outcomes = pd.concat(
            baseline_policy_outcome_tables,
            axis=0,
            ignore_index=True,
        ).copy()

        if dataset_name_column not in df_baseline_policy_outcomes.columns:
            raise ValueError(
                f"Column '{dataset_name_column}' not found in baseline policy outcomes table."
            )

        if policy_value_column not in df_baseline_policy_outcomes.columns:
            raise ValueError(
                f"Column '{policy_value_column}' not found in baseline policy outcomes table."
            )

        df_baseline_policy_outcomes[dataset_name_column] = pd.Categorical(
            df_baseline_policy_outcomes[dataset_name_column],
            categories=list(dataset_order),
            ordered=True,
        )

        df_baseline_policy_outcomes = (
            df_baseline_policy_outcomes
            .sort_values([dataset_name_column, policy_value_column])
            .reset_index(drop=True)
            .copy()
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_baseline_policy_outcomes_artifact_table_completed",
                "rows": df_baseline_policy_outcomes.shape[0],
                "columns": df_baseline_policy_outcomes.shape[1],
            },
        )

        return df_baseline_policy_outcomes

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_baseline_policy_outcomes_artifact_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_baseline_policy_outcomes_display_table(
    df_baseline_policy_outcomes: pd.DataFrame,
    accepted_non_default_loan_amount_column: str = "accepted_non_default_loan_amnt",
    accepted_default_loan_amount_column: str = "accepted_default_loan_amnt",
    rejected_non_default_loan_amount_column: str = "rejected_non_default_loan_amnt",
    net_value_proxy_column: str = "net_value_proxy",
    net_value_proxy_with_opportunity_cost_column: str = "net_value_proxy_with_opportunity_cost",
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a display-ready baseline policy outcomes table with derived proxy value columns.

    This function standardizes the enrichment of the baseline policy outcomes
    artifact by adding derived proxy value measures used for validation display
    and economic interpretation. It ensures that the returned table is ready for
    final formatting without additional calculation in the notebook.

    Parameters
    ----------
    df_baseline_policy_outcomes : pd.DataFrame
        Combined baseline policy outcomes artifact table.
    accepted_non_default_loan_amount_column : str, default="accepted_non_default_loan_amnt"
        Name of the accepted non-default loan amount column.
    accepted_default_loan_amount_column : str, default="accepted_default_loan_amnt"
        Name of the accepted default loan amount column.
    rejected_non_default_loan_amount_column : str, default="rejected_non_default_loan_amnt"
        Name of the rejected non-default loan amount column.
    net_value_proxy_column : str, default="net_value_proxy"
        Name of the derived net value proxy column to create.
    net_value_proxy_with_opportunity_cost_column : str, default="net_value_proxy_with_opportunity_cost"
        Name of the derived proxy column including opportunity cost.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Display-ready baseline policy outcomes table with derived proxy value columns.

    Notes
    -----
    This function is used in the Validation notebook (Policy Simulation section)
    to separate display-oriented proxy calculations from notebook orchestration.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "build_baseline_policy_outcomes_display_table_started",
                "input_rows": df_baseline_policy_outcomes.shape[0],
                "input_columns": df_baseline_policy_outcomes.shape[1],
            },
        )

        required_columns = [
            accepted_non_default_loan_amount_column,
            accepted_default_loan_amount_column,
            rejected_non_default_loan_amount_column,
        ]

        missing_columns = [
            column_name
            for column_name in required_columns
            if column_name not in df_baseline_policy_outcomes.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns for baseline policy display table: {missing_columns}"
            )

        df_baseline_policy_outcomes_display = df_baseline_policy_outcomes.copy()

        df_baseline_policy_outcomes_display[net_value_proxy_column] = (
            df_baseline_policy_outcomes_display[accepted_non_default_loan_amount_column]
            - df_baseline_policy_outcomes_display[accepted_default_loan_amount_column]
        )

        df_baseline_policy_outcomes_display[net_value_proxy_with_opportunity_cost_column] = (
            df_baseline_policy_outcomes_display[net_value_proxy_column]
            - df_baseline_policy_outcomes_display[rejected_non_default_loan_amount_column]
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_baseline_policy_outcomes_display_table_completed",
                "rows": df_baseline_policy_outcomes_display.shape[0],
                "columns": df_baseline_policy_outcomes_display.shape[1],
                "derived_columns": [
                    net_value_proxy_column,
                    net_value_proxy_with_opportunity_cost_column,
                ],
            },
        )

        return df_baseline_policy_outcomes_display

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_baseline_policy_outcomes_display_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def format_baseline_policy_outcomes_display_table(
    df_baseline_policy_outcomes_display: pd.DataFrame,
    percentage_columns: list[str] | None = None,
    money_columns: list[str] | None = None,
    display_columns: list[str] | None = None,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Format a baseline policy outcomes display table for notebook presentation.

    This function standardizes the final presentation formatting of the baseline
    policy outcomes display table by applying percentage and currency formatting
    and restricting the output to a stable set of display columns. It ensures
    that the returned table is directly usable for notebook display without
    additional presentation logic inline.

    Parameters
    ----------
    df_baseline_policy_outcomes_display : pd.DataFrame
        Baseline policy outcomes display table containing derived proxy columns.
    percentage_columns : list[str] | None, default=None
        Columns to format as percentages. If None, project-default percentage
        columns are used.
    money_columns : list[str] | None, default=None
        Columns to format as whole-dollar currency values. If None,
        project-default money columns are used.
    display_columns : list[str] | None, default=None
        Columns to retain in the returned display table. If None,
        project-default display columns are used.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Formatted baseline policy outcomes display table ready for notebook display.

    Notes
    -----
    This function is used in the Validation notebook (Policy Simulation section)
    to separate presentation formatting from analytical orchestration.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "format_baseline_policy_outcomes_display_table_started",
                "input_rows": df_baseline_policy_outcomes_display.shape[0],
                "input_columns": df_baseline_policy_outcomes_display.shape[1],
            },
        )

        if percentage_columns is None:
            percentage_columns = [
                "acceptance_rate",
                "default_rate_among_accepted",
            ]

        if money_columns is None:
            money_columns = [
                "accepted_loan_amnt",
                "accepted_default_loan_amnt",
                "rejected_non_default_loan_amnt",
                "net_value_proxy",
                "net_value_proxy_with_opportunity_cost",
            ]

        if display_columns is None:
            display_columns = [
                "policy_name",
                "policy_value",
                "dataset_name",
                "accepted_count",
                "rejected_count",
                "acceptance_rate",
                "default_rate_among_accepted",
                "accepted_loan_amnt",
                "accepted_default_loan_amnt",
                "rejected_non_default_loan_amnt",
                "net_value_proxy",
                "net_value_proxy_with_opportunity_cost",
            ]

        required_columns = list({*percentage_columns, *money_columns, *display_columns})

        missing_columns = [
            column_name
            for column_name in required_columns
            if column_name not in df_baseline_policy_outcomes_display.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns for formatted baseline policy display table: {missing_columns}"
            )

        df_baseline_policy_outcomes_formatted = df_baseline_policy_outcomes_display.copy()

        for column_name in percentage_columns:
            df_baseline_policy_outcomes_formatted[column_name] = (
                df_baseline_policy_outcomes_formatted[column_name]
                .map(lambda value: f"{value:.2%}")
            )

        for column_name in money_columns:
            df_baseline_policy_outcomes_formatted[column_name] = (
                df_baseline_policy_outcomes_formatted[column_name]
                .map(lambda value: f"${value:,.0f}")
            )

        df_baseline_policy_outcomes_formatted = (
            df_baseline_policy_outcomes_formatted[display_columns]
            .copy()
        )

        log_config.emit_log(
            log,
            {
                "stage": "format_baseline_policy_outcomes_display_table_completed",
                "rows": df_baseline_policy_outcomes_formatted.shape[0],
                "columns": df_baseline_policy_outcomes_formatted.shape[1],
            },
        )

        return df_baseline_policy_outcomes_formatted

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "format_baseline_policy_outcomes_display_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_matched_policy_comparison_table(
    df_model_policy_outcomes: pd.DataFrame,
    df_baseline_policy_outcomes: pd.DataFrame,
    dataset_name: str = "test",
    dataset_name_column: str = "dataset_name",
    policy_value_column: str = "policy_value",
    acceptance_rate_column: str = "acceptance_rate",
    default_rate_column: str = "default_rate_among_accepted",
    accepted_non_default_loan_amount_column: str = "accepted_non_default_loan_amnt",
    accepted_default_loan_amount_column: str = "accepted_default_loan_amnt",
    rejected_non_default_loan_amount_column: str = "rejected_non_default_loan_amnt",
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a matched policy comparison table by aligning baseline and model policies
    on nearest acceptance rate within a single dataset split.

    This function standardizes the comparison between baseline and model policy
    outcomes by matching each baseline policy to the closest model policy in
    terms of acceptance rate. It also computes proxy value measures and
    difference columns required for validation comparison and reporting.

    Parameters
    ----------
    df_model_policy_outcomes : pd.DataFrame
        Model policy outcomes artifact table.
    df_baseline_policy_outcomes : pd.DataFrame
        Baseline policy outcomes artifact table.
    dataset_name : str, default="test"
        Dataset split to compare.
    dataset_name_column : str, default="dataset_name"
        Name of the dataset split column.
    policy_value_column : str, default="policy_value"
        Name of the policy value column.
    acceptance_rate_column : str, default="acceptance_rate"
        Name of the acceptance rate column.
    default_rate_column : str, default="default_rate_among_accepted"
        Name of the accepted-loan default rate column.
    accepted_non_default_loan_amount_column : str, default="accepted_non_default_loan_amnt"
        Name of the accepted non-default loan amount column.
    accepted_default_loan_amount_column : str, default="accepted_default_loan_amnt"
        Name of the accepted default loan amount column.
    rejected_non_default_loan_amount_column : str, default="rejected_non_default_loan_amnt"
        Name of the rejected non-default loan amount column.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Matched policy comparison table sorted by baseline acceptance rate.

    Notes
    -----
    This function is used in the Validation notebook (Policy Simulation section)
    to compare baseline and model policies at similar acceptance levels.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "build_matched_policy_comparison_table_started",
                "dataset_name": dataset_name,
                "model_rows": df_model_policy_outcomes.shape[0],
                "baseline_rows": df_baseline_policy_outcomes.shape[0],
            },
        )

        required_columns = [
            dataset_name_column,
            policy_value_column,
            acceptance_rate_column,
            default_rate_column,
            accepted_non_default_loan_amount_column,
            accepted_default_loan_amount_column,
            rejected_non_default_loan_amount_column,
        ]

        for column_name in required_columns:
            if column_name not in df_model_policy_outcomes.columns:
                raise ValueError(
                    f"Column '{column_name}' not found in model policy outcomes table."
                )
            if column_name not in df_baseline_policy_outcomes.columns:
                raise ValueError(
                    f"Column '{column_name}' not found in baseline policy outcomes table."
                )

        df_model_split = df_model_policy_outcomes.loc[
            df_model_policy_outcomes[dataset_name_column] == dataset_name
        ].copy()

        df_baseline_split = df_baseline_policy_outcomes.loc[
            df_baseline_policy_outcomes[dataset_name_column] == dataset_name
        ].copy()

        if df_model_split.empty:
            raise ValueError(
                f"No model policy rows found for dataset_name='{dataset_name}'."
            )

        if df_baseline_split.empty:
            raise ValueError(
                f"No baseline policy rows found for dataset_name='{dataset_name}'."
            )

        comparison_rows: list[dict[str, float | str]] = []

        for _, baseline_row in df_baseline_split.iterrows():
            baseline_acceptance_rate = baseline_row[acceptance_rate_column]

            df_model_local = df_model_split.copy()
            df_model_local["acceptance_diff"] = (
                df_model_local[acceptance_rate_column] - baseline_acceptance_rate
            ).abs()

            closest_model_row = df_model_local.loc[
                df_model_local["acceptance_diff"].idxmin()
            ]

            baseline_net_value_proxy = (
                baseline_row[accepted_non_default_loan_amount_column]
                - baseline_row[accepted_default_loan_amount_column]
            )

            baseline_net_value_with_opportunity_cost = (
                baseline_net_value_proxy
                - baseline_row[rejected_non_default_loan_amount_column]
            )

            model_net_value_proxy = (
                closest_model_row[accepted_non_default_loan_amount_column]
                - closest_model_row[accepted_default_loan_amount_column]
            )

            model_net_value_with_opportunity_cost = (
                model_net_value_proxy
                - closest_model_row[rejected_non_default_loan_amount_column]
            )

            comparison_rows.append(
                {
                    "baseline_policy": baseline_row[policy_value_column],
                    "baseline_acceptance_rate": baseline_row[acceptance_rate_column],
                    "baseline_default_rate": baseline_row[default_rate_column],
                    "baseline_net_value_proxy": baseline_net_value_proxy,
                    "baseline_net_value_with_opportunity_cost": (
                        baseline_net_value_with_opportunity_cost
                    ),
                    "model_threshold": closest_model_row[policy_value_column],
                    "model_acceptance_rate": closest_model_row[acceptance_rate_column],
                    "model_default_rate": closest_model_row[default_rate_column],
                    "model_net_value_proxy": model_net_value_proxy,
                    "model_net_value_with_opportunity_cost": (
                        model_net_value_with_opportunity_cost
                    ),
                    "acceptance_rate_diff": (
                        closest_model_row[acceptance_rate_column]
                        - baseline_acceptance_rate
                    ),
                    "default_rate_diff": (
                        closest_model_row[default_rate_column]
                        - baseline_row[default_rate_column]
                    ),
                    "net_value_proxy_diff": (
                        model_net_value_proxy - baseline_net_value_proxy
                    ),
                    "net_value_with_opportunity_cost_diff": (
                        model_net_value_with_opportunity_cost
                        - baseline_net_value_with_opportunity_cost
                    ),
                }
            )

        df_policy_comparison = pd.DataFrame(comparison_rows).copy()

        df_policy_comparison = (
            df_policy_comparison
            .sort_values("baseline_acceptance_rate")
            .reset_index(drop=True)
            .copy()
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_matched_policy_comparison_table_completed",
                "rows": df_policy_comparison.shape[0],
                "columns": df_policy_comparison.shape[1],
            },
        )

        return df_policy_comparison

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_matched_policy_comparison_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def format_matched_policy_comparison_table(
    df_policy_comparison: pd.DataFrame,
    percentage_columns: list[str] | None = None,
    money_columns: list[str] | None = None,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Format a matched policy comparison table for notebook presentation.

    This function standardizes the final presentation formatting of the matched
    baseline-versus-model comparison table by applying percentage and currency
    formatting. Negative currency values are rendered with a leading minus sign.

    Parameters
    ----------
    df_policy_comparison : pd.DataFrame
        Matched policy comparison table.
    percentage_columns : list[str] | None, default=None
        Columns to format as percentages. If None, project-default percentage
        columns are used.
    money_columns : list[str] | None, default=None
        Columns to format as whole-dollar currency values. If None,
        project-default money columns are used.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Formatted matched policy comparison table ready for notebook display.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "format_matched_policy_comparison_table_started",
                "input_rows": df_policy_comparison.shape[0],
                "input_columns": df_policy_comparison.shape[1],
            },
        )

        if percentage_columns is None:
            percentage_columns = [
                "baseline_acceptance_rate",
                "model_acceptance_rate",
                "acceptance_rate_diff",
                "baseline_default_rate",
                "model_default_rate",
                "default_rate_diff",
            ]

        if money_columns is None:
            money_columns = [
                "baseline_net_value_proxy",
                "baseline_net_value_with_opportunity_cost",
                "model_net_value_proxy",
                "model_net_value_with_opportunity_cost",
                "net_value_proxy_diff",
                "net_value_with_opportunity_cost_diff",
            ]

        required_columns = list({*percentage_columns, *money_columns})

        missing_columns = [
            column_name
            for column_name in required_columns
            if column_name not in df_policy_comparison.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns for formatted policy comparison table: {missing_columns}"
            )

        df_policy_comparison_formatted = df_policy_comparison.copy()

        for column_name in percentage_columns:
            df_policy_comparison_formatted[column_name] = (
                df_policy_comparison_formatted[column_name]
                .map(lambda value: f"{value:.2%}")
            )

        for column_name in money_columns:
            df_policy_comparison_formatted[column_name] = (
                df_policy_comparison_formatted[column_name]
                .map(
                    lambda value: (
                        f"${value:,.0f}" if value >= 0 else f"-${abs(value):,.0f}"
                    )
                )
            )

        log_config.emit_log(
            log,
            {
                "stage": "format_matched_policy_comparison_table_completed",
                "rows": df_policy_comparison_formatted.shape[0],
                "columns": df_policy_comparison_formatted.shape[1],
            },
        )

        return df_policy_comparison_formatted

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "format_matched_policy_comparison_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_policy_exception_pricing_summary_table(
    df_validation_train: pd.DataFrame,
    df_validation_test: pd.DataFrame,
    target_column: str,
    loan_status_column: str = "loan_status",
    predicted_probability_column: str = "predicted_default_probability",
    interest_rate_column: str = "int_rate",
    loan_amount_column: str = "loan_amnt",
    dataset_name_column: str = "dataset_name",
    policy_exception_flag_column: str = "policy_exception_flag",
    policy_group_column: str = "policy_group",
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a policy-exception and pricing summary table across training and testing datasets.

    This function standardizes the construction of the policy-exception pricing
    audit summary by validating required columns, combining train/test datasets,
    identifying policy-exception loans, aggregating summary metrics, and
    enforcing stable ordering for artifact export and display.

    Parameters
    ----------
    df_validation_train : pd.DataFrame
        Validation dataset for the training split.
    df_validation_test : pd.DataFrame
        Validation dataset for the testing split.
    target_column : str
        Name of the default target column.
    loan_status_column : str, default="loan_status"
        Name of the loan status column.
    predicted_probability_column : str, default="predicted_default_probability"
        Name of the predicted default probability column.
    interest_rate_column : str, default="int_rate"
        Name of the interest rate column.
    loan_amount_column : str, default="loan_amnt"
        Name of the loan amount column.
    dataset_name_column : str, default="dataset_name"
        Name of the dataset split column to create.
    policy_exception_flag_column : str, default="policy_exception_flag"
        Name of the policy-exception boolean flag column to create.
    policy_group_column : str, default="policy_group"
        Name of the policy group column to create.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Summary table of standard-issued versus policy-exception-issued loans
        across training and testing datasets.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "build_policy_exception_pricing_summary_table_started",
                "train_rows": df_validation_train.shape[0],
                "train_columns": df_validation_train.shape[1],
                "test_rows": df_validation_test.shape[0],
                "test_columns": df_validation_test.shape[1],
                "target_column": target_column,
            },
        )

        required_columns = [
            loan_status_column,
            target_column,
            predicted_probability_column,
            interest_rate_column,
            loan_amount_column,
        ]

        missing_columns_train = [
            column_name
            for column_name in required_columns
            if column_name not in df_validation_train.columns
        ]
        missing_columns_test = [
            column_name
            for column_name in required_columns
            if column_name not in df_validation_test.columns
        ]

        if missing_columns_train:
            raise ValueError(
                f"Missing required training columns for policy-exception pricing audit: {missing_columns_train}"
            )

        if missing_columns_test:
            raise ValueError(
                f"Missing required testing columns for policy-exception pricing audit: {missing_columns_test}"
            )

        df_policy_exception_audit_train = df_validation_train.copy()
        df_policy_exception_audit_test = df_validation_test.copy()

        df_policy_exception_audit_train[dataset_name_column] = "train"
        df_policy_exception_audit_test[dataset_name_column] = "test"

        df_policy_exception_audit = pd.concat(
            [df_policy_exception_audit_train, df_policy_exception_audit_test],
            axis=0,
            ignore_index=True,
        ).copy()

        df_policy_exception_audit[policy_exception_flag_column] = (
            df_policy_exception_audit[loan_status_column]
            .astype(str)
            .str.contains("does_not_meet_the_credit_policy", case=False, na=False)
        )

        df_policy_exception_audit[policy_group_column] = (
            df_policy_exception_audit[policy_exception_flag_column]
            .map(
                {
                    False: "standard_issued",
                    True: "policy_exception_issued",
                }
            )
        )

        df_policy_exception_summary = (
            df_policy_exception_audit
            .groupby([dataset_name_column, policy_group_column], observed=True)
            .agg(
                row_count=(loan_status_column, "size"),
                observed_default_rate=(target_column, "mean"),
                predicted_default_probability_mean=(predicted_probability_column, "mean"),
                int_rate_mean=(interest_rate_column, "mean"),
                int_rate_median=(interest_rate_column, "median"),
                loan_amnt_mean=(loan_amount_column, "mean"),
                loan_amnt_median=(loan_amount_column, "median"),
            )
            .reset_index()
            .copy()
        )

        df_policy_exception_population = (
            df_policy_exception_summary
            .groupby(dataset_name_column, observed=True)["row_count"]
            .transform("sum")
        )

        df_policy_exception_summary["population_share"] = (
            df_policy_exception_summary["row_count"] / df_policy_exception_population
        )

        df_policy_exception_summary[dataset_name_column] = pd.Categorical(
            df_policy_exception_summary[dataset_name_column],
            categories=["train", "test"],
            ordered=True,
        )

        df_policy_exception_summary[policy_group_column] = pd.Categorical(
            df_policy_exception_summary[policy_group_column],
            categories=["standard_issued", "policy_exception_issued"],
            ordered=True,
        )

        df_policy_exception_summary = (
            df_policy_exception_summary
            .sort_values([dataset_name_column, policy_group_column])
            .reset_index(drop=True)
            .copy()
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_policy_exception_pricing_summary_table_completed",
                "rows": df_policy_exception_summary.shape[0],
                "columns": df_policy_exception_summary.shape[1],
            },
        )

        return df_policy_exception_summary

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_policy_exception_pricing_summary_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_policy_exception_pricing_comparison_table(
    df_policy_exception_summary: pd.DataFrame,
    dataset_name_column: str = "dataset_name",
    policy_group_column: str = "policy_group",
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a policy-exception versus standard-issued comparison table from the summary artifact.

    This function standardizes the construction of a comparison table by pivoting
    the summary artifact to wide format and computing gap metrics between
    policy-exception-issued and standard-issued loans.

    Parameters
    ----------
    df_policy_exception_summary : pd.DataFrame
        Policy-exception pricing summary table.
    dataset_name_column : str, default="dataset_name"
        Name of the dataset split column.
    policy_group_column : str, default="policy_group"
        Name of the policy group column.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Wide comparison table with gap columns across train/test splits.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "build_policy_exception_pricing_comparison_table_started",
                "input_rows": df_policy_exception_summary.shape[0],
                "input_columns": df_policy_exception_summary.shape[1],
            },
        )

        required_columns = [dataset_name_column, policy_group_column]
        missing_columns = [
            column_name
            for column_name in required_columns
            if column_name not in df_policy_exception_summary.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns for policy-exception comparison table: {missing_columns}"
            )

        df_policy_exception_comparison = (
            df_policy_exception_summary
            .pivot(
                index=dataset_name_column,
                columns=policy_group_column,
                values=[
                    "row_count",
                    "population_share",
                    "observed_default_rate",
                    "predicted_default_probability_mean",
                    "int_rate_mean",
                    "int_rate_median",
                    "loan_amnt_mean",
                    "loan_amnt_median",
                ],
            )
            .copy()
        )

        df_policy_exception_comparison.columns = [
            f"{metric_name}_{policy_group}"
            for metric_name, policy_group in df_policy_exception_comparison.columns
        ]

        df_policy_exception_comparison = (
            df_policy_exception_comparison
            .reset_index()
            .copy()
        )

        df_policy_exception_comparison["default_rate_gap"] = (
            df_policy_exception_comparison["observed_default_rate_policy_exception_issued"]
            - df_policy_exception_comparison["observed_default_rate_standard_issued"]
        )

        df_policy_exception_comparison["predicted_risk_gap"] = (
            df_policy_exception_comparison["predicted_default_probability_mean_policy_exception_issued"]
            - df_policy_exception_comparison["predicted_default_probability_mean_standard_issued"]
        )

        df_policy_exception_comparison["int_rate_mean_gap"] = (
            df_policy_exception_comparison["int_rate_mean_policy_exception_issued"]
            - df_policy_exception_comparison["int_rate_mean_standard_issued"]
        )

        df_policy_exception_comparison["int_rate_median_gap"] = (
            df_policy_exception_comparison["int_rate_median_policy_exception_issued"]
            - df_policy_exception_comparison["int_rate_median_standard_issued"]
        )

        df_policy_exception_comparison[dataset_name_column] = pd.Categorical(
            df_policy_exception_comparison[dataset_name_column],
            categories=["train", "test"],
            ordered=True,
        )

        df_policy_exception_comparison = (
            df_policy_exception_comparison
            .sort_values([dataset_name_column])
            .reset_index(drop=True)
            .copy()
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_policy_exception_pricing_comparison_table_completed",
                "rows": df_policy_exception_comparison.shape[0],
                "columns": df_policy_exception_comparison.shape[1],
            },
        )

        return df_policy_exception_comparison

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_policy_exception_pricing_comparison_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def format_policy_exception_pricing_summary_table(
    df_policy_exception_summary: pd.DataFrame,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Format the policy-exception pricing summary table for notebook display.

    This function standardizes percentage, interest-rate, and monetary formatting
    for the summary table and returns a stable subset of display columns.

    Parameters
    ----------
    df_policy_exception_summary : pd.DataFrame
        Policy-exception pricing summary artifact table.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Formatted summary table ready for notebook display.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "format_policy_exception_pricing_summary_table_started",
                "input_rows": df_policy_exception_summary.shape[0],
                "input_columns": df_policy_exception_summary.shape[1],
            },
        )

        df_policy_exception_summary_display = df_policy_exception_summary.copy()

        for column_name in [
            "population_share",
            "observed_default_rate",
            "predicted_default_probability_mean",
        ]:
            df_policy_exception_summary_display[column_name] = (
                df_policy_exception_summary_display[column_name]
                .map(lambda value: f"{value:.2%}")
            )

        for column_name in ["int_rate_mean", "int_rate_median"]:
            df_policy_exception_summary_display[column_name] = (
                df_policy_exception_summary_display[column_name]
                .map(lambda value: f"{value:.2f}%")
            )

        for column_name in ["loan_amnt_mean", "loan_amnt_median"]:
            df_policy_exception_summary_display[column_name] = (
                df_policy_exception_summary_display[column_name]
                .map(lambda value: f"${value:,.0f}")
            )

        df_policy_exception_summary_display = df_policy_exception_summary_display[
            [
                "dataset_name",
                "policy_group",
                "row_count",
                "population_share",
                "observed_default_rate",
                "predicted_default_probability_mean",
                "int_rate_mean",
                "int_rate_median",
                "loan_amnt_mean",
                "loan_amnt_median",
            ]
        ].copy()

        log_config.emit_log(
            log,
            {
                "stage": "format_policy_exception_pricing_summary_table_completed",
                "rows": df_policy_exception_summary_display.shape[0],
                "columns": df_policy_exception_summary_display.shape[1],
            },
        )

        return df_policy_exception_summary_display

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "format_policy_exception_pricing_summary_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def format_policy_exception_pricing_comparison_table(
    df_policy_exception_comparison: pd.DataFrame,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Format the policy-exception pricing comparison table for notebook display.

    This function standardizes percentage, interest-rate, and monetary formatting
    for the comparison table and returns a stable subset of display columns.

    Parameters
    ----------
    df_policy_exception_comparison : pd.DataFrame
        Policy-exception pricing comparison artifact table.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by emit_log.

    Returns
    -------
    pd.DataFrame
        Formatted comparison table ready for notebook display.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "format_policy_exception_pricing_comparison_table_started",
                "input_rows": df_policy_exception_comparison.shape[0],
                "input_columns": df_policy_exception_comparison.shape[1],
            },
        )

        df_policy_exception_comparison_display = df_policy_exception_comparison.copy()

        for column_name in [
            "population_share_standard_issued",
            "population_share_policy_exception_issued",
            "observed_default_rate_standard_issued",
            "observed_default_rate_policy_exception_issued",
            "predicted_default_probability_mean_standard_issued",
            "predicted_default_probability_mean_policy_exception_issued",
            "default_rate_gap",
            "predicted_risk_gap",
        ]:
            df_policy_exception_comparison_display[column_name] = (
                df_policy_exception_comparison_display[column_name]
                .map(lambda value: f"{value:.2%}")
            )

        for column_name in [
            "int_rate_mean_standard_issued",
            "int_rate_mean_policy_exception_issued",
            "int_rate_median_standard_issued",
            "int_rate_median_policy_exception_issued",
            "int_rate_mean_gap",
            "int_rate_median_gap",
        ]:
            df_policy_exception_comparison_display[column_name] = (
                df_policy_exception_comparison_display[column_name]
                .map(lambda value: f"{value:.2f}%")
            )

        for column_name in [
            "loan_amnt_mean_standard_issued",
            "loan_amnt_mean_policy_exception_issued",
            "loan_amnt_median_standard_issued",
            "loan_amnt_median_policy_exception_issued",
        ]:
            df_policy_exception_comparison_display[column_name] = (
                df_policy_exception_comparison_display[column_name]
                .map(lambda value: f"${value:,.0f}")
            )

        df_policy_exception_comparison_display = df_policy_exception_comparison_display[
            [
                "dataset_name",
                "row_count_standard_issued",
                "row_count_policy_exception_issued",
                "population_share_standard_issued",
                "population_share_policy_exception_issued",
                "observed_default_rate_standard_issued",
                "observed_default_rate_policy_exception_issued",
                "default_rate_gap",
                "predicted_default_probability_mean_standard_issued",
                "predicted_default_probability_mean_policy_exception_issued",
                "predicted_risk_gap",
                "int_rate_mean_standard_issued",
                "int_rate_mean_policy_exception_issued",
                "int_rate_mean_gap",
            ]
        ].copy()

        log_config.emit_log(
            log,
            {
                "stage": "format_policy_exception_pricing_comparison_table_completed",
                "rows": df_policy_exception_comparison_display.shape[0],
                "columns": df_policy_exception_comparison_display.shape[1],
            },
        )

        return df_policy_exception_comparison_display

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "format_policy_exception_pricing_comparison_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise
    

def build_risk_band_summary_table(
    df_validation: pd.DataFrame,
    score_column: str,
    target_column: str,
    dataset_name: str,
    n_bands: int = 10,
    interest_rate_column: str | None = None,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a risk band summary table for validation.

    This function groups loans into quantile-based risk bands using predicted
    default probability and computes observed default rates per band. If an
    interest rate column is provided, the summary is extended with average and
    median interest rates per band.

    Interest rates are returned on a 0–1 scale when the source column appears
    to be stored in percentage-point form (for example 13.78 rather than 0.1378).

    Parameters
    ----------
    df_validation : pd.DataFrame
        Validation dataset containing predictions and target.
    score_column : str
        Column containing predicted default probabilities.
    target_column : str
        Binary target column (1 = default, 0 = non-default).
    dataset_name : str
        Dataset identifier (e.g. "train", "test").
    n_bands : int, default=10
        Number of quantile bands to create.
    interest_rate_column : str | None, default=None
        Interest rate column to summarize per band. If None, no pricing
        summaries are added.
    log : Callable | Path | str | None
        Logger compatible with emit_log.

    Returns
    -------
    pd.DataFrame
        Risk band summary table.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "build_risk_band_summary_table_started",
                "rows": df_validation.shape[0],
                "score_column": score_column,
                "target_column": target_column,
                "dataset_name": dataset_name,
                "n_bands": n_bands,
                "interest_rate_column": interest_rate_column,
            },
        )

        df = df_validation.copy()

        if score_column not in df.columns:
            raise KeyError(f"{score_column} not found in dataframe")

        if target_column not in df.columns:
            raise KeyError(f"{target_column} not found in dataframe")

        if interest_rate_column is not None and interest_rate_column not in df.columns:
            raise KeyError(f"{interest_rate_column} not found in dataframe")

        if interest_rate_column is not None:
            non_null_interest_rates = df[interest_rate_column].dropna().copy()

            if non_null_interest_rates.empty:
                raise ValueError(
                    f"{interest_rate_column} contains only missing values"
                )

            interest_rate_max = float(non_null_interest_rates.max())
            interest_rate_scale_applied = "none"

            if interest_rate_max > 1.0:
                df[interest_rate_column] = df[interest_rate_column] / 100.0
                interest_rate_scale_applied = "divided_by_100"

            log_config.emit_log(
                log,
                {
                    "stage": "risk_band_interest_rate_scale_checked",
                    "dataset_name": dataset_name,
                    "interest_rate_column": interest_rate_column,
                    "source_max_value": interest_rate_max,
                    "scale_applied": interest_rate_scale_applied,
                },
            )

        df["risk_band"] = pd.qcut(
            df[score_column],
            q=n_bands,
            duplicates="drop",
        )

        aggregation_mapping: dict[str, tuple[str, str]] = {
            "loan_count": (target_column, "size"),
            "predicted_default_probability_mean": (score_column, "mean"),
            "observed_default_rate": (target_column, "mean"),
        }

        if interest_rate_column is not None:
            aggregation_mapping["int_rate_mean"] = (interest_rate_column, "mean")
            aggregation_mapping["int_rate_median"] = (interest_rate_column, "median")

        df_summary = (
            df.groupby("risk_band", observed=True)
            .agg(**aggregation_mapping)
            .reset_index()
            .copy()
        )

        df_summary["dataset_name"] = dataset_name

        df_summary = (
            df_summary
            .sort_values("predicted_default_probability_mean")
            .reset_index(drop=True)
            .copy()
        )

        log_config.emit_log(
            log,
            {
                "stage": "build_risk_band_summary_table_completed",
                "rows": df_summary.shape[0],
                "columns": df_summary.shape[1],
            },
        )

        return df_summary

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_risk_band_summary_table_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise