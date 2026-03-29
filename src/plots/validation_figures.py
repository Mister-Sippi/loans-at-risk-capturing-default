from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd

import config.logging as log_config


def plot_normalized_subgrade_distribution_by_grade(
    df_subgrade_distribution_normalized: pd.DataFrame,
    split_column: str,
    grade_column: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> plt.Figure:
    """
    Plot the normalized subgrade distribution within each grade for training and
    testing data.

    The function expects a wide-format table with one row per split-grade
    combination and one column per subgrade (e.g., a1, a2, ..., g5).

    Parameters
    ----------
    df_subgrade_distribution_normalized : pd.DataFrame
        Wide-format normalized subgrade distribution table.
    split_column : str
        Name of the split column (e.g., train/test).
    grade_column : str
        Name of the grade column.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by ``log_config.emit_log``.

    Returns
    -------
    plt.Figure
        Matplotlib figure containing one subplot per grade.

    Raises
    ------
    ValueError
        If required columns are missing, expected grades are absent, or
        required split-grade rows are missing.
    """
    try:
        log_config.emit_log(log, "Building normalized subgrade distribution figure")

        required_columns = {split_column, grade_column}
        missing_columns = required_columns - set(df_subgrade_distribution_normalized.columns)

        if missing_columns:
            raise ValueError(
                {
                    "stage": "plot_normalized_subgrade_distribution_by_grade",
                    "error": "missing_required_columns",
                    "missing_columns": sorted(missing_columns),
                }
            )

        df_normalized_local = df_subgrade_distribution_normalized.copy()

        expected_grades = ["a", "b", "c", "d", "e", "f", "g"]
        available_grades = sorted(df_normalized_local[grade_column].dropna().unique().tolist())

        missing_grades = sorted(set(expected_grades) - set(available_grades))
        if missing_grades:
            raise ValueError(
                {
                    "stage": "plot_normalized_subgrade_distribution_by_grade",
                    "error": "missing_expected_grades",
                    "missing_grades": missing_grades,
                }
            )

        split_values = df_normalized_local[split_column].dropna().unique().tolist()
        expected_splits = ["train", "test"]
        missing_splits = sorted(set(expected_splits) - set(split_values))
        if missing_splits:
            raise ValueError(
                {
                    "stage": "plot_normalized_subgrade_distribution_by_grade",
                    "error": "missing_expected_splits",
                    "missing_splits": missing_splits,
                }
            )

        subgrade_columns = [
            column
            for column in df_normalized_local.columns
            if column not in {split_column, grade_column}
        ]

        figure, axes = plt.subplots(nrows=7, ncols=1, figsize=(12, 20), sharex=False)

        if not isinstance(axes, (list, tuple)):
            axes = list(axes)

        for index, grade_value in enumerate(expected_grades):
            axis = axes[index]

            train_row = df_normalized_local[
                (df_normalized_local[split_column] == "train")
                & (df_normalized_local[grade_column] == grade_value)
            ]
            test_row = df_normalized_local[
                (df_normalized_local[split_column] == "test")
                & (df_normalized_local[grade_column] == grade_value)
            ]

            if train_row.shape[0] != 1 or test_row.shape[0] != 1:
                raise ValueError(
                    {
                        "stage": "plot_normalized_subgrade_distribution_by_grade",
                        "error": "unexpected_split_grade_row_count",
                        "grade": grade_value,
                        "train_rows": train_row.shape[0],
                        "test_rows": test_row.shape[0],
                    }
                )

            grade_subgrade_columns = [
                column
                for column in subgrade_columns
                if column.startswith(grade_value)
            ]

            if len(grade_subgrade_columns) != 5:
                raise ValueError(
                    {
                        "stage": "plot_normalized_subgrade_distribution_by_grade",
                        "error": "unexpected_subgrade_column_count",
                        "grade": grade_value,
                        "subgrade_columns": grade_subgrade_columns,
                    }
                )

            train_values = train_row.iloc[0][grade_subgrade_columns].tolist()
            test_values = test_row.iloc[0][grade_subgrade_columns].tolist()

            axis.plot(grade_subgrade_columns, train_values, marker="o", label="train")
            axis.plot(grade_subgrade_columns, test_values, marker="o", label="test")

            axis.set_title(f"Grade {grade_value.upper()}")
            axis.set_ylabel("Proportion")
            axis.set_ylim(0, max(train_values + test_values) * 1.15 if max(train_values + test_values) > 0 else 1)
            axis.grid(True, alpha=0.3)
            axis.legend()

        axes[-1].set_xlabel("Subgrade")

        figure.suptitle("Normalized Subgrade Distribution by Grade", fontsize=14)
        figure.tight_layout(rect=[0, 0, 1, 0.98])

        log_config.emit_log(
            log,
            {
                "stage": "plot_normalized_subgrade_distribution_by_grade_complete",
                "grades": expected_grades,
                "rows": df_normalized_local.shape[0],
                "columns": df_normalized_local.shape[1],
            },
        )

        return figure

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "plot_normalized_subgrade_distribution_by_grade_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def plot_default_rate_by_subgrade(
    df_default_rate_by_subgrade: pd.DataFrame,
    split_column: str,
    subgrade_column: str,
    default_rate_column: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> plt.Figure:
    """
    Plot default rate by subgrade for training and testing datasets.

    Parameters
    ----------
    df_default_rate_by_subgrade : pd.DataFrame
        Combined default-rate table containing one row per split-subgrade
        combination.
    split_column : str
        Name of the split column (e.g., train/test).
    subgrade_column : str
        Name of the subgrade column.
    default_rate_column : str
        Name of the default-rate column.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by ``log_config.emit_log``.

    Returns
    -------
    plt.Figure
        Matplotlib figure showing default rate by subgrade for train and test.

    Raises
    ------
    ValueError
        If required columns are missing or expected split rows are absent.
    """
    try:
        log_config.emit_log(log, "Building default rate by subgrade figure")

        required_columns = {split_column, subgrade_column, default_rate_column}
        missing_columns = required_columns - set(df_default_rate_by_subgrade.columns)

        if missing_columns:
            raise ValueError(
                {
                    "stage": "plot_default_rate_by_subgrade",
                    "error": "missing_required_columns",
                    "missing_columns": sorted(missing_columns),
                }
            )

        df_default_rate_local = df_default_rate_by_subgrade.copy()

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

        df_train = df_default_rate_local[
            df_default_rate_local[split_column] == "train"
        ].sort_values(subgrade_column)

        df_test = df_default_rate_local[
            df_default_rate_local[split_column] == "test"
        ].sort_values(subgrade_column)

        if df_train.shape[0] != len(ordered_subgrades):
            raise ValueError(
                {
                    "stage": "plot_default_rate_by_subgrade",
                    "error": "unexpected_train_subgrade_count",
                    "observed_rows": df_train.shape[0],
                    "expected_rows": len(ordered_subgrades),
                }
            )

        if df_test.shape[0] != len(ordered_subgrades):
            raise ValueError(
                {
                    "stage": "plot_default_rate_by_subgrade",
                    "error": "unexpected_test_subgrade_count",
                    "observed_rows": df_test.shape[0],
                    "expected_rows": len(ordered_subgrades),
                }
            )

        figure, axis = plt.subplots(figsize=(14, 6))

        axis.plot(
            df_train[subgrade_column].astype(str),
            df_train[default_rate_column],
            marker="o",
            label="train",
        )
        axis.plot(
            df_test[subgrade_column].astype(str),
            df_test[default_rate_column],
            marker="o",
            label="test",
        )

        axis.set_title("Default Rate by Subgrade")
        axis.set_xlabel("Subgrade")
        axis.set_ylabel("Default Rate")
        axis.grid(True, alpha=0.3)
        axis.legend()

        figure.tight_layout()

        log_config.emit_log(
            log,
            {
                "stage": "plot_default_rate_by_subgrade_complete",
                "train_rows": df_train.shape[0],
                "test_rows": df_test.shape[0],
            },
        )

        return figure

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "plot_default_rate_by_subgrade_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def plot_risk_separation_roc(
    df_roc_curve_artifact: pd.DataFrame,
    dataset_name: str,
    system_name_column: str,
    dataset_name_column: str,
    false_positive_rate_column: str,
    true_positive_rate_column: str,
    auc_column: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> plt.Figure:
    """
    Plot ROC curves for a single dataset split using a combined ROC artifact table.

    Parameters
    ----------
    df_roc_curve_artifact : pd.DataFrame
        Combined ROC curve artifact table containing one or more systems across
        one or more dataset splits.
    dataset_name : str
        Dataset split to plot (e.g., "train" or "test").
    system_name_column : str
        Name of the system-name column.
    dataset_name_column : str
        Name of the dataset-name column.
    false_positive_rate_column : str
        Name of the false positive rate column.
    true_positive_rate_column : str
        Name of the true positive rate column.
    auc_column : str
        Name of the AUC column.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by ``log_config.emit_log``.

    Returns
    -------
    plt.Figure
        ROC figure for the requested dataset split.

    Raises
    ------
    ValueError
        If required columns are missing, the requested dataset split is not
        present, or no valid systems are available for plotting.
    """

    try:
        log_config.emit_log(
            log,
            f"Building risk separation ROC figure for dataset={dataset_name}",
        )

        if df_roc_curve_artifact is None:
            raise ValueError(
                {
                    "stage": "plot_risk_separation_roc",
                    "error": "input_is_none",
                }
            )

        if not isinstance(df_roc_curve_artifact, pd.DataFrame):
            raise ValueError(
                {
                    "stage": "plot_risk_separation_roc",
                    "error": "input_is_not_dataframe",
                    "input_type": type(df_roc_curve_artifact).__name__,
                }
            )

        if df_roc_curve_artifact.empty:
            raise ValueError(
                {
                    "stage": "plot_risk_separation_roc",
                    "error": "input_is_empty",
                }
            )

        if not dataset_name or not str(dataset_name).strip():
            raise ValueError(
                {
                    "stage": "plot_risk_separation_roc",
                    "error": "dataset_name_is_empty",
                }
            )

        required_columns = {
            system_name_column,
            dataset_name_column,
            false_positive_rate_column,
            true_positive_rate_column,
            auc_column,
        }
        missing_columns = required_columns - set(df_roc_curve_artifact.columns)

        if missing_columns:
            raise KeyError(
                {
                    "stage": "plot_risk_separation_roc",
                    "error": "missing_required_columns",
                    "missing_columns": sorted(missing_columns),
                }
            )

        df_roc_curve_local = df_roc_curve_artifact.copy()

        available_datasets = sorted(
            df_roc_curve_local[dataset_name_column].dropna().astype(str).unique().tolist()
        )

        if dataset_name not in available_datasets:
            raise ValueError(
                {
                    "stage": "plot_risk_separation_roc",
                    "error": "dataset_name_not_found",
                    "dataset_name": dataset_name,
                    "available_datasets": available_datasets,
                }
            )

        df_plot = df_roc_curve_local[
            df_roc_curve_local[dataset_name_column].astype(str) == dataset_name
        ].copy()

        if df_plot.empty:
            raise ValueError(
                {
                    "stage": "plot_risk_separation_roc",
                    "error": "no_rows_for_dataset",
                    "dataset_name": dataset_name,
                }
            )

        if df_plot[false_positive_rate_column].isna().any():
            raise ValueError(
                {
                    "stage": "plot_risk_separation_roc",
                    "error": "missing_false_positive_rate_values",
                    "missing_rows": int(df_plot[false_positive_rate_column].isna().sum()),
                }
            )

        if df_plot[true_positive_rate_column].isna().any():
            raise ValueError(
                {
                    "stage": "plot_risk_separation_roc",
                    "error": "missing_true_positive_rate_values",
                    "missing_rows": int(df_plot[true_positive_rate_column].isna().sum()),
                }
            )

        if df_plot[auc_column].isna().any():
            raise ValueError(
                {
                    "stage": "plot_risk_separation_roc",
                    "error": "missing_auc_values",
                    "missing_rows": int(df_plot[auc_column].isna().sum()),
                }
            )

        system_names = sorted(
            df_plot[system_name_column].dropna().astype(str).unique().tolist()
        )

        if not system_names:
            raise ValueError(
                {
                    "stage": "plot_risk_separation_roc",
                    "error": "no_systems_found",
                    "dataset_name": dataset_name,
                }
            )

        figure, axis = plt.subplots(figsize=(8, 6))

        for system_name in system_names:
            df_system = df_plot[
                df_plot[system_name_column].astype(str) == system_name
            ].copy()

            if df_system.empty:
                raise ValueError(
                    {
                        "stage": "plot_risk_separation_roc",
                        "error": "empty_system_subset",
                        "dataset_name": dataset_name,
                        "system_name": system_name,
                    }
                )

            df_system = df_system.sort_values(
                by=[false_positive_rate_column, true_positive_rate_column]
            )

            auc_values = df_system[auc_column].dropna().unique().tolist()

            if len(auc_values) != 1:
                raise ValueError(
                    {
                        "stage": "plot_risk_separation_roc",
                        "error": "unexpected_auc_values_per_system",
                        "dataset_name": dataset_name,
                        "system_name": system_name,
                        "auc_values": auc_values,
                    }
                )

            auc_value = float(auc_values[0])

            axis.plot(
                df_system[false_positive_rate_column],
                df_system[true_positive_rate_column],
                label=f"{system_name} (AUC={auc_value:.3f})",
            )

        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")

        axis.set_title(f"ROC Curve — {dataset_name.capitalize()}")
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(True, alpha=0.3)
        axis.legend()

        figure.tight_layout()

        log_config.emit_log(
            log,
            {
                "stage": "plot_risk_separation_roc_complete",
                "dataset_name": dataset_name,
                "system_count": len(system_names),
                "rows": df_plot.shape[0],
            },
        )

        return figure

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "plot_risk_separation_roc_failed",
                "dataset_name": dataset_name if "dataset_name" in locals() else "unknown",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def plot_calibration_curve(
    df_calibration_artifact: pd.DataFrame,
    dataset_name: str,
    system_name_column: str,
    dataset_name_column: str,
    predicted_probability_column: str,
    observed_default_rate_column: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> plt.Figure:
    """
    Plot calibration curves for a single dataset split using a combined
    calibration artifact table.
    """

    try:
        log_config.emit_log(
            log,
            f"Building calibration figure for dataset={dataset_name}",
        )

        if df_calibration_artifact is None:
            raise ValueError(
                {
                    "stage": "plot_calibration_curve",
                    "error": "input_is_none",
                }
            )

        if not isinstance(df_calibration_artifact, pd.DataFrame):
            raise ValueError(
                {
                    "stage": "plot_calibration_curve",
                    "error": "input_is_not_dataframe",
                    "input_type": type(df_calibration_artifact).__name__,
                }
            )

        if df_calibration_artifact.empty:
            raise ValueError(
                {
                    "stage": "plot_calibration_curve",
                    "error": "input_is_empty",
                }
            )

        required_columns = {
            system_name_column,
            dataset_name_column,
            predicted_probability_column,
            observed_default_rate_column,
        }
        missing_columns = required_columns - set(df_calibration_artifact.columns)

        if missing_columns:
            raise KeyError(
                {
                    "stage": "plot_calibration_curve",
                    "error": "missing_required_columns",
                    "missing_columns": sorted(missing_columns),
                }
            )

        available_datasets = sorted(
            df_calibration_artifact[dataset_name_column].dropna().astype(str).unique().tolist()
        )

        if dataset_name not in available_datasets:
            raise ValueError(
                {
                    "stage": "plot_calibration_curve",
                    "error": "dataset_name_not_found",
                    "dataset_name": dataset_name,
                    "available_datasets": available_datasets,
                }
            )

        df_plot = df_calibration_artifact[
            df_calibration_artifact[dataset_name_column].astype(str) == dataset_name
        ].copy()

        if df_plot.empty:
            raise ValueError(
                {
                    "stage": "plot_calibration_curve",
                    "error": "no_rows_for_dataset",
                    "dataset_name": dataset_name,
                }
            )

        system_names = sorted(
            df_plot[system_name_column].dropna().astype(str).unique().tolist()
        )

        if not system_names:
            raise ValueError(
                {
                    "stage": "plot_calibration_curve",
                    "error": "no_systems_found",
                    "dataset_name": dataset_name,
                }
            )

        figure, axis = plt.subplots(figsize=(8, 6))

        for system_name in system_names:
            df_system = df_plot[
                df_plot[system_name_column].astype(str) == system_name
            ].copy()

            axis.plot(
                df_system[predicted_probability_column],
                df_system[observed_default_rate_column],
                marker="o",
                label=system_name,
            )

        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")

        axis.set_title(f"Calibration Curve — {dataset_name.capitalize()}")
        axis.set_xlabel("Mean Predicted Probability")
        axis.set_ylabel("Observed Default Rate")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(True, alpha=0.3)
        axis.legend()

        figure.tight_layout()

        log_config.emit_log(
            log,
            {
                "stage": "plot_calibration_curve_complete",
                "dataset_name": dataset_name,
                "system_count": len(system_names),
                "rows": df_plot.shape[0],
            },
        )

        return figure

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "plot_calibration_curve_failed",
                "dataset_name": dataset_name if "dataset_name" in locals() else "unknown",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise