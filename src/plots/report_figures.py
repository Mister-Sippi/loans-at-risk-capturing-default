from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

import config.logging as log_config


# ==============================================================================
# EDA artifact figures
# ==============================================================================


def plot_portfolio_evolution_figure(
    issuance_volume_table: pd.DataFrame,
    default_rate_table: pd.DataFrame,
    *,
    output_path: Path | str = "artifacts/eda/figures/portfolio_evolution.png",
    issue_year_column: str = "issue_year",
    dataset_split_column: str = "dataset_split",
    issuance_volume_column: str = "loan_count",
    default_rate_column: str = "default_rate_percent",
    dataset_split_order: Sequence[str] = ("train", "test"),
    log: Callable[[str], None] | Path | str | None = None,
) -> Path:
    """
    Create and save the EDA artifact figure showing LendingClub portfolio evolution.
    """
    try:
        required_issuance_columns = {
            issue_year_column,
            dataset_split_column,
            issuance_volume_column,
        }
        required_default_columns = {
            issue_year_column,
            dataset_split_column,
            default_rate_column,
        }

        missing_issuance_columns = [
            column_name
            for column_name in required_issuance_columns
            if column_name not in issuance_volume_table.columns
        ]

        missing_default_columns = [
            column_name
            for column_name in required_default_columns
            if column_name not in default_rate_table.columns
        ]

        if missing_issuance_columns:
            raise KeyError(
                "plot_portfolio_evolution_figure: issuance table missing required columns "
                f"{sorted(missing_issuance_columns)}"
            )

        if missing_default_columns:
            raise KeyError(
                "plot_portfolio_evolution_figure: default-rate table missing required columns "
                f"{sorted(missing_default_columns)}"
            )

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure, axis_array = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(12, 4.8),
        )

        issuance_axis = axis_array[0]
        default_rate_axis = axis_array[1]

        for dataset_split_name in dataset_split_order:
            issuance_split_table = (
                issuance_volume_table.loc[
                    issuance_volume_table[dataset_split_column] == dataset_split_name
                ]
                .copy()
                .sort_values(issue_year_column)
            )

            if issuance_split_table.empty:
                continue

            issuance_axis.plot(
                issuance_split_table[issue_year_column],
                issuance_split_table[issuance_volume_column],
                marker="o",
                label=dataset_split_name,
            )

        issuance_axis.set_title("Loan Issuance Volume by Issue Year")
        issuance_axis.set_xlabel("Issue Year")
        issuance_axis.set_ylabel("Loan Count")
        issuance_axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
        issuance_axis.legend()

        for dataset_split_name in dataset_split_order:
            default_rate_split_table = (
                default_rate_table.loc[
                    default_rate_table[dataset_split_column] == dataset_split_name
                ]
                .copy()
                .sort_values(issue_year_column)
            )

            if default_rate_split_table.empty:
                continue

            default_rate_axis.plot(
                default_rate_split_table[issue_year_column],
                default_rate_split_table[default_rate_column],
                marker="o",
                label=dataset_split_name,
            )

        default_rate_axis.set_title("Default Rate by Issue Year")
        default_rate_axis.set_xlabel("Issue Year")
        default_rate_axis.set_ylabel("Default Rate (%)")
        default_rate_axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
        default_rate_axis.legend()

        figure.suptitle("LendingClub Portfolio Evolution")
        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)

        log_config.emit_log(
            log,
            "plot_portfolio_evolution_figure completed | "
            f"output_path=Path('{output_path}')",
        )

        return output_path

    except Exception as exception:

        log_config.emit_log(
            log,
            "plot_portfolio_evolution_figure failed | "
            f"output_path={output_path if 'output_path' in locals() else 'not_created'} | "
            f"error={type(exception).__name__}: {exception}",
        )
        raise


def plot_log_transformation_comparison_figure(
    df_train: pd.DataFrame,
    feature_names: list[str],
    output_path: Path,
    log: Callable[[str], None] | Path | str | None = None,
) -> Path:
    """
    Create a report figure comparing raw and log-transformed distributions
    for selected numeric features.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataset containing the selected numeric features.
    feature_names : list[str]
        Numeric feature names to visualize. These should be representative
        features selected for log transformation.
    output_path : Path
        File path where the figure will be saved.
    log : Callable[[str], None] | Path | str | None, default=None
        Callable logger or log file path supported by log_config.emit_log.

    Returns
    -------
    Path
        Saved figure path.

    Raises
    ------
    KeyError
        If one or more selected features are not present in the dataset.
    ValueError
        If no features are provided or if all selected features are empty.
    """
    try:
        if not feature_names:
            raise ValueError("feature_names must contain at least one feature.")

        missing_features = [
            feature_name
            for feature_name in feature_names
            if feature_name not in df_train.columns
        ]
        if missing_features:
            raise KeyError(
                f"Selected features not found in training data: {missing_features}"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        valid_feature_names: list[str] = []
        for feature_name in feature_names:
            feature_series = df_train[feature_name].dropna().copy()
            if not feature_series.empty:
                valid_feature_names.append(feature_name)

        if not valid_feature_names:
            raise ValueError("All selected features are empty after dropping missing values.")

        figure, axes = plt.subplots(
            nrows=len(valid_feature_names),
            ncols=2,
            figsize=(12, 3.8 * len(valid_feature_names)),
        )

        if len(valid_feature_names) == 1:
            axes = np.array([axes])

        for row_index, feature_name in enumerate(valid_feature_names):
            feature_series = df_train[feature_name].dropna().copy()

            if (feature_series < 0).any():
                raise ValueError(
                    f"Feature '{feature_name}' contains negative values and cannot be log1p-transformed safely."
                )

            log_transformed_series = np.log1p(feature_series)

            raw_axis = axes[row_index, 0]
            log_axis = axes[row_index, 1]

            raw_axis.hist(
                feature_series,
                bins=40,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
            )
            raw_axis.set_title(f"{feature_name} — Raw")
            raw_axis.set_xlabel(feature_name)
            raw_axis.set_ylabel("Frequency")

            log_axis.hist(
                log_transformed_series,
                bins=40,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
            )
            log_axis.set_title(f"{feature_name} — log1p")
            log_axis.set_xlabel(f"log1p({feature_name})")
            log_axis.set_ylabel("Frequency")

        figure.suptitle(
            "Representative Numeric Features Before and After Log Transformation",
            fontsize=14,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)

        log_config.emit_log(
            log=log,
            message=(
                "[log_transformation_comparison_figure] "
                f"Saved figure for {len(valid_feature_names)} features to {output_path}"
            ),
        )

        return output_path

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=(
                "[log_transformation_comparison_figure] "
                f"Failed to create figure: {exc}"
            ),
        )
        raise
    

# ==============================================================================
# Modeling artifact figures
# ==============================================================================


def plot_model_roc_comparison_figure(
    model_curves: list[dict[str, object]],
    output_path: Path,
    log: Callable[[str], None] | Path | str | None = None,
) -> Path:
    """
    Plot ROC curves for multiple models on a single figure.

    Parameters
    ----------
    model_curves : list[dict[str, object]]
        List of dictionaries with keys:
        - "model_name": str
        - "y_true": pd.Series | np.ndarray
        - "y_score": pd.Series | np.ndarray
    output_path : Path
        Output file path for the saved figure.
    log : Callable[[str], None] | Path | str | None, default=None
        Logging target supported by log_config.emit_log.

    Returns
    -------
    Path
        Saved figure path.
    """
    try:
        if not model_curves:
            raise ValueError("model_curves must contain at least one model specification.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure, axis = plt.subplots(figsize=(8, 6))

        for model_curve in model_curves:
            if not isinstance(model_curve, dict):
                raise ValueError("Each model_curve must be a dictionary.")

            required_keys = {"model_name", "y_true", "y_score"}
            missing_keys = required_keys - set(model_curve.keys())
            if missing_keys:
                raise KeyError(
                    f"Each model_curve must contain keys {sorted(required_keys)}. "
                    f"Missing keys: {sorted(missing_keys)}"
                )

            model_name = str(model_curve["model_name"])
            y_true = model_curve["y_true"]
            y_score = model_curve["y_score"]

            RocCurveDisplay.from_predictions(
                y_true=y_true,
                y_score=y_score,
                name=model_name,
                ax=axis,
            )

        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        axis.set_title("ROC Curve Comparison")
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.grid(alpha=0.25)

        figure.tight_layout()
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)

        log_config.emit_log(
            log=log,
            message=(
                "[plot_model_roc_comparison_figure] "
                f"Saved figure for {len(model_curves)} models to {output_path}"
            ),
        )

        return output_path

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=f"[plot_model_roc_comparison_figure][error] {type(exc).__name__}: {exc}",
        )
        raise


def plot_model_calibration_figure(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    model_name: str,
    output_path: Path,
    n_bins: int = 10,
    strategy: str = "quantile",
    log: Callable[[str], None] | Path | str | None = None,
) -> Path:
    """
    Plot calibration curve for a single model.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        True binary labels.
    y_score : pd.Series | np.ndarray
        Predicted probabilities for the positive class.
    model_name : str
        Model label for the plot.
    output_path : Path
        Output file path for the saved figure.
    n_bins : int, default=10
        Number of probability bins.
    strategy : str, default="quantile"
        Binning strategy passed to sklearn.calibration.calibration_curve.
    log : Callable[[str], None] | Path | str | None, default=None
        Logging target supported by log_config.emit_log.

    Returns
    -------
    Path
        Saved figure path.
    """
    try:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        if n_bins <= 1:
            raise ValueError("n_bins must be greater than 1.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true=y_true,
            y_prob=y_score,
            n_bins=n_bins,
            strategy=strategy,
        )

        figure, axis = plt.subplots(figsize=(8, 6))
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
        axis.plot(
            mean_predicted_value,
            fraction_of_positives,
            marker="o",
            linewidth=1.5,
            label=model_name,
        )

        axis.set_title(f"Calibration Curve — {model_name}")
        axis.set_xlabel("Mean Predicted Probability")
        axis.set_ylabel("Observed Default Rate")
        axis.grid(alpha=0.25)
        axis.legend()

        figure.tight_layout()
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)

        log_config.emit_log(
            log=log,
            message=(
                "[plot_model_calibration_figure] "
                f"Saved calibration figure for model={model_name} to {output_path}"
            ),
        )

        return output_path

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=f"[plot_model_calibration_figure][error] {type(exc).__name__}: {exc}",
        )
        raise


def plot_confusion_matrix_figure(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    output_path: Path,
    labels: list[int] | None = None,
    log: Callable[[str], None] | Path | str | None = None,
) -> Path:
    """
    Plot confusion matrix for a single model.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        True binary labels.
    y_pred : pd.Series | np.ndarray
        Predicted binary labels.
    model_name : str
        Model label for the plot.
    output_path : Path
        Output file path for the saved figure.
    labels : list[int] | None, default=None
        Label order for the confusion matrix.
    log : Callable[[str], None] | Path | str | None, default=None
        Logging target supported by log_config.emit_log.

    Returns
    -------
    Path
        Saved figure path.
    """
    try:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure, axis = plt.subplots(figsize=(6, 6))

        display = ConfusionMatrixDisplay.from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            ax=axis,
            colorbar=False,
        )

        for text in display.text_.ravel():
            text.set_fontsize(10)

        axis.set_title(f"Confusion Matrix — {model_name}")
        axis.set_xlabel("Predicted Label")
        axis.set_ylabel("True Label")

        figure.tight_layout()
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)

        log_config.emit_log(
            log=log,
            message=(
                "[plot_confusion_matrix_figure] "
                f"Saved confusion matrix figure for model={model_name} to {output_path}"
            ),
        )

        return output_path

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=f"[plot_confusion_matrix_figure][error] {type(exc).__name__}: {exc}",
        )
        raise


def plot_monetary_outcome_comparison_figure(
    outcome_summary_df: pd.DataFrame,
    output_path: Path,
    selected_models: list[str] | None = None,
    value_column: str = "total_loan_amnt",
    log: Callable[[str], None] | Path | str | None = None,
) -> Path:
    """
    Plot grouped bar chart comparing monetary outcomes across models.

    Parameters
    ----------
    outcome_summary_df : pd.DataFrame
        Outcome summary table containing at least:
        - model_name
        - outcome_type
        - total_loan_amnt
    output_path : Path
        Output file path for the saved figure.
    selected_models : list[str] | None, default=None
        Optional subset of models to include.
    value_column : str, default="total_loan_amnt"
        Monetary column to plot.
    log : Callable[[str], None] | Path | str | None, default=None
        Logging target supported by log_config.emit_log.

    Returns
    -------
    Path
        Saved figure path.
    """
    try:
        if outcome_summary_df is None:
            raise ValueError("outcome_summary_df must not be None.")

        if not isinstance(outcome_summary_df, pd.DataFrame):
            raise ValueError("outcome_summary_df must be a pandas DataFrame.")

        if outcome_summary_df.empty:
            raise ValueError("outcome_summary_df must not be empty.")

        required_columns = {"model_name", "outcome_type", value_column}
        missing_columns = required_columns - set(outcome_summary_df.columns)
        if missing_columns:
            raise KeyError(
                f"outcome_summary_df missing required columns: {sorted(missing_columns)}"
            )

        plot_df = outcome_summary_df.copy()

        if selected_models is not None:
            if not isinstance(selected_models, list) or not selected_models:
                raise ValueError("selected_models must be a non-empty list or None.")

            plot_df = plot_df.loc[plot_df["model_name"].isin(selected_models)].copy()

            if plot_df.empty:
                raise ValueError("No rows remain after filtering outcome_summary_df by selected_models.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        outcome_order = [
            "false_negative",
            "false_positive",
            "true_negative",
            "true_positive",
        ]
        plot_df["outcome_type"] = pd.Categorical(
            plot_df["outcome_type"],
            categories=outcome_order,
            ordered=True,
        )

        pivot_df = (
            plot_df.pivot(index="outcome_type", columns="model_name", values=value_column)
            .reindex(outcome_order)
        )

        figure, axis = plt.subplots(figsize=(10, 6))

        pivot_df.plot(
            kind="bar",
            ax=axis,
            edgecolor="black",
            linewidth=0.8,
        )

        axis.set_title("Loan Amount by Outcome Type and Model")
        axis.set_xlabel("Outcome Type")
        axis.set_ylabel("Total Loan Amount")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Model")

        figure.tight_layout()
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)

        log_config.emit_log(
            log=log,
            message=(
                "[plot_monetary_outcome_comparison_figure] "
                f"Saved figure to {output_path}"
            ),
        )

        return output_path

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=f"[plot_monetary_outcome_comparison_figure][error] {type(exc).__name__}: {exc}",
        )
        raise


# ==============================================================================
# Validation artifact figures
# ==============================================================================

def plot_report_roc_curve(
    df_roc_curve_artifact: pd.DataFrame,
    system_name_column: str = "system_name",
    dataset_name_column: str = "dataset_name",
    false_positive_rate_column: str = "false_positive_rate",
    true_positive_rate_column: str = "true_positive_rate",
    auc_column: str = "auc",
    dataset_name: str = "test",
    log: Callable[[str], None] | Path | str | None = None,
) -> Figure:
    """
    Plot ROC curves for the model and baseline on a single dataset split.

    Parameters
    ----------
    df_roc_curve_artifact : pd.DataFrame
        Combined ROC artifact table.
    system_name_column : str, default="system_name"
        Column containing system names.
    dataset_name_column : str, default="dataset_name"
        Column containing dataset split names.
    false_positive_rate_column : str, default="false_positive_rate"
        Column containing false positive rates.
    true_positive_rate_column : str, default="true_positive_rate"
        Column containing true positive rates.
    auc_column : str, default="auc"
        Column containing AUC values.
    dataset_name : str, default="test"
        Dataset split to plot.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger compatible with emit_log.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "plot_report_roc_curve_started",
                "rows": df_roc_curve_artifact.shape[0],
                "dataset_name": dataset_name,
            },
        )

        df_plot = df_roc_curve_artifact.copy()

        required_columns = [
            system_name_column,
            dataset_name_column,
            false_positive_rate_column,
            true_positive_rate_column,
            auc_column,
        ]
        missing_columns = [
            column_name
            for column_name in required_columns
            if column_name not in df_plot.columns
        ]
        if missing_columns:
            raise KeyError(
                f"Missing required columns for report ROC curve: {missing_columns}"
            )

        df_plot = (
            df_plot.loc[df_plot[dataset_name_column] == dataset_name]
            .copy()
        )

        if df_plot.empty:
            raise ValueError(
                f"No ROC rows found for dataset_name='{dataset_name}'."
            )

        figure, axis = plt.subplots(figsize=(8, 6))

        color_map = {
            "model": "tab:blue",
            "baseline": "tab:orange",
        }

        system_names = list(df_plot[system_name_column].dropna().unique())
        for system_name in system_names:
            df_system = (
                df_plot.loc[df_plot[system_name_column] == system_name]
                .sort_values(false_positive_rate_column)
                .copy()
            )

            auc_value = float(df_system[auc_column].iloc[0])

            axis.plot(
                df_system[false_positive_rate_column],
                df_system[true_positive_rate_column],
                linewidth=2,
                color=color_map.get(system_name.lower(), "tab:gray"),
                label=f"{system_name} (AUC={auc_value:.3f})",
            )

        axis.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            linewidth=1.5,
            color="gray",
            alpha=0.7,
            label="No-skill reference",
        )

        axis.set_title(f"ROC Curve — {dataset_name.capitalize()} Set")
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
                "stage": "plot_report_roc_curve_completed",
                "dataset_name": dataset_name,
                "systems": system_names,
            },
        )

        return figure

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "plot_report_roc_curve_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def plot_report_calibration_curve(
    df_calibration_artifact: pd.DataFrame,
    dataset_name: str = "test",
    system_name_column: str = "system_name",
    dataset_name_column: str = "dataset_name",
    predicted_probability_column: str = "predicted_probability_mean",
    observed_default_rate_column: str = "observed_default_rate",
    log: Callable[[str], None] | Path | str | None = None,
) -> Figure:
    """
    Plot calibration curve for a single dataset split.

    Parameters
    ----------
    df_calibration_artifact : pd.DataFrame
        Combined calibration artifact table.
    dataset_name : str, default="test"
        Dataset split to plot.
    system_name_column : str, default="system_name"
        Column containing system names.
    dataset_name_column : str, default="dataset_name"
        Column containing dataset split names.
    predicted_probability_column : str, default="predicted_probability_mean"
        Column containing mean predicted probabilities.
    observed_default_rate_column : str, default="observed_default_rate"
        Column containing observed default rates.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger compatible with emit_log.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "plot_report_calibration_curve_started",
                "rows": df_calibration_artifact.shape[0],
                "dataset_name": dataset_name,
            },
        )

        df_plot = df_calibration_artifact.copy()

        required_columns = [
            system_name_column,
            dataset_name_column,
            predicted_probability_column,
            observed_default_rate_column,
        ]
        missing_columns = [
            column_name
            for column_name in required_columns
            if column_name not in df_plot.columns
        ]
        if missing_columns:
            raise KeyError(
                f"Missing required columns for report calibration curve: {missing_columns}"
            )

        df_plot = (
            df_plot.loc[df_plot[dataset_name_column] == dataset_name]
            .copy()
        )

        if df_plot.empty:
            raise ValueError(
                f"No calibration rows found for dataset_name='{dataset_name}'."
            )

        figure, axis = plt.subplots(figsize=(8, 6))

        system_names = list(df_plot[system_name_column].dropna().unique())
        for system_name in system_names:
            df_system = (
                df_plot.loc[df_plot[system_name_column] == system_name]
                .sort_values(predicted_probability_column)
                .copy()
            )

            axis.plot(
                df_system[predicted_probability_column],
                df_system[observed_default_rate_column],
                marker="o",
                linewidth=2,
                label=system_name,
            )

        axis.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            linewidth=1.5,
            label="Perfect calibration",
        )

        axis.set_title(f"Calibration Curve — {dataset_name.capitalize()} Set")
        axis.set_xlabel("Predicted Default Probability")
        axis.set_ylabel("Observed Default Rate")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()

        log_config.emit_log(
            log,
            {
                "stage": "plot_report_calibration_curve_completed",
                "dataset_name": dataset_name,
                "systems": system_names,
            },
        )

        return figure

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "plot_report_calibration_curve_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def plot_report_risk_stratification(
    df_risk_band_summary: pd.DataFrame,
    band_column: str = "risk_band",
    predicted_risk_column: str = "predicted_default_probability_mean",
    observed_default_rate_column: str = "observed_default_rate",
    log: Callable[[str], None] | Path | str | None = None,
) -> Figure:
    """
    Plot predicted and observed risk across ordered risk bands.

    Parameters
    ----------
    df_risk_band_summary : pd.DataFrame
        Risk band summary table.
    band_column : str, default="risk_band"
        Column containing raw band intervals.
    predicted_risk_column : str, default="predicted_default_probability_mean"
        Column containing mean predicted risk.
    observed_default_rate_column : str, default="observed_default_rate"
        Column containing observed default rate.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger compatible with emit_log.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "plot_report_risk_stratification_started",
                "rows": df_risk_band_summary.shape[0],
            },
        )

        df_plot = df_risk_band_summary.copy()

        required_columns = [
            band_column,
            predicted_risk_column,
            observed_default_rate_column,
        ]
        missing_columns = [
            column_name
            for column_name in required_columns
            if column_name not in df_plot.columns
        ]
        if missing_columns:
            raise KeyError(
                f"Missing required columns for report risk stratification: {missing_columns}"
            )

        df_plot = (
            df_plot.sort_values(predicted_risk_column)
            .reset_index(drop=True)
            .copy()
        )

        def format_band_label(interval_text: str) -> str:
            cleaned_interval_text = (
                interval_text
                .replace("(", "")
                .replace("]", "")
            )
            lower_bound_text, upper_bound_text = cleaned_interval_text.split(",")
            lower_bound = float(lower_bound_text.strip())
            upper_bound = float(upper_bound_text.strip())
            return f"{lower_bound:.0%}–{upper_bound:.0%}"

        df_plot["risk_band_label"] = (
            df_plot[band_column]
            .astype(str)
            .map(format_band_label)
        )

        figure, axis = plt.subplots(figsize=(10, 6))

        axis.plot(
            df_plot["risk_band_label"],
            df_plot[predicted_risk_column],
            marker="o",
            linewidth=2,
            linestyle="--",
            label="Predicted default probability",
        )

        axis.plot(
            df_plot["risk_band_label"],
            df_plot[observed_default_rate_column],
            marker="o",
            linewidth=2,
            label="Observed default rate",
        )

        axis.set_title("Risk Stratification")
        axis.set_xlabel("Predicted Default Risk Band")
        axis.set_ylabel("Rate")
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.grid(True, alpha=0.3)
        axis.legend()
        plt.xticks(rotation=30)
        figure.tight_layout()

        log_config.emit_log(
            log,
            {
                "stage": "plot_report_risk_stratification_completed",
                "rows": df_plot.shape[0],
            },
        )

        return figure

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "plot_report_risk_stratification_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def plot_report_policy_frontier(
    df_model_policy_outcomes: pd.DataFrame,
    df_baseline_policy_outcomes: pd.DataFrame,
    dataset_name: str = "test",
    dataset_name_column: str = "dataset_name",
    acceptance_rate_column: str = "acceptance_rate",
    default_rate_column: str = "default_rate_among_accepted",
    model_label: str = "Model threshold policy",
    baseline_label: str = "Baseline subgrade policy",
    log: Callable[[str], None] | Path | str | None = None,
) -> Figure:
    """
    Plot policy frontier for model and baseline on a single dataset split.

    Parameters
    ----------
    df_model_policy_outcomes : pd.DataFrame
        Model policy outcomes table.
    df_baseline_policy_outcomes : pd.DataFrame
        Baseline policy outcomes table.
    dataset_name : str, default="test"
        Dataset split to plot.
    dataset_name_column : str, default="dataset_name"
        Column containing dataset split names.
    acceptance_rate_column : str, default="acceptance_rate"
        Column containing acceptance rates.
    default_rate_column : str, default="default_rate_among_accepted"
        Column containing accepted-loan default rates.
    model_label : str, default="Model threshold policy"
        Legend label for model line.
    baseline_label : str, default="Baseline subgrade policy"
        Legend label for baseline line.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger compatible with emit_log.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "plot_report_policy_frontier_started",
                "dataset_name": dataset_name,
                "model_rows": df_model_policy_outcomes.shape[0],
                "baseline_rows": df_baseline_policy_outcomes.shape[0],
            },
        )

        df_model_plot = df_model_policy_outcomes.copy()
        df_baseline_plot = df_baseline_policy_outcomes.copy()

        required_columns = [
            dataset_name_column,
            acceptance_rate_column,
            default_rate_column,
        ]

        for column_name in required_columns:
            if column_name not in df_model_plot.columns:
                raise KeyError(
                    f"Column '{column_name}' not found in model policy outcomes."
                )
            if column_name not in df_baseline_plot.columns:
                raise KeyError(
                    f"Column '{column_name}' not found in baseline policy outcomes."
                )

        df_model_plot = (
            df_model_plot.loc[df_model_plot[dataset_name_column] == dataset_name]
            .sort_values(acceptance_rate_column)
            .reset_index(drop=True)
            .copy()
        )
        df_baseline_plot = (
            df_baseline_plot.loc[df_baseline_plot[dataset_name_column] == dataset_name]
            .sort_values(acceptance_rate_column)
            .reset_index(drop=True)
            .copy()
        )

        if df_model_plot.empty or df_baseline_plot.empty:
            raise ValueError(
                f"Missing policy rows for dataset_name='{dataset_name}'."
            )

        figure, axis = plt.subplots(figsize=(10, 6))

        axis.plot(
            df_model_plot[acceptance_rate_column],
            df_model_plot[default_rate_column],
            marker="o",
            linewidth=2,
            label=model_label,
        )

        axis.plot(
            df_baseline_plot[acceptance_rate_column],
            df_baseline_plot[default_rate_column],
            marker="o",
            linestyle="--",
            linewidth=2,
            label=baseline_label,
        )

        axis.set_title(f"Policy Frontier — {dataset_name.capitalize()} Set")
        axis.set_xlabel("Acceptance Rate")
        axis.set_ylabel("Default Rate Among Accepted Loans")
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()

        log_config.emit_log(
            log,
            {
                "stage": "plot_report_policy_frontier_completed",
                "dataset_name": dataset_name,
            },
        )

        return figure

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "plot_report_policy_frontier_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def plot_proxy_economic_difference(
    df_policy_comparison: pd.DataFrame,
    acceptance_rate_column: str = "baseline_acceptance_rate",
    value_diff_column: str = "net_value_with_opportunity_cost_diff",
    title: str = "Proxy Economic Difference — Test Set",
    x_label: str = "Acceptance Rate",
    y_label: str = "Model Advantage ($, with Opportunity Cost)",
    log: Callable[[str], None] | Path | str | None = None,
) -> plt.Figure:
    """
    Plot proxy economic difference (model - baseline) across matched policies.

    This figure shows model advantage (including opportunity cost) as a function
    of acceptance rate. It is used in the Validation notebook to identify the
    operating region where the model outperforms the baseline.

    Returns
    -------
    plt.Figure
        Matplotlib figure containing the proxy economic difference plot.
    """
    try:
        log_config.emit_log(
            log,
            {
                "stage": "plot_proxy_economic_difference_started",
                "rows": df_policy_comparison.shape[0],
                "columns": df_policy_comparison.shape[1],
                "acceptance_rate_column": acceptance_rate_column,
                "value_diff_column": value_diff_column,
            },
        )

        for col in (acceptance_rate_column, value_diff_column):
            if col not in df_policy_comparison.columns:
                raise ValueError(f"Column '{col}' not found in policy comparison table.")

        df_plot = (
            df_policy_comparison
            .sort_values(acceptance_rate_column)
            .reset_index(drop=True)
            .copy()
        )

        figure, axis = plt.subplots(figsize=(10, 6))

        # Model advantage line
        axis.plot(
            df_plot[acceptance_rate_column],
            df_plot[value_diff_column],
            marker="o",
            linewidth=2,
            label="Model Advantage",
        )

        # Parity line (baseline reference)
        axis.axhline(
            y=0,
            linestyle="--",
            linewidth=1.5,
            color="orange",
            label="Baseline (parity)",
        )

        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)

        axis.set_xlim(0, 1)
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))

        axis.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: f"${value:,.0f}" if value >= 0 else f"-${abs(value):,.0f}"
            )
        )

        axis.grid(True, alpha=0.3)
        axis.legend()

        figure.tight_layout()

        log_config.emit_log(
            log,
            {
                "stage": "plot_proxy_economic_difference_completed",
                "rows": df_plot.shape[0],
            },
        )

        return figure

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "plot_proxy_economic_difference_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise