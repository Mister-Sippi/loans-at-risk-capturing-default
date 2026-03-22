from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import config.logging as log_config


#==============================================================================
#EDA artifact figures
#==============================================================================


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