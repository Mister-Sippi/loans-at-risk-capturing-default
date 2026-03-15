from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

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