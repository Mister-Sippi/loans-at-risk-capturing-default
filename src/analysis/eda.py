from typing import Callable, Sequence
import pandas as pd

from config.logging import get_logger


def build_terminal_cohort(
    dataframe: pd.DataFrame,
    *,
    status_column: str,
    terminal_statuses: Sequence[str],
    positive_statuses: Sequence[str],
    target_column: str = "target_default",
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Filter to terminal statuses and add a binary target column.

    Notes:
    - Returns a copy (does not mutate input)
    - Intended for EDA + modeling preparation
    """
    log: Callable[[str], None] = get_logger(log_file)

    try:
        if status_column not in dataframe.columns:
            raise KeyError(f"build_terminal_cohort: missing required column '{status_column}'")

        terminal_mask = dataframe[status_column].isin(list(terminal_statuses))
        terminal_dataframe = dataframe.loc[terminal_mask].copy()

        terminal_dataframe[target_column] = (
            terminal_dataframe[status_column]
            .isin(list(positive_statuses))
            .astype(int)
        )

        positive_rate = float(terminal_dataframe[target_column].mean())

        log(
            "build_terminal_cohort: terminal cohort created | "
            f"rows={terminal_dataframe.shape[0]}, "
            f"columns={terminal_dataframe.shape[1]}, "
            f"positive_rate={positive_rate:.4f}"
        )

        return terminal_dataframe

    except Exception as error:
        log(f"Error in build_terminal_cohort: {error}")
        raise
