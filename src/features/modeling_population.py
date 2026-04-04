from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

import config.logging as log_config


def build_terminal_cohort(
    df: pd.DataFrame,
    *,
    status_column: str,
    terminal_statuses: Sequence[str],
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Restrict a dataset to realized repayment outcomes.

	Notes
	-----
	This function filters the dataset to terminal repayment statuses only.
	Binary target construction is performed separately by the calling code.
    """
    try:
        missing_columns = [column_name for column_name in [status_column] if column_name not in df.columns]
        if missing_columns:
            raise KeyError(
                "build_terminal_cohort: missing required columns "
                f"{missing_columns}"
            )

        df_terminal = df[df[status_column].isin(terminal_statuses)].copy()

        log_config.emit_log(
            log,
            "[build_terminal_cohort] completed | "
            f"rows={df_terminal.shape[0]}",
        )

        return df_terminal

    except Exception as exc:
        if log:
            log(f"build_terminal_cohort failed | error={exc}")
        raise
    

def create_target_default(
    df: pd.DataFrame,
    *,
    status_column: str,
    positive_statuses: Sequence[str],
    target_column: str = "target_default",
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Create a binary default target from loan status.

    This function maps loan status values to a binary target variable,
    where observations with statuses in `positive_statuses` are labeled
    as default (1) and all other observations are labeled as non-default (0).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing loan status information.
    status_column : str
        Column containing loan status values.
    positive_statuses : Sequence[str]
        Status values that should be treated as default events.
        For example: ["charged_off", "default"].
    target_column : str, default="target_default"
        Name of the output binary target column.
    log : Callable | Path | str | None
        Logger compatible with emit_log.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with the binary target column added.

    Notes
    -----
    This function defines the default event used throughout the modeling
    and validation pipeline. The choice of `positive_statuses` must remain
    consistent across all stages to ensure comparability of results.
    """
    try:
        if status_column not in df.columns:
            raise KeyError(
                f"create_target_default: missing required column '{status_column}'"
            )

        df_target = df.copy()

        df_target[target_column] = (
            df_target[status_column]
            .isin(positive_statuses)
            .astype("int64")
        )

        log_config.emit_log(
            log,
            "[create_target_default] completed | "
            f"positive_count={int(df_target[target_column].sum())}",
        )

        return df_target

    except Exception as exc:
        log_config.emit_log(
            log,
            f"[create_target_default][error] {type(exc).__name__}: {exc}",
        )
        raise