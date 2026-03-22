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
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Restrict a dataset to realized repayment outcomes and derive a binary target.
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