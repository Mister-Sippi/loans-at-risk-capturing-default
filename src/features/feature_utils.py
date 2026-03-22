from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

import config.logging as log_config


def _format_column_list(
    columns: list[str],
    max_items: int = 25,
) -> str:
    """
    Format a list of column names for logging, truncating long lists
    to keep log output readable.
    """
    if len(columns) <= max_items:
        return ", ".join(columns)

    head = ", ".join(columns[:max_items])
    remaining_count = len(columns) - max_items

    return f"{head}, ... (+{remaining_count} more)"


def drop_columns_with_logging(
    df: pd.DataFrame,
    columns_to_drop: list[str],
    dataset_name: str,
    log: Callable[[str], None] | Path | str | None = None,
    errors: str = "ignore",
) -> pd.DataFrame:
    """
    Drop columns from a DataFrame and log what happened.
    """
    try:
        if df is None:
            raise ValueError("df must not be None")

        if not dataset_name or not str(dataset_name).strip():
            raise ValueError("dataset_name must be a non-empty string")

        shape_before = df.shape

        existing_columns = set(df.columns)
        requested_columns = list(dict.fromkeys(columns_to_drop))

        dropped_columns = [
            column_name for column_name in requested_columns if column_name in existing_columns
        ]

        missing_columns = [
            column_name for column_name in requested_columns if column_name not in existing_columns
        ]

        transformed_dataframe = df.drop(columns=requested_columns, errors=errors)

        shape_after = transformed_dataframe.shape

        log_config.emit_log(log, f"[drop_columns] dataset={dataset_name}")
        log_config.emit_log(log, f"[drop_columns] shape_before={shape_before} shape_after={shape_after}")

        log_config.emit_log(
            log,
            f"[drop_columns] requested={len(requested_columns)} "
            f"dropped={len(dropped_columns)} missing={len(missing_columns)} errors={errors}",
        )

        if dropped_columns:
            log_config.emit_log(log, "[drop_columns] dropped_columns=" + _format_column_list(dropped_columns))

        if missing_columns:
            log_config.emit_log(log, "[drop_columns] missing_columns=" + _format_column_list(missing_columns))

        return transformed_dataframe

    except Exception as exc:
        try:
            log_config.emit_log(
                log,
                f"[drop_columns_with_logging][error] dataset={dataset_name} "
                f"Error={type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise

