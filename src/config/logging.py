from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import pandas as pd


# -------------------------------
# Logging
# -------------------------------

def log_messages(message: str, log_file: Path | str) -> None:
    """Log a message with a UTC timestamp to a file."""
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        full_message = f"[{timestamp}] {message}\n"
        with open(log_path, "a", encoding="utf-8") as log_file_handle:
            log_file_handle.write(full_message)
    except Exception as exc:
        print(f"[Logging Failure] {exc}")
        raise


def get_logger(log_file: Path | str | None) -> Callable[[str], None]:
    """
    Return a lightweight logger function.

    If log_file is provided, messages are written to file.
    If log_file is None, logging is disabled.
    """
    if log_file is None:
        def log(_: str) -> None:
            return
        return log

    log_path = Path(log_file)

    def log(message: str) -> None:
        log_messages(message, log_path)

    return log


def log_dataframe_checkpoint(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    checkpoint_name: str,
    log: Callable[[str], None],
    id_column_name: Optional[str] = None,
    extra_metrics: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Log a high-signal checkpoint summary for a dataframe.

    Intended usage:
    - call after major phases (row_id creation, structural drops, normalization, save)
    - log only metrics that detect breakage or unintended drift
    """
    try:
        row_count = int(df.shape[0])
        column_count = int(df.shape[1])

        base_message_parts = [
            f"[{checkpoint_name}][{dataset_name}]",
            f"rows={row_count}",
            f"cols={column_count}",
        ]

        # Optional id-column checks (only if caller requested one)
        if id_column_name:
            id_column_present = id_column_name in df.columns
            base_message_parts.append(f"{id_column_name}_present={id_column_present}")

            id_null_count: Optional[int] = None
            id_is_unique: Optional[bool] = None

            if id_column_present:
                id_null_count = int(df[id_column_name].isna().sum())
                id_is_unique = df[id_column_name].is_unique

                base_message_parts.extend(
                    [
                        f"{id_column_name}_nulls={id_null_count}",
                        f"{id_column_name}_unique={id_is_unique}",
                    ]
                )

        # Extra metrics
        if extra_metrics:
            for metric_name, metric_value in extra_metrics.items():
                base_message_parts.append(f"{metric_name}={metric_value}")

        log(" | ".join(base_message_parts))

        # Warnings (only if id column requested + present)
        if id_column_name and (id_column_name in df.columns):
            if id_null_count is not None and id_null_count > 0:
                log(
                    f"[{checkpoint_name}][{dataset_name}][warning] "
                    f"{id_column_name} contains nulls (nulls={id_null_count})."
                )

            if id_is_unique is False:
                log(
                    f"[{checkpoint_name}][{dataset_name}][warning] "
                    f"{id_column_name} is not unique."
                )

    except Exception as exc:
        log(
            f"[log_dataframe_checkpoint][error] "
            f"Failed checkpoint logging for dataset_name={dataset_name}, checkpoint_name={checkpoint_name}. "
            f"Error={type(exc).__name__}: {exc}"
        )
        raise


def emit_log(
    log: Callable[[str], None] | Path | str | None,
    message: str,
) -> None:
    """Emit a log message to a callable logger or append to a log file path."""
    if log is None:
        return

    if callable(log):
        log(message)
        return

    if isinstance(log, (Path, str)):
        log_path = Path(log)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{message}\n")
        return

    raise TypeError("log must be a callable, Path, str, or None")