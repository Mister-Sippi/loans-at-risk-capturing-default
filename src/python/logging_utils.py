from datetime import datetime, timezone
from typing import Callable
import pandas as pd


# -------------------------------
# Logging
# -------------------------------

def log_messages(message: str, log_file: str) -> None:
    """Log a message with a UTC timestamp to a file."""
    try:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        full_message = f"[{timestamp}] {message}\n"
        with open(log_file, "a", encoding="utf-8") as log_file_handle:
            log_file_handle.write(full_message)
    except Exception as error:
        # Logging never breaks the pipeline
        print(f"[Logging Failure] {error}")


def get_logger(log_file: str | None) -> Callable[[str], None]:
    """
    Return a lightweight logger function.

    If log_file is provided, messages are written to file.
    If log_file is None, logging is disabled.
    """
    if log_file is None:
        def log(_: str) -> None:
            return
        return log

    def log(message: str) -> None:
        log_messages(message, log_file)

    return log


def log_category_differences(
    categorical_value_differences: dict[str, pd.DataFrame],
    *,
    max_values_to_log: int = 10,
    log_file: str | None = None,
) -> None:
    """
    Log categorical differences between training and test datasets.

    For each column:
    - total categories
    - count only in training
    - count only in test
    - sample of differing values (truncated)

    Designed for schema validation and drift monitoring.
    """

    log = get_logger(log_file)

    try:
        if not categorical_value_differences:
            log("log_category_differences: no categorical differences provided.")
            return

        log("===== CATEGORICAL VALUE COMPARISON SUMMARY =====")

        for column_name, comparison_df in categorical_value_differences.items():

            if comparison_df.empty:
                log(f"{column_name}: no categories found.")
                continue

            only_in_training = comparison_df[
                (comparison_df["present_in_training"]) &
                (~comparison_df["present_in_test"])
            ]["category"].tolist()

            only_in_test = comparison_df[
                (~comparison_df["present_in_training"]) &
                (comparison_df["present_in_test"])
            ]["category"].tolist()

            total_categories = len(comparison_df)

            log(
                f"{column_name} | "
                f"total={total_categories} | "
                f"only_in_training={len(only_in_training)} | "
                f"only_in_test={len(only_in_test)}"
            )

            if only_in_training:
                log(
                    f"  sample_only_in_training: "
                    + ", ".join(only_in_training[:max_values_to_log])
                    + (" ..." if len(only_in_training) > max_values_to_log else "")
                )

            if only_in_test:
                log(
                    f"  sample_only_in_test: "
                    + ", ".join(only_in_test[:max_values_to_log])
                    + (" ..." if len(only_in_test) > max_values_to_log else "")
                )

        log("===== END CATEGORICAL VALUE COMPARISON =====")

    except Exception as error:
        log(f"Error in log_category_differences: {error}")
        raise
