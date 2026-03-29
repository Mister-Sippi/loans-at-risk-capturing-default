import pandas as pd

from typing import Callable, Any
from pathlib import Path

import config.logging as log_config


def prepare_clean_validation_data(
    df_clean: pd.DataFrame,
    df_model_input: pd.DataFrame,
    target_column: str,
    row_identifier_column: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Prepare the clean dataset for validation by restricting it to the modeled
    row universe and ensuring that the binary target column is present.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Clean dataset containing contextual variables such as LendingClub grade
        and raw loan amount.
    df_model_input : pd.DataFrame
        Engineered model-input dataset used by the selected model.
    target_column : str
        Name of the binary target column to validate or construct.
    row_identifier_column : str
        Name of the row-level identifier used to align datasets.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by ``emit_log``.

    Returns
    -------
    pd.DataFrame
        Clean validation dataset restricted to the modeled population and
        containing the required target column.

    Raises
    ------
    ValueError
        If required columns are missing, row identifiers are duplicated,
        target construction fails, or row alignment does not hold.
    """
    try:
        log_config.emit_log(log, "Preparing clean validation dataset")

        required_clean_columns = {row_identifier_column}
        required_model_input_columns = {row_identifier_column}

        missing_clean_columns = required_clean_columns - set(df_clean.columns)
        missing_model_input_columns = required_model_input_columns - set(df_model_input.columns)

        if missing_clean_columns:
            raise ValueError(
                {
                    "stage": "prepare_clean_validation_data",
                    "error": "missing_required_clean_columns",
                    "missing_columns": sorted(missing_clean_columns),
                }
            )

        if missing_model_input_columns:
            raise ValueError(
                {
                    "stage": "prepare_clean_validation_data",
                    "error": "missing_required_model_input_columns",
                    "missing_columns": sorted(missing_model_input_columns),
                }
            )

        if df_clean[row_identifier_column].duplicated().any():
            raise ValueError(
                {
                    "stage": "prepare_clean_validation_data",
                    "error": "duplicate_row_ids_in_clean_data",
                    "row_identifier_column": row_identifier_column,
                }
            )

        if df_model_input[row_identifier_column].duplicated().any():
            raise ValueError(
                {
                    "stage": "prepare_clean_validation_data",
                    "error": "duplicate_row_ids_in_model_input",
                    "row_identifier_column": row_identifier_column,
                }
            )

        df_clean_local = df_clean.copy()
        df_model_input_local = df_model_input.copy()

        log_config.emit_log(
            log,
            {
                "stage": "clean_validation_subset_started",
                "clean_rows": df_clean_local.shape[0],
                "model_input_rows": df_model_input_local.shape[0],
            },
        )

        modeled_row_ids = set(df_model_input_local[row_identifier_column])

        df_clean_validation = df_clean_local[
            df_clean_local[row_identifier_column].isin(modeled_row_ids)
        ].copy()

        if target_column not in df_clean_validation.columns:
            if "loan_status" not in df_clean_validation.columns:
                raise ValueError(
                    {
                        "stage": "prepare_clean_validation_data",
                        "error": "missing_target_and_loan_status",
                        "target_column": target_column,
                    }
                )

            status_to_target_mapping = {
                "fully_paid": 0,
                "charged_off": 1,
                "default": 1,
                "does_not_meet_the_credit_policy._status:fully_paid": 0,
                "does_not_meet_the_credit_policy._status:charged_off": 1,
            }

            unmapped_statuses = sorted(
                set(df_clean_validation["loan_status"].dropna().unique())
                - set(status_to_target_mapping.keys())
            )

            if unmapped_statuses:
                raise ValueError(
                    {
                        "stage": "prepare_clean_validation_data",
                        "error": "unmapped_loan_status_values",
                        "unmapped_statuses": unmapped_statuses,
                    }
                )

            df_clean_validation[target_column] = (
                df_clean_validation["loan_status"]
                .map(status_to_target_mapping)
                .astype("int8")
            )

        if df_clean_validation[target_column].isna().any():
            raise ValueError(
                {
                    "stage": "prepare_clean_validation_data",
                    "error": "missing_target_values_after_construction",
                    "missing_target_rows": int(df_clean_validation[target_column].isna().sum()),
                }
            )

        if df_clean_validation[row_identifier_column].duplicated().any():
            raise ValueError(
                {
                    "stage": "prepare_clean_validation_data",
                    "error": "duplicate_row_ids_after_subsetting",
                    "row_identifier_column": row_identifier_column,
                }
            )

        if df_clean_validation.shape[0] != df_model_input_local.shape[0]:
            clean_validation_row_ids = set(df_clean_validation[row_identifier_column])

            missing_in_clean_validation = sorted(modeled_row_ids - clean_validation_row_ids)
            extra_in_clean_validation = sorted(clean_validation_row_ids - modeled_row_ids)

            raise ValueError(
                {
                    "stage": "prepare_clean_validation_data",
                    "error": "row_count_mismatch_after_clean_subset",
                    "model_input_rows": df_model_input_local.shape[0],
                    "clean_validation_rows": df_clean_validation.shape[0],
                    "missing_row_ids_count": len(missing_in_clean_validation),
                    "extra_row_ids_count": len(extra_in_clean_validation),
                }
            )

        log_config.emit_log(
            log,
            {
                "stage": "prepare_clean_validation_data_complete",
                "validation_rows": df_clean_validation.shape[0],
                "validation_columns": df_clean_validation.shape[1],
                "target_column": target_column,
            },
        )

        return df_clean_validation

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "prepare_clean_validation_data_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def build_validation_dataset(
    df_model_input: pd.DataFrame,
    df_clean: pd.DataFrame,
    model: Any,
    target_column: str,
    row_identifier_column: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a validation dataset by scoring the selected model on the engineered
    model-input dataset and aligning predicted probabilities with the clean dataset.

    Returns a dataframe containing:
    - row_id
    - predicted_default_probability
    - target_default
    - LendingClub grade
    - contextual variables (e.g., loan_amnt)
    """

    try:
        log_config.emit_log(log, "Starting validation dataset build")

        required_model_input_columns = {target_column, row_identifier_column}
        required_clean_columns = {target_column, row_identifier_column}

        missing_model_input_columns = required_model_input_columns - set(df_model_input.columns)
        missing_clean_columns = required_clean_columns - set(df_clean.columns)

        if missing_model_input_columns:
            raise ValueError(
                {
                    "stage": "build_validation_dataset",
                    "error": "missing_required_model_input_columns",
                    "missing_columns": sorted(missing_model_input_columns),
                }
            )

        if missing_clean_columns:
            raise ValueError(
                {
                    "stage": "build_validation_dataset",
                    "error": "missing_required_clean_columns",
                    "missing_columns": sorted(missing_clean_columns),
                }
            )

        if df_model_input[row_identifier_column].duplicated().any():
            raise ValueError(
                {
                    "stage": "build_validation_dataset",
                    "error": "duplicate_row_ids_in_model_input",
                    "row_identifier_column": row_identifier_column,
                }
            )

        if df_clean[row_identifier_column].duplicated().any():
            raise ValueError(
                {
                    "stage": "build_validation_dataset",
                    "error": "duplicate_row_ids_in_clean_data",
                    "row_identifier_column": row_identifier_column,
                }
            )

        if not hasattr(model, "predict_proba"):
            raise ValueError(
                {
                    "stage": "build_validation_dataset",
                    "error": "model_missing_predict_proba",
                    "model_type": type(model).__name__,
                }
            )

        log_config.emit_log(log, "Preparing model feature matrix")

        df_model_input_local = df_model_input.copy()
        df_clean_local = df_clean.copy()

        df_features = df_model_input_local.drop(
            columns=[target_column, row_identifier_column]
        ).copy()

        log_config.emit_log(
            log,
            {
                "stage": "validation_feature_matrix_ready",
                "rows": df_model_input_local.shape[0],
                "features": df_features.shape[1],
            },
        )

        log_config.emit_log(log, "Generating predicted probabilities")

        predicted_probabilities = model.predict_proba(df_features)[:, 1]

        df_predictions = pd.DataFrame(
            {
                row_identifier_column: df_model_input_local[row_identifier_column].copy(),
                "predicted_default_probability": predicted_probabilities,
            }
        )

        log_config.emit_log(log, "Merging predictions with clean dataset")

        df_validation = df_clean_local.merge(
            df_predictions,
            on=row_identifier_column,
            how="inner",
            validate="one_to_one",
        )

        if df_validation.shape[0] != df_model_input_local.shape[0]:
            raise ValueError(
                {
                    "stage": "build_validation_dataset",
                    "error": "row_count_mismatch_after_merge",
                    "model_input_rows": df_model_input_local.shape[0],
                    "clean_rows": df_clean_local.shape[0],
                    "validation_rows": df_validation.shape[0],
                }
            )

        if df_validation[row_identifier_column].duplicated().any():
            raise ValueError(
                {
                    "stage": "build_validation_dataset",
                    "error": "duplicate_row_ids_after_merge",
                    "row_identifier_column": row_identifier_column,
                }
            )

        log_config.emit_log(
            log,
            {
                "stage": "build_validation_dataset_complete",
                "validation_rows": df_validation.shape[0],
                "validation_columns": df_validation.shape[1],
            },
        )

        return df_validation

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "build_validation_dataset_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def summarize_grade_structure(
    df: pd.DataFrame,
    grade_column: str,
    subgrade_column: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize grade structure and subgrade distribution.

    Returns:
    - grade_summary
    - subgrade_distribution
    """

    try:
        log_config.emit_log(log, "Summarizing grade structure")

        required_columns = {grade_column, subgrade_column}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                {
                    "stage": "summarize_grade_structure",
                    "error": "missing_required_columns",
                    "missing_columns": sorted(missing_columns),
                }
            )

        df_local = df.copy()

        # ------------------------------
        # Grade-level summary
        # ------------------------------
        grade_summary = (
            df_local.groupby(grade_column)
            .agg(
                loan_count=(grade_column, "count"),
                distinct_subgrades=(subgrade_column, lambda x: x.nunique(dropna=True)),
                missing_subgrade_count=(subgrade_column, lambda x: x.isna().sum()),
            )
            .reset_index()
        )

        grade_summary["missing_subgrade_rate"] = (
            grade_summary["missing_subgrade_count"] / grade_summary["loan_count"]
        )

        # ------------------------------
        # Subgrade distribution
        # ------------------------------
        subgrade_distribution = (
            df_local.groupby([grade_column, subgrade_column])
            .size()
            .reset_index(name="loan_count")
            .sort_values([grade_column, subgrade_column])
        )

        log_config.emit_log(
            log,
            {
                "stage": "summarize_grade_structure_complete",
                "grade_rows": grade_summary.shape[0],
                "subgrade_rows": subgrade_distribution.shape[0],
            },
        )

        return grade_summary, subgrade_distribution

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "summarize_grade_structure_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def compute_default_rate_by_subgrade(
    df_validation: pd.DataFrame,
    target_column: str,
    subgrade_column: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Compute loan counts, default counts, and default rates by subgrade.

    Parameters
    ----------
    df_validation : pd.DataFrame
        Validation dataset containing observed outcomes and subgrade labels.
    target_column : str
        Name of the binary target column.
    subgrade_column : str
        Name of the subgrade column.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by ``log_config.emit_log``.

    Returns
    -------
    pd.DataFrame
        Table with one row per subgrade containing loan count, default count,
        non-default count, and default rate.

    Raises
    ------
    ValueError
        If required columns are missing, the target contains invalid values,
        or subgrade values are missing.
    """
    try:
        log_config.emit_log(log, "Computing default rate by subgrade")

        required_columns = {target_column, subgrade_column}
        missing_columns = required_columns - set(df_validation.columns)

        if missing_columns:
            raise ValueError(
                {
                    "stage": "compute_default_rate_by_subgrade",
                    "error": "missing_required_columns",
                    "missing_columns": sorted(missing_columns),
                }
            )

        df_validation_local = df_validation.copy()

        if df_validation_local[subgrade_column].isna().any():
            raise ValueError(
                {
                    "stage": "compute_default_rate_by_subgrade",
                    "error": "missing_subgrade_values",
                    "missing_rows": int(df_validation_local[subgrade_column].isna().sum()),
                }
            )

        target_values = set(df_validation_local[target_column].dropna().unique().tolist())
        valid_target_values = {0, 1}

        if not target_values.issubset(valid_target_values):
            raise ValueError(
                {
                    "stage": "compute_default_rate_by_subgrade",
                    "error": "invalid_target_values",
                    "observed_values": sorted(target_values),
                }
            )

        df_default_rate_by_subgrade = (
            df_validation_local.groupby(subgrade_column)
            .agg(
                loan_count=(target_column, "size"),
                default_count=(target_column, "sum"),
            )
            .reset_index()
        )

        df_default_rate_by_subgrade["non_default_count"] = (
            df_default_rate_by_subgrade["loan_count"]
            - df_default_rate_by_subgrade["default_count"]
        )

        df_default_rate_by_subgrade["default_rate"] = (
            df_default_rate_by_subgrade["default_count"]
            / df_default_rate_by_subgrade["loan_count"]
        )

        log_config.emit_log(
            log,
            {
                "stage": "compute_default_rate_by_subgrade_complete",
                "rows": df_default_rate_by_subgrade.shape[0],
                "columns": df_default_rate_by_subgrade.shape[1],
            },
        )

        return df_default_rate_by_subgrade

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "compute_default_rate_by_subgrade_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise