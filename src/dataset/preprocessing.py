from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from config.logging import get_logger


# =============================================================================
# Inspection & Audit
# =============================================================================


def initial_inspection(
    df: pd.DataFrame, 
    log_file: str | None = None
) -> dict[str, Any]:
    """
    Perform an initial inspection of a DataFrame.

    Logs key dataset information and returns a structured summary suitable for notebook display.

    Summary includes:
    - shape and memory usage
    - numeric and object columns
    - missing values (all columns)
    - constant columns (excluding fully-null columns)
    - mixed type columns
    - numeric columns stored as objects
    - consolidated feature summary table
    """
    log = get_logger(log_file)

    try:
        if df is None:
            raise ValueError("df must not be None")

        log("===== Starting initial data inspection =====")

        # Shape and memory
        row_count, column_count = df.shape
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024**2)
        log(f"Shape: {row_count} rows x {column_count} columns | Memory: {memory_usage_mb:.2f} MB")

        # Column types
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        object_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
        log(f"Numeric columns: {len(numeric_columns)} | Object/string columns: {len(object_columns)}")

        # Missing values
        null_counts = df.isna().sum()
        null_percent = (null_counts / len(df)) * 100

        missing_values_df = (
            pd.DataFrame({"null_count": null_counts, "null_percent": null_percent.round(2)})
            .sort_values(by="null_percent", ascending=False)
        )

        log("\nColumns with missing values (% missing):")
        for column_name, missing_row in missing_values_df.iterrows():
            log(f"{column_name:<30} {missing_row['null_percent']:.2f}%")

        # Fully-null columns
        fully_null_columns = missing_values_df.index[missing_values_df["null_percent"] == 100].tolist()

        # Constant columns (exclude fully-null)
        constant_columns_all = df.columns[df.nunique(dropna=False) <= 1].tolist()
        constant_columns = [
            column_name for column_name in constant_columns_all if column_name not in fully_null_columns
        ]

        if constant_columns:
            log("\nConstant columns (non-null):")
            log(", ".join(constant_columns))

        # Mixed type columns
        mixed_type_columns: list[str] = []
        for column_name in df.columns:
            if df[column_name].dropna().map(type).nunique() > 1:
                mixed_type_columns.append(column_name)

        if mixed_type_columns:
            log("\nColumns with mixed types:")
            log(", ".join(mixed_type_columns))

        # Object columns that are numeric (ignore nulls)
        numeric_object_columns: list[str] = []
        for column_name in object_columns:
            non_null_values = df[column_name].dropna()
            if non_null_values.empty:
                continue
            coerced_values = pd.to_numeric(non_null_values, errors="coerce")
            if coerced_values.notna().all():
                numeric_object_columns.append(column_name)

        if numeric_object_columns:
            log("\nColumns stored as object but numeric:")
            log(", ".join(numeric_object_columns))

        # Consolidated Feature Summary Table
        feature_summary = pd.DataFrame(index=df.columns)
        feature_summary["dtype"] = df.dtypes
        feature_summary["n_unique"] = df.nunique(dropna=False)
        feature_summary["null_percent"] = missing_values_df["null_percent"]
        feature_summary["is_numeric"] = feature_summary.index.isin(numeric_columns)
        feature_summary["is_object"] = feature_summary.index.isin(object_columns)
        feature_summary["is_mixed_type"] = feature_summary.index.isin(mixed_type_columns)
        feature_summary["is_numeric_object"] = feature_summary.index.isin(numeric_object_columns)
        feature_summary["is_fully_null"] = feature_summary["null_percent"] == 100
        feature_summary["is_constant"] = (feature_summary["n_unique"] <= 1) & (~feature_summary["is_fully_null"])

        log("\nFeature structure summary created.")
        log(f"Total columns: {len(feature_summary)}")
        log(f"Fully null columns: {int(feature_summary['is_fully_null'].sum())}")
        log(f"Constant columns (non-null): {int(feature_summary['is_constant'].sum())}")
        log(f"Mixed-type columns: {int(feature_summary['is_mixed_type'].sum())}")
        log(f"Numeric stored as object: {int(feature_summary['is_numeric_object'].sum())}")

        log("===== Initial data inspection completed successfully =====")

        return {
            "shape": df.shape,
            "memory_usage_mb": memory_usage_mb,
            "numeric_columns": numeric_columns,
            "object_columns": object_columns,
            "missing_values": missing_values_df,
            "constant_columns": constant_columns,
            "mixed_type_columns": mixed_type_columns,
            "numeric_object_columns": numeric_object_columns,
            "feature_summary": feature_summary,
        }

    except Exception as exc:
        log(f"Error during initial data inspection: {exc}")
        raise


def audit_string_columns(
    df: pd.DataFrame,
    sample_size: int = 5,
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Create an audit table for string-like columns.

    Audited dtypes:
    - object
    - string
    - category
    """
    log = get_logger(log_file)

    try:
        if df is None:
            raise ValueError("df must not be None")

        if sample_size <= 0:
            raise ValueError("sample_size must be > 0")

        audited_dtypes = ["object", "string", "category"]
        string_like_columns = df.select_dtypes(include=audited_dtypes).columns.tolist()

        if not string_like_columns:
            log("[audit_string_columns] no string-like columns found")
            return pd.DataFrame(
                columns=[
                    "column_name",
                    "dtype",
                    "unique_count_including_null",
                    "unique_count_non_null",
                    "null_percent",
                    "sample_values",
                ]
            )

        records: list[dict[str, Any]] = []

        for column_name in string_like_columns:
            series = df[column_name]

            unique_count_including_null = int(series.nunique(dropna=False))
            unique_count_non_null = int(series.nunique(dropna=True))
            null_percent = float((series.isna().mean() * 100).round(2))

            sample_values = (
                series.dropna()
                .astype("string")
                .str.strip()
                .drop_duplicates()
                .head(sample_size)
                .tolist()
            )

            records.append(
                {
                    "column_name": column_name,
                    "dtype": str(series.dtype),
                    "unique_count_including_null": unique_count_including_null,
                    "unique_count_non_null": unique_count_non_null,
                    "null_percent": null_percent,
                    "sample_values": sample_values,
                }
            )

        audit_dataframe = (
            pd.DataFrame(records)
            .sort_values(
                by=["unique_count_non_null", "null_percent"],
                ascending=[False, False],
            )
            .reset_index(drop=True)
        )

        log(
            "[audit_string_columns] done | "
            f"audited_cols={int(audit_dataframe.shape[0])} "
            f"dtypes={audited_dtypes}"
        )

        return audit_dataframe

    except Exception as exception:
        log(
            f"[audit_string_columns][error] "
            f"Error={type(exception).__name__}: {exception}"
        )
        raise


def compare_categorical_column_values(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    column_name: str,
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Compare categorical values between training and test datasets.

    Returns a DataFrame containing:
    - value
    - whether present in training
    - whether present in test

    Intended for schema validation and category alignment prior to encoding.
    """
    log = get_logger(log_file)

    try:
        if df_train is None or df_test is None:
            raise ValueError("df_train and df_test must not be None")

        if not column_name or not str(column_name).strip():
            raise ValueError("column_name must be a non-empty string")

        if column_name not in df_train.columns:
            raise ValueError(f"Column not found in training dataset: {column_name}")

        if column_name not in df_test.columns:
            raise ValueError(f"Column not found in test dataset: {column_name}")

        log(f"[categorical_compare] start | column='{column_name}'")

        training_values = set(
            df_train[column_name].dropna().astype("string").str.strip().unique()
        )
        test_values = set(
            df_test[column_name].dropna().astype("string").str.strip().unique()
        )

        all_values = sorted(training_values.union(test_values))

        comparison_dataframe = pd.DataFrame(
            {
                "value": all_values,
                "present_in_training": [value in training_values for value in all_values],
                "present_in_test": [value in test_values for value in all_values],
            }
        )

        log(
            f"[categorical_compare] done | column='{column_name}' | "
            f"training_unique={len(training_values)} | test_unique={len(test_values)} | "
            f"combined_unique={len(all_values)}"
        )

        return comparison_dataframe

    except Exception as exc:
        log(f"[categorical_compare] failed | column='{column_name}' | error={exc}")
        raise


def build_combined_schema(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    stage_name: str,
    log: Callable[[str], None],
) -> pd.DataFrame:
    """
    Build a combined schema comparison between train and test datasets.

    Returns a DataFrame containing:
    - column_name
    - train_dtype
    - test_dtype
    - present_in_train
    - present_in_test
    - dtype_mismatch

    Logs structural deltas.
    """
    try:
        if df_train is None or df_test is None:
            raise ValueError("df_train and df_test must not be None")

        if not stage_name or not str(stage_name).strip():
            raise ValueError("stage_name must be a non-empty string")

        train_schema = (
            df_train.dtypes.rename("train_dtype").reset_index().rename(columns={"index": "column_name"})
        )

        test_schema = (
            df_test.dtypes.rename("test_dtype").reset_index().rename(columns={"index": "column_name"})
        )

        combined_schema = pd.merge(train_schema, test_schema, on="column_name", how="outer")

        combined_schema["present_in_train"] = combined_schema["train_dtype"].notna()
        combined_schema["present_in_test"] = combined_schema["test_dtype"].notna()
        combined_schema["dtype_mismatch"] = (
            combined_schema["present_in_train"]
            & combined_schema["present_in_test"]
            & (combined_schema["train_dtype"] != combined_schema["test_dtype"])
        )

        combined_schema = combined_schema.sort_values("column_name").reset_index(drop=True)

        train_only_columns = combined_schema.loc[
            combined_schema["present_in_train"] & ~combined_schema["present_in_test"],
            "column_name",
        ].tolist()

        test_only_columns = combined_schema.loc[
            ~combined_schema["present_in_train"] & combined_schema["present_in_test"],
            "column_name",
        ].tolist()

        dtype_mismatch_records = combined_schema.loc[
            combined_schema["dtype_mismatch"],
            ["column_name", "train_dtype", "test_dtype"],
        ].to_dict(orient="records")

        log(f"[{stage_name}][schema] train_only_count={len(train_only_columns)}")
        log(f"[{stage_name}][schema] test_only_count={len(test_only_columns)}")
        log(f"[{stage_name}][schema] dtype_mismatch_count={len(dtype_mismatch_records)}")

        if train_only_columns:
            log(f"[{stage_name}][schema] train_only_columns={train_only_columns}")

        if test_only_columns:
            log(f"[{stage_name}][schema] test_only_columns={test_only_columns}")

        if dtype_mismatch_records:
            log(f"[{stage_name}][schema] dtype_mismatches={dtype_mismatch_records}")

        return combined_schema

    except Exception as exc:
        log(f"[{stage_name}][schema] Failed to build combined schema: {exc}")
        raise


def build_structural_issues_report(
    combined_feature_universe_df: pd.DataFrame,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Build a structural issues report from a combined train/test feature universe table.

    Expected columns in combined_feature_universe_df:
      - column_name
      - present_in_train, present_in_test
      - dtype_mismatch
      - train_is_fully_null, test_is_fully_null
      - train_null_percent, test_null_percent
      - train_is_constant, test_is_constant
    """
    try:
        if combined_feature_universe_df is None:
            raise ValueError("combined_feature_universe_df must not be None")

        required_columns = [
            "column_name",
            "present_in_train",
            "present_in_test",
            "dtype_mismatch",
            "train_is_fully_null",
            "test_is_fully_null",
            "train_null_percent",
            "test_null_percent",
            "train_is_constant",
            "test_is_constant",
        ]
        missing = [c for c in required_columns if c not in combined_feature_universe_df.columns]
        if missing:
            raise ValueError(f"Missing required columns for structural issues report: {missing}")

        issues_records: list[dict[str, object]] = []

        for _, row in combined_feature_universe_df.iterrows():
            column_name = row["column_name"]

            present_in_train = bool(row["present_in_train"])
            present_in_test = bool(row["present_in_test"])

            if present_in_train and not present_in_test:
                issues_records.append(
                    {"column_name": column_name, "issue": "Present in train only", "applies_to": "train"}
                )

            if present_in_test and not present_in_train:
                issues_records.append(
                    {"column_name": column_name, "issue": "Present in test only", "applies_to": "test"}
                )

            if present_in_train and present_in_test and bool(row["dtype_mismatch"]):
                issues_records.append(
                    {"column_name": column_name, "issue": "Dtype mismatch", "applies_to": "both"}
                )

            if present_in_train:
                if bool(row["train_is_fully_null"]):
                    issues_records.append({"column_name": column_name, "issue": "100% Null", "applies_to": "train"})
                elif float(row["train_null_percent"]) >= 50:
                    issues_records.append(
                        {"column_name": column_name, "issue": "High Missing (>50%)", "applies_to": "train"}
                    )

                if bool(row["train_is_constant"]):
                    issues_records.append({"column_name": column_name, "issue": "Constant", "applies_to": "train"})

            if present_in_test:
                if bool(row["test_is_fully_null"]):
                    issues_records.append({"column_name": column_name, "issue": "100% Null", "applies_to": "test"})
                elif float(row["test_null_percent"]) >= 50:
                    issues_records.append(
                        {"column_name": column_name, "issue": "High Missing (>50%)", "applies_to": "test"}
                    )

                if bool(row["test_is_constant"]):
                    issues_records.append({"column_name": column_name, "issue": "Constant", "applies_to": "test"})

        issues_df = pd.DataFrame(issues_records)
        if issues_df.empty:
            issues_df = pd.DataFrame(columns=["column_name", "issue", "applies_to"])

        severity_order = [
            "Present in train only",
            "Present in test only",
            "Dtype mismatch",
            "100% Null",
            "High Missing (>50%)",
            "Constant",
        ]

        issues_df["issue"] = pd.Categorical(issues_df["issue"], categories=severity_order, ordered=True)
        issues_df = issues_df.sort_values(
            ["issue", "column_name", "applies_to"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        if log is not None:
            log(f"[structural_issues] rows={issues_df.shape[0]}")

        return issues_df

    except Exception as exc:
        if log is not None:
            log(f"[structural_issues] failed: {exc}")
        raise


# =============================================================================
# Transformations
# =============================================================================


def create_row_identifier(
    df: pd.DataFrame,
    id_column_name: str = "row_id",
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Create a stable internal row identifier for the dataset.

    If the column already exists, it will not be recreated.
    """
    log = get_logger(log_file)

    try:
        if df is None:
            raise ValueError("df must not be None")

        if id_column_name in df.columns:
            log(f"Row identifier already exists: {id_column_name}")
            return df

        transformed_dataframe = df.copy()
        transformed_dataframe.insert(0, id_column_name, range(1, len(transformed_dataframe) + 1))

        log(f"Row identifier created: {id_column_name} | Total rows: {len(transformed_dataframe)}")
        return transformed_dataframe

    except Exception as exc:
        log(f"Error while creating row identifier: {exc}")
        raise


def _format_column_list(
    columns: list[str], 
    max_items: int = 25
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
    log_file: str | None = None,
    errors: str = "ignore",
) -> pd.DataFrame:
    """
    Drop columns from a DataFrame and log what happened.
    """
    log = get_logger(log_file)

    try:
        if df is None:
            raise ValueError("df must not be None")

        if not dataset_name or not str(dataset_name).strip():
            raise ValueError("dataset_name must be a non-empty string")

        shape_before = df.shape

        existing_columns = set(df.columns)
        requested_columns = list(dict.fromkeys(columns_to_drop))  # de-duplicate, preserve order

        dropped_columns = [
            column_name for column_name in requested_columns if column_name in existing_columns
        ]

        missing_columns = [
            column_name for column_name in requested_columns if column_name not in existing_columns
        ]

        transformed_dataframe = df.drop(columns=requested_columns, errors=errors)

        shape_after = transformed_dataframe.shape

        log(f"[drop_columns] dataset={dataset_name}")
        log(f"[drop_columns] shape_before={shape_before} shape_after={shape_after}")

        log(
            f"[drop_columns] requested={len(requested_columns)} "
            f"dropped={len(dropped_columns)} missing={len(missing_columns)} errors={errors}"
        )

        if dropped_columns:
            log("[drop_columns] dropped_columns=" + _format_column_list(dropped_columns))

        if missing_columns:
            log("[drop_columns] missing_columns=" + _format_column_list(missing_columns))

        return transformed_dataframe

    except Exception as exc:
        log(
            f"[drop_columns_with_logging][error] dataset={dataset_name} "
            f"Error={type(exc).__name__}: {exc}"
        )
        raise


def apply_credit_sentinel_encoding(
    df: pd.DataFrame,
    credit_recency_columns: list[str],
    sentinel_value: int = 9999,
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Apply sentinel encoding to credit recency variables.
    - Adds indicator columns: has_<col> (int8)
    - Fills missing values in <col> with sentinel_value
    """
    log = get_logger(log_file)

    try:
        if df is None:
            raise ValueError("df must not be None")

        transformed_dataframe = df.copy()
        log("[credit_sentinel_encoding] start")

        missing_configured_columns: list[str] = []

        for column_name in credit_recency_columns:
            if column_name not in transformed_dataframe.columns:
                missing_configured_columns.append(column_name)
                continue

            indicator_column_name = f"has_{column_name}"
            missing_value_count = int(transformed_dataframe[column_name].isna().sum())

            transformed_dataframe[indicator_column_name] = (
                transformed_dataframe[column_name].notna().astype("int8")
            )
            transformed_dataframe[column_name] = transformed_dataframe[column_name].fillna(sentinel_value)

            log(
                f"[credit_sentinel_encoding] column={column_name} "
                f"missing={missing_value_count} sentinel={sentinel_value} "
                f"indicator={indicator_column_name}"
            )

        if missing_configured_columns:
            log(
                f"[credit_sentinel_encoding][error] Missing configured columns: {missing_configured_columns}"
            )
            raise KeyError(f"Missing configured credit recency columns: {missing_configured_columns}")

        log("[credit_sentinel_encoding] completed")
        return transformed_dataframe

    except Exception as exc:
        log(
            f"[credit_sentinel_encoding][error] "
            f"Error={type(exc).__name__}: {exc}"
        )
        raise


def convert_month_year_columns_to_datetime(
    df: pd.DataFrame,
    column_names: list[str],
    datetime_format: str = "%b-%y",
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Convert month-year string columns (e.g. 'Jan-16') to pandas datetime.
    """
    log = get_logger(log_file)

    try:
        if df is None:
            raise ValueError("df must not be None")

        transformed_dataframe = df.copy()

        requested_columns = list(dict.fromkeys(column_names))
        log(f"Starting month-year to datetime conversion for {len(requested_columns)} columns")

        for column_name in requested_columns:
            if column_name not in transformed_dataframe.columns:
                log(f"Column not found (skipped): {column_name}")
                continue

            series_before = transformed_dataframe[column_name]
            normalized_series = series_before.astype("string").str.strip()
            non_null_mask = normalized_series.notna()

            parsed_series = pd.to_datetime(
                normalized_series[non_null_mask],
                format=datetime_format,
                errors="raise",
            )

            datetime_series = pd.Series(pd.NaT, index=transformed_dataframe.index, dtype="datetime64[ns]")
            datetime_series.loc[non_null_mask] = parsed_series.to_numpy()

            transformed_dataframe[column_name] = datetime_series.to_numpy()
            log(f"Converted to datetime: {column_name} | dtype={transformed_dataframe[column_name].dtype}")

        log("Month-year datetime conversion completed successfully")
        return transformed_dataframe

    except Exception as exc:
        log(f"Error in convert_month_year_columns_to_datetime: {exc}")
        raise


def transform_emp_length(
    df: pd.DataFrame,
    column_name: str = "emp_length",
    output_column_name: str = "emp_length_years",
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Normalize LendingClub-style employment length strings into numeric years.

    Policy
    ------
    - Known missing tokens (e.g. "n/a") are treated as missing values.
    - Any other unexpected non-null values raise an error.
    """

    log = get_logger(log_file)

    try:
        if df is None:
            raise ValueError("df must not be None")

        if column_name not in df.columns:
            log(f"[transform_emp_length] column not found (skipped): {column_name}")
            return df

        transformed_dataframe = df.copy()

        raw_series = transformed_dataframe[column_name]

        normalized_series = (
            raw_series
            .astype("string")
            .str.strip()
            .str.lower()
        )

        # Define known missing value tokens and replace them with pd.NA
        missing_tokens = {"n/a", "na", "n.a."}
        normalized_series = normalized_series.replace(list(missing_tokens), pd.NA)

        employment_length_mapping: dict[str, int] = {
            "< 1 year": 0,
            "10+ years": 10,
        }

        # Ensure mapping keys match normalized casing
        employment_length_mapping = {
            mapping_key.lower(): mapping_value
            for mapping_key, mapping_value in employment_length_mapping.items()
        }

        for year_value in range(1, 10):
            employment_length_mapping[f"{year_value} year"] = year_value
            employment_length_mapping[f"{year_value} years"] = year_value

        non_null_values = normalized_series.dropna()

        unexpected_values = sorted(
            set(non_null_values.unique()) - set(employment_length_mapping.keys())
        )

        if unexpected_values:
            log(
                "[transform_emp_length][error] unexpected non-null values detected: "
                + ", ".join(unexpected_values[:25])
                + (" ..." if len(unexpected_values) > 25 else "")
            )

            raise ValueError(
                f"Unexpected values in '{column_name}': {unexpected_values[:10]}"
                + (" (and more)" if len(unexpected_values) > 10 else "")
            )

        mapped_series = normalized_series.map(employment_length_mapping)

        transformed_dataframe[output_column_name] = mapped_series.astype("Float32")

        total_rows = int(len(transformed_dataframe))
        input_null_count = int(raw_series.isna().sum())
        normalized_null_count = int(normalized_series.isna().sum())
        output_null_count = int(transformed_dataframe[output_column_name].isna().sum())

        log(
            f"[transform_emp_length] created '{output_column_name}' from '{column_name}' | "
            f"rows={total_rows} | "
            f"input_nulls={input_null_count} | "
            f"normalized_nulls={normalized_null_count} | "
            f"output_nulls={output_null_count}"
        )

        return transformed_dataframe

    except Exception as exception:
        log(
            f"[transform_emp_length][error] "
            f"Error={type(exception).__name__}: {exception}"
        )
        raise


def normalize_home_ownership(
    df: pd.DataFrame,
    column_name: str = "home_ownership",
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Normalize home_ownership so train/test have a stable category space.
    """
    log = get_logger(log_file)

    try:
        if df is None:
            raise ValueError("df must not be None")

        if column_name not in df.columns:
            log(f"normalize_home_ownership: column not found (skipped): {column_name}")
            return df

        transformed_dataframe = df.copy()

        series_before = transformed_dataframe[column_name]
        normalized_series = series_before.astype("string").str.strip().str.lower()

        mapping = {"none": "other", "any": "other"}
        normalized_series = normalized_series.replace(mapping)

        transformed_dataframe[column_name] = normalized_series

        unique_before = int(series_before.nunique(dropna=False))
        unique_after = int(transformed_dataframe[column_name].nunique(dropna=False))

        log(
            f"normalize_home_ownership: {column_name} | unique_before={unique_before} | "
            f"unique_after={unique_after} | mapped={list(mapping.keys())} -> 'other'"
        )

        return transformed_dataframe

    except Exception as exc:
        log(f"Error in normalize_home_ownership: {exc}")
        raise


def normalize_text_columns(
    df: pd.DataFrame,
    columns_to_normalize: Iterable[str],
    *,
    lowercase: bool = True,
    strip_whitespace: bool = True,
    collapse_whitespace: bool = True,
    replace_spaces_with_underscore: bool = True,
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Normalize categorical/text-like columns.
    """
    log: Callable[[str], None] = get_logger(log_file)

    try:
        if df is None:
            raise ValueError("df must not be None")

        normalized_dataframe = df.copy()

        requested_columns = list(dict.fromkeys(columns_to_normalize))
        existing_columns = [c for c in requested_columns if c in normalized_dataframe.columns]
        missing_columns = [c for c in requested_columns if c not in normalized_dataframe.columns]

        if missing_columns:
            log("normalize_text_columns: missing columns (skipped): " + ", ".join(missing_columns))

        if not existing_columns:
            log("normalize_text_columns: no matching columns to normalize.")
            return normalized_dataframe

        log(f"normalize_text_columns: normalizing {len(existing_columns)} columns")

        for column_name in existing_columns:
            series_before = normalized_dataframe[column_name]

            unique_before = int(series_before.nunique(dropna=True))
            null_percent = float((series_before.isna().mean() * 100).round(2))

            normalized_series = series_before.where(series_before.isna(), series_before.astype("string"))

            if strip_whitespace:
                normalized_series = normalized_series.str.strip()

            if collapse_whitespace:
                normalized_series = normalized_series.str.replace(r"\s+", " ", regex=True)

            if lowercase:
                normalized_series = normalized_series.str.lower()

            if replace_spaces_with_underscore:
                normalized_series = normalized_series.str.replace(" ", "_", regex=False)

            normalized_dataframe[column_name] = normalized_series
            unique_after = int(normalized_dataframe[column_name].nunique(dropna=True))

            log(
                f"Normalized '{column_name}': unique_non_null {unique_before} -> {unique_after} | "
                f"null_percent={null_percent:.2f}%"
            )

        return normalized_dataframe

    except Exception as exc:
        log(f"Error in normalize_text_columns: {exc}")
        raise


def apply_categorical_mapping(
    df: pd.DataFrame,
    column_name: str,
    mapping: Mapping[str, object],
    *,
    lowercase: bool = True,
    strip_whitespace: bool = True,
    output_column_name: str | None = None,
    allow_unmapped_values: bool = False,
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Apply a controlled mapping to a categorical column.
    """
    log = get_logger(log_file)

    try:
        if df is None:
            raise ValueError("df must not be None")

        if mapping is None:
            raise ValueError("mapping must not be None")

        if column_name not in df.columns:
            log(f"apply_categorical_mapping: column not found (skipped): {column_name}")
            return df

        transformed_dataframe = df.copy()

        normalized_series = transformed_dataframe[column_name].astype("string")

        if strip_whitespace:
            normalized_series = normalized_series.str.strip()

        if lowercase:
            normalized_series = normalized_series.str.lower()

        non_null_values = normalized_series.dropna().unique().tolist()
        unmapped_values = sorted(set(non_null_values) - set(mapping.keys()))

        if unmapped_values and not allow_unmapped_values:
            log(
                f"apply_categorical_mapping: unmapped values in '{column_name}': "
                + ", ".join(unmapped_values[:25])
                + (" ..." if len(unmapped_values) > 25 else "")
            )
            raise ValueError(f"Unmapped values in '{column_name}': {unmapped_values[:10]}")

        mapped_series = normalized_series.map(mapping)

        if allow_unmapped_values:
            mapped_series = mapped_series.where(normalized_series.isna() | mapped_series.notna(), normalized_series)

        target_column_name = output_column_name or column_name
        transformed_dataframe[target_column_name] = mapped_series

        log(
            f"apply_categorical_mapping: '{column_name}' -> '{target_column_name}' | "
            f"mapped_keys={len(mapping)} | unmapped_values={len(unmapped_values)} | "
            f"allow_unmapped_values={allow_unmapped_values}"
        )

        return transformed_dataframe

    except Exception as exc:
        log(f"Error in apply_categorical_mapping: {exc}")
        raise
   

def parse_term_column(
    df: pd.DataFrame,
    term_column: str,
    new_column_name: str,
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Parse LendingClub loan term (e.g. '36 months') into numeric months.
    """
    log = get_logger(log_file)

    try:
        if df is None:
            raise ValueError("df must not be None")

        transformed_dataframe = df.copy()

        log("[parse_term_column] parsing loan term")

        transformed_dataframe[new_column_name] = (
            transformed_dataframe[term_column]
            .astype("string")
            .str.strip()
            .str.extract(r"(\d+)")[0]
            .astype("Int16")
        )

        log(f"[parse_term_column] created column={new_column_name}")

        return transformed_dataframe

    except Exception as exc:
        log(
            f"[parse_term_column][error] "
            f"Error={type(exc).__name__}: {exc}"
        )
        raise


def cast_categorical_columns(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    column_names: list[str],
    log_file: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cast selected columns to pandas categorical dtype with a shared
    category space across train and test datasets.
    """
    log = get_logger(log_file)

    try:
        if df_train is None or df_test is None:
            raise ValueError("Input dataframes must not be None")

        train_df = df_train.copy()
        test_df = df_test.copy()

        log("[cast_categorical_columns] casting string -> category")

        for column_name in column_names:

            if column_name not in train_df.columns:
                log(f"Train missing categorical column (skipped): {column_name}")
                continue

            if column_name not in test_df.columns:
                log(f"Test missing categorical column (skipped): {column_name}")
                continue

            train_df[column_name] = train_df[column_name].astype("string")
            test_df[column_name] = test_df[column_name].astype("string")

            train_values = train_df[column_name].dropna().unique().tolist()
            test_values = test_df[column_name].dropna().unique().tolist()

            combined_categories = sorted(set(train_values) | set(test_values))

            train_df[column_name] = pd.Categorical(
                train_df[column_name],
                categories=combined_categories
            )

            test_df[column_name] = pd.Categorical(
                test_df[column_name],
                categories=combined_categories
            )

            log(
                f"[cast_categorical_columns] column={column_name} "
                f"train_unique={len(train_values)} "
                f"test_unique={len(test_values)} "
                f"combined={len(combined_categories)}"
            )

        return train_df, test_df

    except Exception as exc:
        log(
            f"[cast_categorical_columns][error] "
            f"Error={type(exc).__name__}: {exc}"
        )
        raise


# =============================================================================
#  Alignment Analysis
# =============================================================================


def _resolve_export_dir_and_suffix(
    export_dir: Path | str | None,
    export_tag: str | None,
    *,
    log: Callable[[str], None],
) -> tuple[Path | None, str]:
    try:
        export_dir_path: Path | None = None

        if export_dir is not None and str(export_dir).strip():
            export_dir_path = Path(export_dir)
            export_dir_path.mkdir(parents=True, exist_ok=True)

        export_suffix = f"_{export_tag}" if export_tag else ""

        if export_dir_path is None:
            log(f"[string_alignment][export] disabled | export_tag='{export_tag or ''}'")
        else:
            log(
                f"[string_alignment][export] enabled | export_dir='{export_dir_path}' | "
                f"export_suffix='{export_suffix}'"
            )

        return export_dir_path, export_suffix

    except Exception as exc:
        log(f"[string_alignment][export] failed to resolve export dir | error={exc}")
        raise


def _run_string_audits(
    *,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    sample_size: int,
    log_file: str | Path | None,
    log: Callable[[str], None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if df_train is None or df_test is None:
            raise ValueError("df_train and df_test must not be None")

        if sample_size <= 0:
            raise ValueError("sample_size must be > 0")

        log(
            "[string_alignment] building audits | "
            f"sample_size={sample_size} | train_shape={df_train.shape} | test_shape={df_test.shape}"
        )

        train_audit_df = audit_string_columns(
            df=df_train,
            sample_size=sample_size,
            log_file=str(log_file) if log_file is not None else None,
        )

        test_audit_df = audit_string_columns(
            df=df_test,
            sample_size=sample_size,
            log_file=str(log_file) if log_file is not None else None,
        )

        log(
            "[string_alignment] audits built | "
            f"train_string_cols={train_audit_df.shape[0]} | test_string_cols={test_audit_df.shape[0]}"
        )

        return train_audit_df, test_audit_df

    except Exception as exc:
        log(f"[string_alignment] failed building audits | error={exc}")
        raise


def _build_combined_string_alignment_table(
    df_train_audit: pd.DataFrame,
    df_test_audit: pd.DataFrame,
    *,
    log: Callable[[str], None],
) -> pd.DataFrame:
    try:
        required_columns = [
            "column_name",
            "dtype",
            "unique_count_including_null",
            "unique_count_non_null",
            "null_percent",
            "sample_values",
        ]

        missing_train = [
            column_name for column_name in required_columns 
            if column_name not in df_train_audit.columns
        ]
        
        missing_test = [
            column_name for column_name in required_columns 
            if column_name not in df_test_audit.columns
        ]

        if missing_train or missing_test:
            raise ValueError(
                "audit_string_columns output schema mismatch. "
                f"missing_train={missing_train} missing_test={missing_test}"
            )

        train_renamed_df = df_train_audit.copy().rename(
            columns={
                "dtype": "dtype_train",
                "unique_count_including_null": "unique_including_null_train",
                "unique_count_non_null": "unique_non_null_train",
                "null_percent": "null_percent_train",
                "sample_values": "sample_values_train",
            }
        )

        test_renamed_df = df_test_audit.copy().rename(
            columns={
                "dtype": "dtype_test",
                "unique_count_including_null": "unique_including_null_test",
                "unique_count_non_null": "unique_non_null_test",
                "null_percent": "null_percent_test",
                "sample_values": "sample_values_test",
            }
        )

        combined_df = (
            pd.merge(train_renamed_df, test_renamed_df, on="column_name", how="outer")
            .sort_values("column_name")
            .reset_index(drop=True)
        )

        combined_df["present_in_train"] = combined_df["dtype_train"].notna()
        combined_df["present_in_test"] = combined_df["dtype_test"].notna()

        for col in ["unique_non_null_train", "unique_non_null_test", "null_percent_train", "null_percent_test"]:
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

        combined_df["dtype_mismatch"] = (
            combined_df["present_in_train"]
            & combined_df["present_in_test"]
            & (combined_df["dtype_train"].astype("string") != combined_df["dtype_test"].astype("string"))
        )

        combined_df["unique_non_null_gap_test_minus_train"] = (
            combined_df["unique_non_null_test"].fillna(0) - combined_df["unique_non_null_train"].fillna(0)
        )

        combined_df["null_gap_test_minus_train"] = combined_df["null_percent_test"] - combined_df["null_percent_train"]
        combined_df["max_null_percent"] = combined_df[["null_percent_train", "null_percent_test"]].max(axis=1)

        combined_df["has_difference"] = (
            (combined_df["present_in_train"] ^ combined_df["present_in_test"])
            | combined_df["dtype_mismatch"]
            | (combined_df["unique_non_null_train"].fillna(-1) != combined_df["unique_non_null_test"].fillna(-1))
            | (
                combined_df["null_percent_train"].round(2).fillna(-1)
                != combined_df["null_percent_test"].round(2).fillna(-1)
            )
        )

        log(
            "[string_alignment] combined table built | "
            f"union_cols={combined_df.shape[0]} | diffs={int(combined_df['has_difference'].sum())}"
        )

        return combined_df

    except Exception as exc:
        log(f"[string_alignment] failed building combined alignment table | error={exc}")
        raise


def _rank_top_deltas(
    deltas_df: pd.DataFrame,
    *,
    top_k: int,
    log: Callable[[str], None],
) -> pd.DataFrame:
    try:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        if deltas_df is None or deltas_df.empty:
            return pd.DataFrame(columns=deltas_df.columns if deltas_df is not None else [])

        ranked_df = deltas_df.copy()

        ranked_df["severity_score"] = 0
        ranked_df.loc[(ranked_df["present_in_train"] ^ ranked_df["present_in_test"]), "severity_score"] += 100
        ranked_df.loc[ranked_df["dtype_mismatch"], "severity_score"] += 60
        ranked_df.loc[ranked_df["null_gap_test_minus_train"].abs() >= 5, "severity_score"] += 20
        ranked_df.loc[ranked_df["unique_non_null_gap_test_minus_train"].abs() >= 10, "severity_score"] += 10

        ranked_df["abs_null_gap"] = ranked_df["null_gap_test_minus_train"].abs()
        ranked_df["abs_unique_gap"] = ranked_df["unique_non_null_gap_test_minus_train"].abs()

        top_deltas_df = (
            ranked_df.sort_values(
                ["severity_score", "abs_null_gap", "abs_unique_gap", "column_name"],
                ascending=[False, False, False, True],
            )
            .head(top_k)
            .reset_index(drop=True)
        )

        log(f"[string_alignment] ranked top deltas | requested={top_k} | returned={top_deltas_df.shape[0]}")
        return top_deltas_df

    except Exception as exc:
        log(f"[string_alignment] failed ranking top deltas | error={exc}")
        raise


def _build_value_differences(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    top_deltas_df: pd.DataFrame,
    *,
    drilldown_max_columns: int,
    drilldown_top_values_per_side: int,
    log_file: str | Path | None,
) -> pd.DataFrame:
    log = get_logger(str(log_file) if log_file is not None else None)

    try:
        if drilldown_max_columns <= 0 or top_deltas_df is None or top_deltas_df.empty:
            return pd.DataFrame(columns=["column_name", "value", "present_in"])

        columns_to_drill = top_deltas_df["column_name"].dropna().astype(str).head(drilldown_max_columns).tolist()
        value_records: list[dict[str, Any]] = []

        for column_name in columns_to_drill:
            in_train = column_name in df_train.columns
            in_test = column_name in df_test.columns

            if in_train and in_test:
                diff_df = compare_categorical_column_values(
                    df_train=df_train,
                    df_test=df_test,
                    column_name=column_name,
                    log_file=str(log_file) if log_file is not None else None,
                )

                required = {"value", "present_in_training", "present_in_test"}
                if not required.issubset(set(diff_df.columns)):
                    log(
                        f"[string_alignment][drilldown] unexpected schema for '{column_name}' | "
                        f"required={sorted(required)} got={sorted(diff_df.columns.tolist())}"
                    )
                    continue

                train_only_values = (
                    diff_df[diff_df["present_in_test"] == False]["value"]
                    .head(drilldown_top_values_per_side)
                    .tolist()
                )
                test_only_values = (
                    diff_df[diff_df["present_in_training"] == False]["value"]
                    .head(drilldown_top_values_per_side)
                    .tolist()
                )

                for value in train_only_values:
                    value_records.append({"column_name": column_name, "value": value, "present_in": "train_only"})

                for value in test_only_values:
                    value_records.append({"column_name": column_name, "value": value, "present_in": "test_only"})

                continue

            if in_train and not in_test:
                log(f"[string_alignment][drilldown] '{column_name}' present in train only; sampling train values")

                train_values = (
                    df_train[column_name]
                    .dropna()
                    .astype("string")
                    .str.strip()
                    .drop_duplicates()
                    .head(drilldown_top_values_per_side)
                    .tolist()
                )

                for value in train_values:
                    value_records.append({"column_name": column_name, "value": value, "present_in": "train_only"})

                continue

            if in_test and not in_train:
                log(f"[string_alignment][drilldown] '{column_name}' present in test only; sampling test values")

                test_values = (
                    df_test[column_name]
                    .dropna()
                    .astype("string")
                    .str.strip()
                    .drop_duplicates()
                    .head(drilldown_top_values_per_side)
                    .tolist()
                )

                for value in test_values:
                    value_records.append({"column_name": column_name, "value": value, "present_in": "test_only"})

                continue

            log(f"[string_alignment][drilldown] skipped missing column in both: {column_name}")

        if not value_records:
            return pd.DataFrame(columns=["column_name", "value", "present_in"])

        value_differences_df = pd.DataFrame(value_records)
        log(f"[string_alignment][drilldown] done | rows={value_differences_df.shape[0]}")

        return value_differences_df

    except Exception as exc:
        log(f"[string_alignment][drilldown] failed | error={exc}")
        raise


def _serialize_sample_values_for_export(
    df: pd.DataFrame,
    *,
    columns_to_serialize: list[str],
    log: Callable[[str], None],
) -> pd.DataFrame:
    try:
        export_df = df.copy()

        def serialize_cell(cell_value: Any) -> Any:
            if isinstance(cell_value, list):
                return json.dumps(cell_value, ensure_ascii=False)
            return cell_value

        for column_name in columns_to_serialize:
            if column_name in export_df.columns:
                export_df[column_name] = export_df[column_name].apply(serialize_cell)

        return export_df

    except Exception as exc:
        log(f"[string_alignment][export] failed serializing sample values | error={exc}")
        raise


def _export_alignment_reports(
    *,
    summary_df: pd.DataFrame,
    deltas_df: pd.DataFrame,
    top_deltas_df: pd.DataFrame,
    value_differences_df: pd.DataFrame,
    export_dir_path: Path,
    export_base_name: str,
    export_suffix: str,
    export_sample_values_as_json: bool,
    log: Callable[[str], None],
) -> None:
    try:
        summary_path = export_dir_path / f"{export_base_name}{export_suffix}_summary.csv"
        deltas_path = export_dir_path / f"{export_base_name}{export_suffix}_deltas.csv"
        top_deltas_path = export_dir_path / f"{export_base_name}{export_suffix}_top_deltas.csv"

        deltas_export_df = deltas_df.copy()
        top_deltas_export_df = top_deltas_df.copy()

        if export_sample_values_as_json:
            deltas_export_df = _serialize_sample_values_for_export(
                deltas_export_df,
                columns_to_serialize=["sample_values_train", "sample_values_test"],
                log=log,
            )
            top_deltas_export_df = _serialize_sample_values_for_export(
                top_deltas_export_df,
                columns_to_serialize=["sample_values_train", "sample_values_test"],
                log=log,
            )

        summary_df.to_csv(summary_path, index=False, encoding="utf-8")
        deltas_export_df.to_csv(deltas_path, index=False, encoding="utf-8")
        top_deltas_export_df.to_csv(top_deltas_path, index=False, encoding="utf-8")

        values_path: Path | None = None
        if value_differences_df is not None and not value_differences_df.empty:
            values_path = export_dir_path / f"{export_base_name}{export_suffix}_value_differences.csv"
            value_differences_df.to_csv(values_path, index=False, encoding="utf-8")
            log(f"[string_alignment][export] values={values_path}")
        else:
            log("[string_alignment][export] values=skipped (empty)")

        log(
            "[string_alignment][export] done | "
            f"summary={summary_path} | deltas={deltas_path} | top={top_deltas_path}"
            + (f" | values={values_path}" if values_path is not None else "")
        )

    except Exception as exc:
        log(f"[string_alignment][export] failed | error={exc}")
        raise


def build_string_alignment_report(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    sample_size: int = 5,
    top_k: int = 10,
    drilldown_max_columns: int = 5,
    drilldown_top_values_per_side: int = 10,
    log_file: str | Path | None = None,
    export_dir: Path | str | None = None,
    export_base_name: str = "string_alignment",
    export_tag: str | None = None,
    export_sample_values_as_json: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Compare train vs test for string-like columns using audit_string_columns output.
    """
    log: Callable[[str], None] = get_logger(str(log_file) if log_file is not None else None)

    try:
        if df_train is None or df_test is None:
            raise ValueError("df_train and df_test must not be None")

        if sample_size <= 0:
            raise ValueError("sample_size must be > 0")

        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        if drilldown_max_columns < 0:
            raise ValueError("drilldown_max_columns must be >= 0")

        if drilldown_top_values_per_side <= 0:
            raise ValueError("drilldown_top_values_per_side must be > 0")

        export_dir_path, export_suffix = _resolve_export_dir_and_suffix(
            export_dir=export_dir,
            export_tag=export_tag,
            log=log,
        )

        log(
            "[string_alignment] start | "
            f"train_shape={df_train.shape} | test_shape={df_test.shape} | "
            f"sample_size={sample_size} | top_k={top_k} | drilldown_max_columns={drilldown_max_columns} | "
            f"export_suffix='{export_suffix}'"
        )

        df_train_audit, df_test_audit = _run_string_audits(
            df_train=df_train,
            df_test=df_test,
            sample_size=sample_size,
            log_file=log_file,
            log=log,
        )

        combined_df = _build_combined_string_alignment_table(
            df_train_audit=df_train_audit,
            df_test_audit=df_test_audit,
            log=log,
        )

        deltas_df = combined_df[combined_df["has_difference"]].copy()
        top_deltas_df = _rank_top_deltas(deltas_df=deltas_df, top_k=top_k, log=log)

        summary_df = pd.DataFrame(
            [
                {
                    "string_like_cols_train": int(df_train_audit.shape[0]),
                    "string_like_cols_test": int(df_test_audit.shape[0]),
                    "string_like_cols_union": int(combined_df.shape[0]),
                    "string_like_cols_with_differences": int(deltas_df.shape[0]),
                    "top_k_returned": int(top_deltas_df.shape[0]),
                }
            ]
        )

        value_differences_df = _build_value_differences(
            df_train=df_train,
            df_test=df_test,
            top_deltas_df=top_deltas_df,
            drilldown_max_columns=drilldown_max_columns,
            drilldown_top_values_per_side=drilldown_top_values_per_side,
            log_file=log_file,
        )

        deltas_to_return = deltas_df.drop(columns=["severity_score", "abs_null_gap", "abs_unique_gap"], errors="ignore").copy()
        top_deltas_to_return = top_deltas_df.drop(columns=["severity_score", "abs_null_gap", "abs_unique_gap"], errors="ignore").copy()

        if export_dir_path is not None:
            _export_alignment_reports(
                summary_df=summary_df,
                deltas_df=deltas_to_return,
                top_deltas_df=top_deltas_to_return,
                value_differences_df=value_differences_df,
                export_dir_path=export_dir_path,
                export_base_name=export_base_name,
                export_suffix=export_suffix,
                export_sample_values_as_json=export_sample_values_as_json,
                log=log,
            )

        log(
            "[string_alignment] done | "
            f"train_cols={summary_df.loc[0, 'string_like_cols_train']} | "
            f"test_cols={summary_df.loc[0, 'string_like_cols_test']} | "
            f"deltas={summary_df.loc[0, 'string_like_cols_with_differences']} | "
            f"value_differences_rows={value_differences_df.shape[0]}"
        )

        return {
            "summary": summary_df,
            "deltas": deltas_to_return,
            "top_deltas": top_deltas_to_return,
            "value_differences": value_differences_df,
        }

    except Exception as exc:
        log(f"[string_alignment] failed | error={exc}")
        raise