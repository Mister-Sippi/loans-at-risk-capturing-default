import pandas as pd
from typing import Any, Callable, Iterable, Mapping, List
from python.logging_utils import get_logger


def initial_inspection(df: pd.DataFrame, log_file: str | None = None) -> dict:
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
        log("===== Starting initial data inspection =====")

        # Shape and memory
        row_count, column_count = df.shape
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
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
        constant_columns = [column_name for column_name in constant_columns_all if column_name not in fully_null_columns]

        if constant_columns:
            log("\nConstant columns (non-null):")
            log(", ".join(constant_columns))

        # Mixed type columns
        mixed_type_columns = []
        for column_name in df.columns:
            if df[column_name].dropna().map(type).nunique() > 1:
                mixed_type_columns.append(column_name)

        if mixed_type_columns:
            log("\nColumns with mixed types:")
            log(", ".join(mixed_type_columns))

        # Object columns that are numeric (ignore nulls)
        numeric_object_columns = []
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

    except Exception as error:
        log(f"Error during initial data inspection: {error}")
        raise


def create_row_identifier(
    dataframe: pd.DataFrame,
    id_column_name: str = "row_id",
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Create a stable internal row identifier for the dataset.

    The identifier:
    - Is sequential
    - Is independent of platform-specific identifiers
    - Is created before any filtering or transformation
    - Ensures consistent traceability across derived datasets

    If the column already exists, it will not be recreated.
    """

    log = get_logger(log_file)

    try:
        if id_column_name in dataframe.columns:
            log(f"Row identifier already exists: {id_column_name}")
            return dataframe

        transformed_dataframe = dataframe.copy()

        transformed_dataframe.insert(
            0,
            id_column_name,
            range(1, len(transformed_dataframe) + 1),
        )

        log(
            f"Row identifier created: {id_column_name} | "
            f"Total rows: {len(transformed_dataframe)}"
        )

        return transformed_dataframe

    except Exception as error:
        log(f"Error while creating row identifier: {error}")
        raise


def apply_credit_sentinel_encoding(
    df: pd.DataFrame,
    credit_recency_columns: list[str],
    sentinel_value: int = 9999,
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Apply sentinel encoding to credit recency variables. In these columns, missing values represent absence of a prior delinquency or derogatory event and are therefore encoded using an indicator variable and a sentinel value.

    For each column in timing_columns:
    - Create an indicator column named: has_<column_name>
      (1 if value is present, 0 if null)
    - Replace nulls with sentinel_value (default: 9999)

    This keeps the original variable usable while preserving the meaning of nulls.
    """
    log = get_logger(log_file)

    try:
        transformed_dataframe = df.copy()
        log("Applying credit sentinel encoding")

        for column_name in credit_recency_columns:
            if column_name not in transformed_dataframe.columns:
                log(f"Credit recency column not found (skipped): {column_name}")
                continue

            indicator_column_name = f"has_{column_name}"
            missing_value_count = int(transformed_dataframe[column_name].isna().sum())

            transformed_dataframe[indicator_column_name] = (
                transformed_dataframe[column_name].notna().astype("int8")
            )
            transformed_dataframe[column_name] = (
                transformed_dataframe[column_name].fillna(sentinel_value)
            )

            log(
                f"Encoded {column_name}: "
                f"missing={missing_value_count} | "
                f"sentinel={sentinel_value} | "
                f"indicator={indicator_column_name}"
            )

        log("Credit sentinel encoding completed")
        return transformed_dataframe

    except Exception as error:
        log(f"Error in apply_credit_sentinel_encoding: {error}")
        raise
    

def drop_columns_with_logging(
    dataframe: pd.DataFrame,
    columns_to_drop: list[str],
    dataset_name: str,
    log_file: str | None = None,
    errors: str = "ignore",
) -> pd.DataFrame:
    """
    Drop columns from a DataFrame and log what happened.

    Logs:
    - dataset name
    - shape before/after
    - requested drop count
    - dropped columns
    - columns missing from the DataFrame
    """

    log = get_logger(log_file)

    try:
        shape_before = dataframe.shape

        existing_columns = set(dataframe.columns)
        requested_columns = list(dict.fromkeys(columns_to_drop))  # de-duplicate, preserve order

        dropped_columns = [column_name for column_name in requested_columns if column_name in existing_columns]
        missing_columns = [column_name for column_name in requested_columns if column_name not in existing_columns]

        transformed_dataframe = dataframe.drop(columns=requested_columns, errors=errors)

        shape_after = transformed_dataframe.shape

        log(f"Dropping columns | dataset={dataset_name}")
        log(f"Shape before: {shape_before} | Shape after: {shape_after}")
        log(f"Requested: {len(requested_columns)} | Dropped: {len(dropped_columns)} | Missing: {len(missing_columns)}")

        if dropped_columns:
            log("Dropped columns: " + ", ".join(dropped_columns))

        if missing_columns:
            log("Missing columns (not present): " + ", ".join(missing_columns))

        return transformed_dataframe

    except Exception as error:
        log(f"Error while dropping columns | dataset={dataset_name}: {error}")
        raise


def audit_string_columns(
    df: pd.DataFrame,
    sample_size: int = 5,
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Create an audit table for string/object columns.

    Returns a DataFrame with:
    - column_name
    - dtype
    - unique_count_including_null
    - unique_count_non_null
    - null_percent
    - sample_values (up to sample_size distinct non-null values)

    Intended for structural inspection, not feature engineering.
    """
    log = get_logger(log_file)

    try:
        string_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

        if not string_columns:
            log("audit_string_columns: No string/object columns found.")
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

        for column_name in string_columns:
            series = df[column_name]

            unique_count_including_null = int(series.nunique(dropna=False))
            unique_count_non_null = int(series.nunique(dropna=True))
            null_percent = float((series.isna().mean() * 100).round(2))

            sample_values = (
                series.dropna()
                .astype(str)
                .drop_duplicates()
                .head(sample_size)
                .tolist()
            )

            records.append({
                "column_name": column_name,
                "dtype": str(series.dtype),
                "unique_count_including_null": unique_count_including_null,
                "unique_count_non_null": unique_count_non_null,
                "null_percent": null_percent,
                "sample_values": sample_values,
            })

        audit_dataframe = (
            pd.DataFrame(records)
            .sort_values(by=["unique_count_non_null", "null_percent"], ascending=[False, False])
            .reset_index(drop=True)
        )

        log(f"audit_string_columns: audited {len(audit_dataframe)} string/object columns.")
        return audit_dataframe

    except Exception as error:
        log(f"Error in audit_string_columns: {error}")
        raise    


def compare_categorical_column_values(
    training_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    column_name: str,
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Compare categorical values between training and test datasets.

    Returns a DataFrame containing:
    - category value
    - whether present in training
    - whether present in test

    Intended for schema validation and category alignment prior to encoding.
    """

    log = get_logger(log_file)

    try:
        if column_name not in training_dataframe.columns:
            raise ValueError(f"Column not found in training dataset: {column_name}")

        if column_name not in test_dataframe.columns:
            raise ValueError(f"Column not found in test dataset: {column_name}")

        log(f"Comparing categorical values for column: {column_name}")

        training_values = set(
            training_dataframe[column_name]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        test_values = set(
            test_dataframe[column_name]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        all_categories = sorted(training_values.union(test_values))

        comparison_dataframe = pd.DataFrame({
            "category": all_categories,
            "present_in_training": [
                category in training_values for category in all_categories
            ],
            "present_in_test": [
                category in test_values for category in all_categories
            ],
        })

        log(
            f"{column_name}: "
            f"training_unique={len(training_values)} | "
            f"test_unique={len(test_values)} | "
            f"combined_unique={len(all_categories)}"
        )

        return comparison_dataframe

    except Exception as error:
        log(f"Error in compare_categorical_column_values: {error}")
        raise


def convert_month_year_columns_to_datetime(
    dataframe: pd.DataFrame,
    column_names: list[str],
    datetime_format: str = "%b-%y",
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Convert month-year string columns (e.g. 'Jan-16') to pandas datetime.

    Notes:
    - Uses the first day of the month.
    - Preserves NaN.
    - Forces safe assignment even if the source column is pandas StringDtype.
    - Raises if parsing fails (strict), so drift is caught early.
    """
    log = get_logger(log_file)

    try:
        transformed_dataframe = dataframe.copy()

        requested_columns = list(dict.fromkeys(column_names))
        log(f"Starting month-year to datetime conversion for {len(requested_columns)} columns")

        for column_name in requested_columns:
            if column_name not in transformed_dataframe.columns:
                log(f"Column not found (skipped): {column_name}")
                continue

            series_before = transformed_dataframe[column_name]

            # Keep true missing values as missing, normalize non-null strings
            normalized_series = series_before.astype("string").str.strip()
            non_null_mask = normalized_series.notna()

            # Parse only non-null values strictly
            parsed_series = pd.to_datetime(
                normalized_series[non_null_mask],
                format=datetime_format,
                errors="raise",
            )

            # Build final datetime Series aligned to dataframe index
            datetime_series = pd.Series(pd.NaT, index=transformed_dataframe.index, dtype="datetime64[ns]")
            datetime_series.loc[non_null_mask] = parsed_series.to_numpy()

            # Critical: force safe dtype replacement (avoids StringArray assignment TypeError)
            transformed_dataframe[column_name] = datetime_series.to_numpy()

            log(f"Converted to datetime: {column_name} | dtype={transformed_dataframe[column_name].dtype}")

        log("Month-year datetime conversion completed successfully")
        return transformed_dataframe

    except Exception as error:
        log(f"Error in convert_month_year_columns_to_datetime: {error}")
        raise


def transform_emp_length(
    df: pd.DataFrame,
    column_name: str = "emp_length",
    output_column_name: str = "emp_length_years",
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Normalize LendingClub-style employment length strings into numeric years.

    Mapping:
    - "< 1 year"  -> 0
    - "1 year"    -> 1
    - "2 years"   -> 2
    - ...
    - "10+ years" -> 10
    - null/NaN    -> NaN (preserved)

    Notes:
    - This is format normalization only (no imputation).
    - Unexpected non-null values raise a ValueError to avoid silent data drift.
    """
    log = get_logger(log_file)

    try:
        if column_name not in df.columns:
            log(f"transform_emp_length: column not found (skipped): {column_name}")
            return df

        transformed_dataframe = df.copy()

        raw_series = transformed_dataframe[column_name]

        # Preserve NaN; normalize strings
        normalized_series = (
            raw_series.astype("string")
            .str.strip()
            .str.lower()
        )

        # Build mapping table (explicit, readable, stable)
        mapping: dict[str, int] = {
            "< 1 year": 0,
            "10+ years": 10,
        }
        for year_value in range(1, 10):
            mapping[f"{year_value} year"] = year_value
            mapping[f"{year_value} years"] = year_value

        # Identify unexpected values before mapping (excluding NaN)
        non_null_values = normalized_series.dropna()
        unexpected_values = sorted(set(non_null_values.unique()) - set(mapping.keys()))
        if unexpected_values:
            log(
                "transform_emp_length: unexpected non-null values detected: "
                + ", ".join(unexpected_values[:25])
                + (" ..." if len(unexpected_values) > 25 else "")
            )
            raise ValueError(
                f"Unexpected values in '{column_name}': {unexpected_values[:10]}"
                + (" (and more)" if len(unexpected_values) > 10 else "")
            )

        # Apply mapping; keep missing as NaN
        transformed_dataframe[output_column_name] = non_null_values.map(mapping)
        transformed_dataframe[output_column_name] = transformed_dataframe[output_column_name].astype("Float32")

        # Log summary
        total_rows = int(len(transformed_dataframe))
        null_count = int(raw_series.isna().sum())
        output_null_count = int(transformed_dataframe[output_column_name].isna().sum())
        log(
            f"transform_emp_length: created '{output_column_name}' from '{column_name}' | "
            f"rows={total_rows} | input_nulls={null_count} | output_nulls={output_null_count}"
        )

        return transformed_dataframe

    except Exception as error:
        log(f"Error in transform_emp_length: {error}")
        raise


def normalize_home_ownership(
    df: pd.DataFrame,
    column_name: str = "home_ownership",
    log_file: str | None = None,
) -> pd.DataFrame:
    """
    Normalize home_ownership so train/test have a stable category space.

    Steps:
    - strip whitespace and lowercase
    - map legacy/rare categories to 'other' (e.g., NONE, ANY)
    """
    log = get_logger(log_file)

    try:
        if column_name not in df.columns:
            log(f"normalize_home_ownership: column not found (skipped): {column_name}")
            return df

        transformed_dataframe = df.copy()

        series_before = transformed_dataframe[column_name]

        # Normalize strings safely
        normalized_series = (
            series_before.astype("string")
            .str.strip()
            .str.lower()
        )

        # Consolidate rare/legacy categories
        mapping = {
            "none": "other",
            "any": "other",
        }
        normalized_series = normalized_series.replace(mapping)

        transformed_dataframe[column_name] = normalized_series

        unique_before = int(series_before.nunique(dropna=False))
        unique_after = int(transformed_dataframe[column_name].nunique(dropna=False))

        log(
            f"normalize_home_ownership: {column_name} | "
            f"unique_before={unique_before} | unique_after={unique_after} | "
            f"mapped={list(mapping.keys())} -> 'other'"
        )

        return transformed_dataframe

    except Exception as error:
        log(f"Error in normalize_home_ownership: {error}")
        raise


def normalize_text_columns(
    dataframe: pd.DataFrame,
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

    Operations (optional):
    - strip leading/trailing whitespace
    - collapse repeated internal whitespace
    - lowercase
    - replace spaces with underscore

    Notes:
    - Preserves NaN
    - Casts non-null values to string safely
    - Intended for categorical normalization, not free-text NLP
    """

    log: Callable[[str], None] = get_logger(log_file)

    try:
        normalized_dataframe = dataframe.copy()

        requested_columns = list(dict.fromkeys(columns_to_normalize))
        existing_columns = [
            column_name for column_name in requested_columns
            if column_name in normalized_dataframe.columns
        ]
        missing_columns = [
            column_name for column_name in requested_columns
            if column_name not in normalized_dataframe.columns
        ]

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

            normalized_series = series_before.where(
                series_before.isna(),
                series_before.astype("string")
            )

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
                f"Normalized '{column_name}': "
                f"unique_non_null {unique_before} -> {unique_after} | "
                f"null_percent={null_percent:.2f}%"
            )

        return normalized_dataframe

    except Exception as error:
        log(f"Error in normalize_text_columns: {error}")
        raise


def apply_categorical_mapping(
    dataframe: pd.DataFrame,
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
    - Preserves NaN
    - Optionally enforces that all non-null values must be mappable (to catch drift)
    """
    log = get_logger(log_file)

    try:
        if column_name not in dataframe.columns:
            log(f"apply_categorical_mapping: column not found (skipped): {column_name}")
            return dataframe

        transformed_dataframe = dataframe.copy()

        series_before = transformed_dataframe[column_name].astype("string")

        if strip_whitespace:
            series_before = series_before.str.strip()

        if lowercase:
            series_before = series_before.str.lower()

        non_null_values = series_before.dropna().unique().tolist()
        unmapped_values = sorted(set(non_null_values) - set(mapping.keys()))

        if unmapped_values and not allow_unmapped_values:
            log(
                f"apply_categorical_mapping: unmapped values in '{column_name}': "
                + ", ".join(unmapped_values[:25])
                + (" ..." if len(unmapped_values) > 25 else "")
            )
            raise ValueError(f"Unmapped values in '{column_name}': {unmapped_values[:10]}")

        target_column_name = output_column_name or column_name
        transformed_dataframe[target_column_name] = series_before.map(mapping).where(series_before.isna(), series_before.map(mapping))

        # If allow_unmapped_values=True, keep original for unmapped values
        if allow_unmapped_values:
            transformed_dataframe[target_column_name] = transformed_dataframe[target_column_name].fillna(series_before)

        log(
            f"apply_categorical_mapping: '{column_name}' -> '{target_column_name}' | "
            f"mapped_keys={len(mapping)} | unmapped_values={len(unmapped_values)}"
        )

        return transformed_dataframe

    except Exception as error:
        log(f"Error in apply_categorical_mapping: {error}")
        raise
