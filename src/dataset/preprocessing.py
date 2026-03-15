from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd

import config.logging as log_config


# =============================================================================
# Transformations
# =============================================================================


def create_row_identifier(
    df: pd.DataFrame,
    id_column_name: str = "row_id",
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Create a stable internal row identifier for the dataset.

    If the column already exists, it will not be recreated.
    """
    try:
        if df is None:
            raise ValueError("df must not be None")

        if id_column_name in df.columns:
            log_config.emit_log(log, f"Row identifier already exists: {id_column_name}")
            return df

        transformed_dataframe = df.copy()
        transformed_dataframe.insert(0, id_column_name, range(1, len(transformed_dataframe) + 1))

        log_config.emit_log(log, f"Row identifier created: {id_column_name} | Total rows: {len(transformed_dataframe)}")
        return transformed_dataframe

    except Exception as exc:
        try:
            log_config.emit_log(log, f"Error while creating row identifier: {exc}")
        except Exception:
            pass
        raise


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


def apply_credit_sentinel_encoding(
    df: pd.DataFrame,
    credit_recency_columns: list[str],
    sentinel_value: int = 9999,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Apply sentinel encoding to credit recency variables.
    - Adds indicator columns: has_<col> (int8)
    - Fills missing values in <col> with sentinel_value
    """
    try:
        if df is None:
            raise ValueError("df must not be None")

        transformed_dataframe = df.copy()
        log_config.emit_log(log, "[credit_sentinel_encoding] start")

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

            log_config.emit_log(
                log,
                f"[credit_sentinel_encoding] column={column_name} "
                f"missing={missing_value_count} sentinel={sentinel_value} "
                f"indicator={indicator_column_name}",
            )

        if missing_configured_columns:
            log_config.emit_log(
                log,
                f"[credit_sentinel_encoding][error] Missing configured columns: {missing_configured_columns}",
            )
            raise KeyError(f"Missing configured credit recency columns: {missing_configured_columns}")

        log_config.emit_log(log, "[credit_sentinel_encoding] completed")
        return transformed_dataframe

    except Exception as exc:
        try:
            log_config.emit_log(
                log,
                f"[credit_sentinel_encoding][error] "
                f"Error={type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise


def convert_month_year_columns_to_datetime(
    df: pd.DataFrame,
    column_names: list[str],
    datetime_formats: list[str] | None = None,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Convert string-based datetime columns to pandas datetime.

    Parsing strategy:
    1. Try any explicitly provided datetime formats in order.
    2. If none succeed, fall back to pandas mixed-format inference.
    3. Raise an error if parsing still fails.

    Examples of supported explicit formats:
    - "%b-%Y" -> "Jan-2016"
    - "%b-%y" -> "Jan-16"
    """
    try:
        if df is None:
            raise ValueError("df must not be None")

        transformed_dataframe = df.copy()
        requested_columns = list(dict.fromkeys(column_names))

        if datetime_formats is None:
            datetime_formats = []

        log_config.emit_log(
            log,
            f"Starting datetime conversion for {len(requested_columns)} columns | "
            f"explicit_formats={datetime_formats if datetime_formats else 'none'}"
        )

        for column_name in requested_columns:
            if column_name not in transformed_dataframe.columns:
                log_config.emit_log(log, f"Column not found (skipped): {column_name}")
                continue

            series_before = transformed_dataframe[column_name]
            normalized_series = series_before.astype("string").str.strip()
            non_null_mask = normalized_series.notna()

            if not non_null_mask.any():
                transformed_dataframe[column_name] = pd.Series(
                    pd.NaT,
                    index=transformed_dataframe.index,
                    dtype="datetime64[ns]",
                ).to_numpy()
                log_config.emit_log(log, f"Converted to datetime: {column_name} | all values missing")
                continue

            parsed_series: pd.Series | None = None
            parsing_method = "unresolved"

            for datetime_format in datetime_formats:
                try:
                    parsed_series = pd.to_datetime(
                        normalized_series[non_null_mask],
                        format=datetime_format,
                        errors="raise",
                    )
                    parsing_method = f"explicit_format={datetime_format}"
                    break
                except Exception:
                    continue

            if parsed_series is None:
                parsed_series = pd.to_datetime(
                    normalized_series[non_null_mask],
                    format="mixed",
                    errors="raise",
                )
                parsing_method = "mixed_inference"

            datetime_series = pd.Series(
                pd.NaT,
                index=transformed_dataframe.index,
                dtype="datetime64[ns]",
            )
            datetime_series.loc[non_null_mask] = parsed_series.to_numpy()

            transformed_dataframe[column_name] = datetime_series.to_numpy()

            log_config.emit_log(
                log,
                f"Converted to datetime: {column_name} | "
                f"dtype={transformed_dataframe[column_name].dtype} | "
                f"method={parsing_method}"
            )

        log_config.emit_log(log, "Datetime conversion completed successfully")
        return transformed_dataframe

    except Exception as exc:
        try:
            log_config.emit_log(log, f"Error in convert_month_year_columns_to_datetime: {exc}")
        except Exception:
            pass
        raise


def convert_percent_string_columns_to_numeric(
    df_to_transform: pd.DataFrame,
    *,
    percent_columns: list[str],
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Convert percent-encoded numeric columns (e.g. "13.56%") to float.
    """

    try:

        df_transformed: pd.DataFrame = df_to_transform.copy()

        for column_name in percent_columns:

            if column_name not in df_transformed.columns:
                log_config.emit_log(
                    log,
                    f"[convert_percent_string_columns_to_numeric] column not found (skipped): {column_name}",
                )
                continue

            df_transformed[column_name] = (
                df_transformed[column_name]
                .astype("string")
                .str.replace("%", "", regex=False)
                .str.strip()
                .astype(float)
            )

            log_config.emit_log(
                log,
                f"Converted percent string column to numeric: {column_name} | "
                f"dtype={df_transformed[column_name].dtype}"
            )

        log_config.emit_log(
            log,
            "Percent string numeric conversion completed successfully | "
            f"columns_converted={percent_columns}",
        )

        return df_transformed

    except Exception as exc:
        try:
            log_config.emit_log(
                log,
                f"Error in convert_percent_string_columns_to_numeric: {exc}",
            )
        except Exception:
            pass
        raise


def transform_emp_length(
    df: pd.DataFrame,
    column_name: str = "emp_length",
    output_column_name: str = "emp_length_years",
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Normalize LendingClub-style employment length strings into numeric years.

    Policy
    ------
    - Known missing tokens (e.g. "n/a") are treated as missing values.
    - Any other unexpected non-null values raise an error.
    """
    try:
        if df is None:
            raise ValueError("df must not be None")

        if column_name not in df.columns:
            log_config.emit_log(log, f"[transform_emp_length] column not found (skipped): {column_name}")
            return df

        transformed_dataframe = df.copy()

        raw_series = transformed_dataframe[column_name]

        normalized_series = (
            raw_series
            .astype("string")
            .str.strip()
            .str.lower()
        )

        missing_tokens = {"n/a", "na", "n.a."}
        normalized_series = normalized_series.replace(list(missing_tokens), pd.NA)

        employment_length_mapping: dict[str, int] = {
            "< 1 year": 0,
            "10+ years": 10,
        }

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
            log_config.emit_log(
                log,
                "[transform_emp_length][error] unexpected non-null values detected: "
                + ", ".join(unexpected_values[:25])
                + (" ..." if len(unexpected_values) > 25 else ""),
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

        log_config.emit_log(
            log,
            f"[transform_emp_length] created '{output_column_name}' from '{column_name}' | "
            f"rows={total_rows} | "
            f"input_nulls={input_null_count} | "
            f"normalized_nulls={normalized_null_count} | "
            f"output_nulls={output_null_count}",
        )

        return transformed_dataframe

    except Exception as exc:
        try:
            log_config.emit_log(
                log,
                f"[transform_emp_length][error] "
                f"Error={type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise


def normalize_home_ownership(
    df: pd.DataFrame,
    column_name: str = "home_ownership",
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Normalize home_ownership so train/test have a stable category space.
    """
    try:
        if df is None:
            raise ValueError("df must not be None")

        if column_name not in df.columns:
            log_config.emit_log(log, f"normalize_home_ownership: column not found (skipped): {column_name}")
            return df

        transformed_dataframe = df.copy()

        series_before = transformed_dataframe[column_name]
        normalized_series = series_before.astype("string").str.strip().str.lower()

        mapping = {"none": "other", "any": "other"}
        normalized_series = normalized_series.replace(mapping)

        transformed_dataframe[column_name] = normalized_series

        unique_before = int(series_before.nunique(dropna=False))
        unique_after = int(transformed_dataframe[column_name].nunique(dropna=False))

        log_config.emit_log(
            log,
            f"normalize_home_ownership: {column_name} | unique_before={unique_before} | "
            f"unique_after={unique_after} | mapped={list(mapping.keys())} -> 'other'",
        )

        return transformed_dataframe

    except Exception as exc:
        try:
            log_config.emit_log(log, f"Error in normalize_home_ownership: {exc}")
        except Exception:
            pass
        raise


def normalize_text_columns(
    df: pd.DataFrame,
    columns_to_normalize: Iterable[str],
    *,
    lowercase: bool = True,
    strip_whitespace: bool = True,
    collapse_whitespace: bool = True,
    replace_spaces_with_underscore: bool = True,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Normalize categorical/text-like columns.
    """
    try:
        if df is None:
            raise ValueError("df must not be None")

        normalized_dataframe = df.copy()

        requested_columns = list(dict.fromkeys(columns_to_normalize))
        existing_columns = [
            column_name for column_name in requested_columns if column_name in normalized_dataframe.columns
        ]
        missing_columns = [
            column_name for column_name in requested_columns if column_name not in normalized_dataframe.columns
        ]

        if missing_columns:
            log_config.emit_log(log, "normalize_text_columns: missing columns (skipped): " + ", ".join(missing_columns))

        if not existing_columns:
            log_config.emit_log(log, "normalize_text_columns: no matching columns to normalize.")
            return normalized_dataframe

        log_config.emit_log(log, f"normalize_text_columns: normalizing {len(existing_columns)} columns")

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

            log_config.emit_log(
                log,
                f"Normalized '{column_name}': unique_non_null {unique_before} -> {unique_after} | "
                f"null_percent={null_percent:.2f}%",
            )

        return normalized_dataframe

    except Exception as exc:
        try:
            log_config.emit_log(log, f"Error in normalize_text_columns: {exc}")
        except Exception:
            pass
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
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Apply a controlled mapping to a categorical column.
    """
    try:
        if df is None:
            raise ValueError("df must not be None")

        if mapping is None:
            raise ValueError("mapping must not be None")

        if column_name not in df.columns:
            log_config.emit_log(log, f"apply_categorical_mapping: column not found (skipped): {column_name}")
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
            log_config.emit_log(
                log,
                f"apply_categorical_mapping: unmapped values in '{column_name}': "
                + ", ".join(unmapped_values[:25])
                + (" ..." if len(unmapped_values) > 25 else ""),
            )
            raise ValueError(f"Unmapped values in '{column_name}': {unmapped_values[:10]}")

        mapped_series = normalized_series.map(mapping)

        if allow_unmapped_values:
            mapped_series = mapped_series.where(normalized_series.isna() | mapped_series.notna(), normalized_series)

        target_column_name = output_column_name or column_name
        transformed_dataframe[target_column_name] = mapped_series

        log_config.emit_log(
            log,
            f"apply_categorical_mapping: '{column_name}' -> '{target_column_name}' | "
            f"mapped_keys={len(mapping)} | unmapped_values={len(unmapped_values)} | "
            f"allow_unmapped_values={allow_unmapped_values}",
        )

        return transformed_dataframe

    except Exception as exc:
        try:
            log_config.emit_log(log, f"Error in apply_categorical_mapping: {exc}")
        except Exception:
            pass
        raise


def parse_term_column(
    df: pd.DataFrame,
    term_column: str,
    new_column_name: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Parse LendingClub loan term (e.g. '36 months') into numeric months.
    """
    try:
        if df is None:
            raise ValueError("df must not be None")

        transformed_dataframe = df.copy()

        log_config.emit_log(log, "[parse_term_column] parsing loan term")

        transformed_dataframe[new_column_name] = (
            transformed_dataframe[term_column]
            .astype("string")
            .str.strip()
            .str.extract(r"(\d+)")[0]
            .astype("Int16")
        )

        log_config.emit_log(log, f"[parse_term_column] created column={new_column_name}")

        return transformed_dataframe

    except Exception as exc:
        try:
            log_config.emit_log(
                log,
                f"[parse_term_column][error] "
                f"Error={type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise


def cast_categorical_columns(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    column_names: list[str],
    log: Callable[[str], None] | Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cast selected columns to pandas categorical dtype with a shared
    category space across train and test datasets.
    """
    try:
        if df_train is None or df_test is None:
            raise ValueError("Input dataframes must not be None")

        train_df = df_train.copy()
        test_df = df_test.copy()

        log_config.emit_log(log, "[cast_categorical_columns] casting string -> category")

        for column_name in column_names:
            if column_name not in train_df.columns:
                log_config.emit_log(log, f"Train missing categorical column (skipped): {column_name}")
                continue

            if column_name not in test_df.columns:
                log_config.emit_log(log, f"Test missing categorical column (skipped): {column_name}")
                continue

            train_df[column_name] = train_df[column_name].astype("string")
            test_df[column_name] = test_df[column_name].astype("string")

            train_values = train_df[column_name].dropna().unique().tolist()
            test_values = test_df[column_name].dropna().unique().tolist()

            combined_categories = sorted(set(train_values) | set(test_values))

            train_df[column_name] = pd.Categorical(
                train_df[column_name],
                categories=combined_categories,
            )

            test_df[column_name] = pd.Categorical(
                test_df[column_name],
                categories=combined_categories,
            )

            log_config.emit_log(
                log,
                f"[cast_categorical_columns] column={column_name} "
                f"train_unique={len(train_values)} "
                f"test_unique={len(test_values)} "
                f"combined={len(combined_categories)}",
            )

        return train_df, test_df

    except Exception as exc:
        try:
            log_config.emit_log(
                log,
                f"[cast_categorical_columns][error] "
                f"Error={type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise


def align_numeric_dtypes_between_train_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    log: Callable[[str], None] | Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align numeric dtypes across shared columns in training and test datasets.

    If a column is numeric in both datasets but the dtypes differ
    (for example float64 in training and int64 in test), both columns
    are coerced to float64.
    """
    try:
        if df_train is None or df_test is None:
            raise ValueError("df_train and df_test must not be None")

        aligned_train = df_train.copy()
        aligned_test = df_test.copy()

        shared_columns = [
            column_name
            for column_name in aligned_train.columns
            if column_name in aligned_test.columns
        ]

        coerced_columns: list[str] = []

        for column_name in shared_columns:
            train_series = aligned_train[column_name]
            test_series = aligned_test[column_name]

            train_is_numeric = pd.api.types.is_numeric_dtype(train_series)
            test_is_numeric = pd.api.types.is_numeric_dtype(test_series)

            if not (train_is_numeric and test_is_numeric):
                continue

            if train_series.dtype != test_series.dtype:
                aligned_train[column_name] = aligned_train[column_name].astype("float64")
                aligned_test[column_name] = aligned_test[column_name].astype("float64")
                coerced_columns.append(column_name)

        log_config.emit_log(
            log,
            f"[align_numeric_dtypes_between_train_test] coerced_columns={len(coerced_columns)}"
        )

        if coerced_columns:
            log_config.emit_log(
                log,
                "[align_numeric_dtypes_between_train_test] columns=" + _format_column_list(coerced_columns)
            )

        return aligned_train, aligned_test

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[align_numeric_dtypes_between_train_test][error] {exc}")
        except Exception:
            pass
        raise