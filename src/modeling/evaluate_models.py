from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config.logging as log_config


def generate_model_predictions(
    model,
    X_data: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    log: Callable[[str], None] | Path | str | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Generate class predictions and predicted default probabilities for a fitted model.

    Parameters
    ----------
    model : object
        Fitted classifier implementing predict_proba.
    X_data : pd.DataFrame
        Feature matrix to score.
    dataset_name : str
        Dataset label used for logging.
    model_name : str
        Model label used for logging.
    log : Callable[[str], None] | Path | str | None, default=None
        Callable logger or log file path supported by log_config.emit_log.
    threshold : float, default=0.5
        Probability threshold used to convert predicted default probabilities
        into binary class predictions.

    Returns
    -------
    pd.DataFrame
        DataFrame containing binary predictions and predicted default probabilities.
    """
    try:
        if model is None:
            raise ValueError("model must not be None.")

        if X_data is None:
            raise ValueError("X_data must not be None.")

        if not isinstance(X_data, pd.DataFrame):
            raise ValueError("X_data must be a pandas DataFrame.")

        if X_data.empty:
            raise ValueError("X_data must not be empty.")

        if not dataset_name or not str(dataset_name).strip():
            raise ValueError("dataset_name must be a non-empty string.")

        if not model_name or not str(model_name).strip():
            raise ValueError("model_name must be a non-empty string.")

        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a numeric value.")

        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must lie between 0.0 and 1.0.")

        if not hasattr(model, "predict_proba"):
            raise ValueError("model must implement predict_proba.")

        predicted_probabilities = model.predict_proba(X_data)[:, 1]
        predicted_labels = (predicted_probabilities >= float(threshold)).astype(int)

        prediction_dataframe = pd.DataFrame(
            {
                "predicted_label": predicted_labels,
                "predicted_probability": predicted_probabilities,
            },
            index=X_data.index,
        )

        log_config.emit_log(
            log=log,
            message=(
                "[generate_model_predictions] "
                f"model={model_name} dataset={dataset_name} rows={prediction_dataframe.shape[0]} "
                f"threshold={float(threshold):.3f}"
            ),
        )

        return prediction_dataframe

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=(
                "[generate_model_predictions][error] "
                f"model={model_name} dataset={dataset_name} "
                f"Error={type(exc).__name__}: {exc}"
            ),
        )
        raise


def build_classification_metrics_table(
    y_true: pd.Series,
    prediction_dataframe: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a one-row classification metrics table for a scored model.
    """
    try:
        if y_true is None:
            raise ValueError("y_true must not be None.")

        if prediction_dataframe is None:
            raise ValueError("prediction_dataframe must not be None.")

        if y_true.empty:
            raise ValueError("y_true must not be empty.")

        if prediction_dataframe.empty:
            raise ValueError("prediction_dataframe must not be empty.")

        if not model_name or not str(model_name).strip():
            raise ValueError("model_name must be a non-empty string.")

        if not dataset_name or not str(dataset_name).strip():
            raise ValueError("dataset_name must be a non-empty string.")

        required_columns = {"predicted_label", "predicted_probability"}
        missing_columns = required_columns - set(prediction_dataframe.columns)
        if missing_columns:
            raise KeyError(
                f"prediction_dataframe missing required columns: {sorted(missing_columns)}"
            )

        if len(y_true) != len(prediction_dataframe):
            raise ValueError(
                f"y_true and prediction_dataframe row counts must match. "
                f"Got len(y_true)={len(y_true)} and len(prediction_dataframe)={len(prediction_dataframe)}."
            )

        predicted_labels = prediction_dataframe["predicted_label"]
        predicted_probabilities = prediction_dataframe["predicted_probability"]

        (
            true_negative_count,
            false_positive_count,
            false_negative_count,
            true_positive_count,
        ) = confusion_matrix(
            y_true,
            predicted_labels,
            labels=[0, 1],
        ).ravel()

        metrics_dataframe = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "roc_auc": float(roc_auc_score(y_true, predicted_probabilities)),
                    "accuracy": float(accuracy_score(y_true, predicted_labels)),
                    "precision": float(
                        precision_score(y_true, predicted_labels, zero_division=0)
                    ),
                    "recall": float(
                        recall_score(y_true, predicted_labels, zero_division=0)
                    ),
                    "f1": float(f1_score(y_true, predicted_labels, zero_division=0)),
                    "brier_score": float(
                        brier_score_loss(y_true, predicted_probabilities)
                    ),
                    "true_negative": int(true_negative_count),
                    "false_positive": int(false_positive_count),
                    "false_negative": int(false_negative_count),
                    "true_positive": int(true_positive_count),
                }
            ]
        )

        log_config.emit_log(
            log=log,
            message=(
                "[build_classification_metrics_table] "
                f"model={model_name} dataset={dataset_name}"
            ),
        )

        return metrics_dataframe

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=(
                "[build_classification_metrics_table][error] "
                f"model={model_name} dataset={dataset_name} "
                f"Error={type(exc).__name__}: {exc}"
            ),
        )
        raise
    

def combine_metrics_tables(
    metrics_tables: list[pd.DataFrame],
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Combine one-row model metrics tables into a single comparison table.
    """
    try:
        if not metrics_tables:
            raise ValueError("metrics_tables must contain at least one table.")

        combined_metrics_dataframe = pd.concat(
            metrics_tables,
            axis=0,
            ignore_index=True,
        ).copy()

        log_config.emit_log(
            log=log,
            message=(
                "[combine_metrics_tables] "
                f"combined_rows={combined_metrics_dataframe.shape[0]}"
            ),
        )

        return combined_metrics_dataframe

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=f"[combine_metrics_tables][error] {type(exc).__name__}: {exc}",
        )
        raise


def build_prediction_outcome_table(
    y_true: pd.Series,
    prediction_dataframe: pd.DataFrame,
    loan_amounts: pd.Series,
    model_name: str,
    dataset_name: str,
    log: Callable[[str], None] | Path | str | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Build a row-level outcome table linking actual labels, predicted labels,
    predicted probabilities, and loan amounts.
    """
    try:
        if y_true is None:
            raise ValueError("y_true must not be None.")

        if prediction_dataframe is None:
            raise ValueError("prediction_dataframe must not be None.")

        if loan_amounts is None:
            raise ValueError("loan_amounts must not be None.")

        if y_true.empty:
            raise ValueError("y_true must not be empty.")

        if prediction_dataframe.empty:
            raise ValueError("prediction_dataframe must not be empty.")

        if loan_amounts.empty:
            raise ValueError("loan_amounts must not be empty.")

        if not model_name or not str(model_name).strip():
            raise ValueError("model_name must be a non-empty string.")

        if not dataset_name or not str(dataset_name).strip():
            raise ValueError("dataset_name must be a non-empty string.")

        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                raise ValueError("threshold must be numeric when provided.")
            if not 0.0 <= float(threshold) <= 1.0:
                raise ValueError("threshold must lie between 0.0 and 1.0.")

        required_columns = {"predicted_label", "predicted_probability"}
        missing_columns = required_columns - set(prediction_dataframe.columns)
        if missing_columns:
            raise KeyError(
                f"prediction_dataframe missing required columns: {sorted(missing_columns)}"
            )

        if len(y_true) != len(prediction_dataframe):
            raise ValueError(
                f"y_true and prediction_dataframe row counts must match. "
                f"Got len(y_true)={len(y_true)} and len(prediction_dataframe)={len(prediction_dataframe)}."
            )

        if len(y_true) != len(loan_amounts):
            raise ValueError(
                f"y_true and loan_amounts row counts must match. "
                f"Got len(y_true)={len(y_true)} and len(loan_amounts)={len(loan_amounts)}."
            )

        if not y_true.index.equals(prediction_dataframe.index):
            raise ValueError("y_true and prediction_dataframe indices must match.")

        if not y_true.index.equals(loan_amounts.index):
            raise ValueError("y_true and loan_amounts indices must match.")

        outcome_dataframe = pd.DataFrame(
            {
                "actual_label": y_true,
                "predicted_label": prediction_dataframe["predicted_label"],
                "predicted_probability": prediction_dataframe["predicted_probability"],
                "loan_amnt": loan_amounts,
                "model_name": model_name,
                "dataset_name": dataset_name,
            },
            index=y_true.index,
        )

        if threshold is not None:
            outcome_dataframe["threshold"] = float(threshold)

        outcome_dataframe["outcome_type"] = "unclassified"

        outcome_dataframe.loc[
            (outcome_dataframe["actual_label"] == 1)
            & (outcome_dataframe["predicted_label"] == 1),
            "outcome_type",
        ] = "true_positive"

        outcome_dataframe.loc[
            (outcome_dataframe["actual_label"] == 0)
            & (outcome_dataframe["predicted_label"] == 0),
            "outcome_type",
        ] = "true_negative"

        outcome_dataframe.loc[
            (outcome_dataframe["actual_label"] == 0)
            & (outcome_dataframe["predicted_label"] == 1),
            "outcome_type",
        ] = "false_positive"

        outcome_dataframe.loc[
            (outcome_dataframe["actual_label"] == 1)
            & (outcome_dataframe["predicted_label"] == 0),
            "outcome_type",
        ] = "false_negative"

        if (outcome_dataframe["outcome_type"] == "unclassified").any():
            raise ValueError("One or more rows could not be assigned an outcome_type.")

        log_message = (
            "[build_prediction_outcome_table] "
            f"model={model_name} dataset={dataset_name} rows={outcome_dataframe.shape[0]}"
        )
        if threshold is not None:
            log_message += f" threshold={float(threshold):.3f}"

        log_config.emit_log(
            log=log,
            message=log_message,
        )

        return outcome_dataframe

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=(
                "[build_prediction_outcome_table][error] "
                f"model={model_name} dataset={dataset_name} "
                f"Error={type(exc).__name__}: {exc}"
            ),
        )
        raise


def summarize_outcomes_by_loan_amount(
    outcome_dataframe: pd.DataFrame,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Summarize classification outcomes by row count and loan amount.
    """
    try:
        if outcome_dataframe is None:
            raise ValueError("outcome_dataframe must not be None.")

        if outcome_dataframe.empty:
            raise ValueError("outcome_dataframe must not be empty.")

        required_columns = {"model_name", "dataset_name", "outcome_type", "loan_amnt"}
        missing_columns = required_columns - set(outcome_dataframe.columns)
        if missing_columns:
            raise KeyError(
                f"outcome_dataframe missing required columns: {sorted(missing_columns)}"
            )

        summary_dataframe = (
            outcome_dataframe.groupby(
                ["model_name", "dataset_name", "outcome_type"],
                dropna=False,
            )
            .agg(
                row_count=("outcome_type", "size"),
                total_loan_amnt=("loan_amnt", "sum"),
                average_loan_amnt=("loan_amnt", "mean"),
                median_loan_amnt=("loan_amnt", "median"),
            )
            .reset_index()
        )

        log_config.emit_log(
            log=log,
            message=(
                "[summarize_outcomes_by_loan_amount] "
                f"rows={summary_dataframe.shape[0]}"
            ),
        )

        return summary_dataframe

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=f"[summarize_outcomes_by_loan_amount][error] {type(exc).__name__}: {exc}",
        )
        raise