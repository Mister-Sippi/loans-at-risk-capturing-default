from __future__ import annotations

from pathlib import Path
from typing import Callable, Any

import joblib
import json

import config.logging as log_config


def save_model(
    model: Any,
    output_path: Path,
    metadata: dict[str, object] | None = None,
    log: Callable[[str], None] | Path | str | None = None,
) -> Path:
    """
    Save a trained model to disk with optional metadata.

    Parameters
    ----------
    model : Any
        Trained model object (e.g., CatBoostClassifier).
    output_path : Path
        File path where the model will be saved.
    metadata : dict[str, object] | None, default=None
        Optional metadata dictionary saved alongside the model.
    log : Callable | Path | str | None, default=None
        Logging target supported by log_config.emit_log.

    Returns
    -------
    Path
        Path to the saved model file.
    """

    try:
        if model is None:
            raise ValueError("model must not be None.")

        if output_path is None:
            raise ValueError("output_path must not be None.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(model, output_path)

        log_config.emit_log(
            log=log,
            message=f"[save_model] Model saved to {output_path}",
        )

        # Save metadata if provided
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a dictionary or None.")

            metadata_path = output_path.with_suffix(".metadata.json")

            with metadata_path.open("w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=4)

            log_config.emit_log(
                log=log,
                message=f"[save_model] Metadata saved to {metadata_path}",
            )

        return output_path

    except Exception as exc:
        log_config.emit_log(
            log=log,
            message=f"[save_model][error] {type(exc).__name__}: {exc}",
        )
        raise