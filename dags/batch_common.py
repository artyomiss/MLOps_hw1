import hashlib
import json
import math
import os
import re
import tempfile
from datetime import date
from pathlib import Path
import logging
import numbers
from collections.abc import Mapping

import numpy as np

FEATURES = ["age", "tenure_months", "monthly_spend", "support_tickets_30d"]
LIMITS = [(18, 100), (0, 240), (0, 10000), (0, 100)]
ARTIFACTS = Path(os.environ.get("BATCH_ARTIFACTS", "/opt/airflow/include/artifacts"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save(value):
    content = canonical(value)
    sha = digest(content)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ARTIFACTS, delete=False) as file:
        file.write(content)
        temporary = Path(file.name)
    temporary.replace(ARTIFACTS / f"{sha}.json")
    return {"sha256": sha}


def read(ref):
    sha = ref["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid artifact reference")
    content = (ARTIFACTS / f"{sha}.json").read_bytes()
    if digest(content) != sha:
        raise ValueError("Artifact checksum mismatch")
    return json.loads(content)


def business_date(explicit, interval_start):
    if explicit:
        return date.fromisoformat(explicit).isoformat()
    if interval_start is None:
        raise ValueError("business_date is required without a data interval")
    return interval_start.in_timezone("Europe/Moscow").to_date_string()


def validate_rows(rows, day):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Rows must be a non-empty list")

    seen_ids = set()

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Row {index} must be a mapping")

        customer_id = row["customer_id"]

        if not isinstance(customer_id, str) or not customer_id.strip():
            raise ValueError(f"Invalid customer_id in row {index}")

        if customer_id in seen_ids:
            raise ValueError(f"Duplicate customer_id: {customer_id}")

        seen_ids.add(customer_id)

        if row["business_date"] != day:
            raise ValueError(
                f"Invalid business_date for customer {customer_id}: "
                f"expected {day}, got {row['business_date']}"
            )

        for feature, (minimum, maximum) in zip(FEATURES, LIMITS):
            value = row[feature]

            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(
                    f"Invalid value for {feature} "
                    f"for customer {customer_id}"
                )

            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    f"Invalid numeric value for {feature} "
                    f"for customer {customer_id}"
                )

            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Non-finite value for {feature} "
                    f"for customer {customer_id}"
                )

            if not minimum <= numeric_value <= maximum:
                raise ValueError(
                    f"Value out of range for {feature} "
                    f"for customer {customer_id}: {value}"
                )

    return rows


def validate_model(model):
    if model["feature_names"] != FEATURES:
        raise ValueError("Feature order mismatch")
    for field in ("mean", "scale", "weights"):
        if len(model[field]) != len(FEATURES):
            raise ValueError("Model dimensions mismatch")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in model[field]):
            raise ValueError("Invalid model coefficient")
    if any(v <= 0 for v in model["scale"]) or type(model["bias"]) not in (int, float) or not math.isfinite(model["bias"]):
        raise ValueError("Invalid model scale or bias")


def predict(rows, model):
    validate_model(model)
    matrix = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float64)
    normalized = (matrix - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = normalized @ np.asarray(model["weights"]) + model["bias"]
    scores = np.round(1 / (1 + np.exp(-np.clip(logits, -700, 700))), 8)
    return [{"customer_id": row["customer_id"], "score": float(score), "label": int(score >= 0.5)}
            for row, score in zip(rows, scores)]


def infer_batches(rows, model, batch_size):
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= 1000
    ):
        raise ValueError("batch_size must be an integer between 1 and 1000")

    logger = logging.getLogger(__name__)
    predictions = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        batch_number = start // batch_size + 1

        logger.info(
            "Processing batch %d: %d rows",
            batch_number,
            len(batch),
        )

        batch_predictions = predict(batch, model)
        predictions.extend(batch_predictions)

    return predictions


def validate_predictions(predictions, expected_ids):
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list")

    expected_ids = list(expected_ids)

    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Expected customer IDs contain duplicates")

    if len(predictions) != len(expected_ids):
        raise ValueError(
            f"Prediction count mismatch: expected {len(expected_ids)}, "
            f"got {len(predictions)}"
        )

    actual_ids = []

    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, Mapping):
            raise ValueError(f"Prediction {index} must be a mapping")

        try:
            customer_id = prediction["customer_id"]
            score = prediction["score"]
            label = prediction["label"]
        except KeyError as exc:
            raise ValueError(
                f"Missing prediction field: {exc.args[0]}"
            ) from exc

        if not isinstance(customer_id, str) or not customer_id.strip():
            raise ValueError(f"Invalid customer_id in prediction {index}")

        actual_ids.append(customer_id)

        if isinstance(score, bool) or not isinstance(score, numbers.Real):
            raise ValueError(f"Invalid score for customer {customer_id}")

        try:
            numeric_score = float(score)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"Invalid score for customer {customer_id}")

        if not math.isfinite(numeric_score) or not 0 <= numeric_score <= 1:
            raise ValueError(
                f"Score must be finite and between 0 and 1 "
                f"for customer {customer_id}"
            )

        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"Invalid label for customer {customer_id}")

        expected_label = int(numeric_score >= 0.5)

        if label != expected_label:
            raise ValueError(
                f"Label does not match score for customer {customer_id}"
            )

    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Duplicate customer IDs in predictions")

    if set(actual_ids) != set(expected_ids):
        raise ValueError("Prediction customer IDs do not match input IDs")

    return predictions


def replace_day(conn, day, model_version, predictions):
    if not predictions:
        raise ValueError("Empty predictions")

    lock_key = f"{day}:{model_version}"

    with conn.transaction():
        with conn.cursor() as cursor:
            # Serialize publications for the same date and model version.
            cursor.execute(
                """
                SELECT pg_advisory_xact_lock(
                    hashtextextended(%s, 0)
                )
                """,
                (lock_key,),
            )

            # Remove the previous complete snapshot.
            cursor.execute(
                """
                DELETE FROM predictions
                WHERE business_date = %s
                  AND model_version = %s
                """,
                (day, model_version),
            )

            # Insert the new snapshot using parameterized SQL.
            cursor.executemany(
                """
                INSERT INTO predictions (
                    business_date,
                    model_version,
                    customer_id,
                    score,
                    label
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (
                        day,
                        model_version,
                        prediction["customer_id"],
                        prediction["score"],
                        prediction["label"],
                    )
                    for prediction in predictions
                ],
            )

    return len(predictions)
