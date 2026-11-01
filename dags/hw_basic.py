
from datetime import timedelta

import pendulum
import requests

from airflow.sdk import DAG, Param, task, get_current_context, BaseHook
from airflow.providers.postgres.hooks.postgres import PostgresHook

from batch_common import (
    business_date,
    digest,
    save,
    read,
    validate_rows,
    infer_batches,
    validate_predictions,
    replace_day,
)


with DAG(
    dag_id="hw_basic_batch_inference",
    description="Daily batch inference of customer churn model",
    start_date=pendulum.datetime(2026, 9, 1, tz="Europe/Moscow"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["homework", "basic"],
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=5),
        "execution_timeout": timedelta(minutes=2),
    },
    params={
        "business_date": Param(
            "2026-09-01",
            type="string",
            format="date",
        ),
        "batch_size": Param(
            32,
            type="integer",
            minimum=1,
            maximum=1000,
        ),
    },
) as dag:

    @task(task_id="extract")
    def extract():
        context = get_current_context()

        params = context["params"]
        interval_start = context.get("data_interval_start")

        day = business_date(
            params.get("business_date"),
            interval_start,
        )

        # Подключаемся к API
        source_connection = BaseHook.get_connection("source_api")
        base_url = source_connection.host.rstrip("/")

        # Качаем данные
        rows_response = requests.get(
            f"{base_url}/basic/{day}.json",
            timeout=10,
        )
        rows_response.raise_for_status()

        rows_payload = rows_response.json()

        if rows_payload["business_date"] != day:
            raise ValueError(
                f"Source returned business_date "
                f"{rows_payload['business_date']}, expected {day}"
            )

        rows = rows_payload["rows"]

        # Скачиваем модельку
        model_response = requests.get(
            f"{base_url}/models/v1.json",
            timeout=10,
        )
        model_response.raise_for_status()

        # Хэшируем ответ
        model_sha256 = digest(model_response.content)
        model = model_response.json()

        # СОздаём артефакт
        artifact_ref = save(
            {
                "business_date": day,
                "rows": rows,
                "model": model,
                "model_sha256": model_sha256,
            }
        )

        return artifact_ref

    @task(
        task_id="infer",
        retries=0,
    )
    def infer(extract_ref):
        context = get_current_context()
        batch_size = context["params"]["batch_size"]

        # Загружаем созданный артефакт
        artifact = read(extract_ref)

        day = artifact["business_date"]
        rows = artifact["rows"]
        model = artifact["model"]
        model_sha256 = artifact["model_sha256"]

        # Валидируем строки
        validated_rows = validate_rows(rows, day)

        # batch inference
        predictions = infer_batches(
            validated_rows,
            model,
            batch_size,
        )

        # Убеждаемся в детерминорованности предсказаний
        expected_ids = [
            row["customer_id"]
            for row in validated_rows
        ]

        validate_predictions(
            predictions,
            expected_ids,
        )

        # Сохраняем предсказания
        result_ref = save(
            {
                "business_date": day,
                "model_version": "v1",
                "predictions": predictions,
                "model_sha256": model_sha256,
            }
        )

        return result_ref

    @task(task_id="publish")
    def publish(infer_ref):
        # Загружаем результаты инференса
        result = read(infer_ref)

        day = result["business_date"]
        model_version = result["model_version"]
        predictions = result["predictions"]

        # Подключаемся к БД
        warehouse_hook = PostgresHook(
            postgres_conn_id="warehouse"
        )
        conn = warehouse_hook.get_conn()

        try:
            published_rows = replace_day(
                conn,
                day,
                model_version,
                predictions,
            )
        finally:
            conn.close()

        return {
            "business_date": day,
            "model_version": model_version,
            "rows": published_rows,
        }

    extract_result = extract()
    inference_result = infer(extract_result)
    publish(inference_result)
