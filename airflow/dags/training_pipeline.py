"""DAG d'entraînement automatisé."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
}

with DAG(
    dag_id="training_pipeline",
    default_args=default_args,
    description="Pipeline d'entraînement ML automatisé",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training"],
) as dag:

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="cd /opt/airflow && dvc repro prepare",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow && dvc repro train",
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command="cd /opt/airflow && dvc repro evaluate",
    )

    check_drift = BashOperator(
        task_id="check_drift",
        bash_command="cd /opt/airflow && python monitoring/evidently/drift_report.py",
    )

    prepare_data >> train_model >> evaluate_model >> check_drift
