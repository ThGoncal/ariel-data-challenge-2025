"""Main ML pipeline DAG."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_pipeline",
    default_args=default_args,
    description="Full ML pipeline: ingest → train → evaluate → deploy",
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops", "training"],
) as dag:

    ingest = BashOperator(
        task_id="ingest_data",
        bash_command="python /opt/airflow/src/data/ingest.py",
    )

    train = BashOperator(
        task_id="train_model",
        bash_command="python /opt/airflow/src/models/train.py",
    )

    monitor = BashOperator(
        task_id="monitor_drift",
        bash_command="python /opt/airflow/src/monitoring/drift.py",
    )

    ingest >> train >> monitor
