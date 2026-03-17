"""Main ML pipeline DAG."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="ml_pipeline",
    description="End-to-end ML pipeline: ingest → features → train → evaluate → monitor",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training"],
    default_args=default_args,
) as dag:

    ingest = BashOperator(
        task_id="ingest_data",
        bash_command="python /opt/airflow/src/data/ingest.py",
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command="python /opt/airflow/src/features/build_features.py",
    )

    train = BashOperator(
        task_id="train_model",
        bash_command="python /opt/airflow/src/models/train.py",
    )

    evaluate = BashOperator(
        task_id="evaluate_model",
        bash_command="python /opt/airflow/src/models/evaluate.py",
    )

    monitor_drift = BashOperator(
        task_id="monitor_drift",
        bash_command="python /opt/airflow/src/monitoring/drift.py",
    )

    # Pipeline order
    ingest >> feature_engineering >> train >> evaluate >> monitor_drift
