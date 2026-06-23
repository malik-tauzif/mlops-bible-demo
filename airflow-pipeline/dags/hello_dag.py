from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# This function is what ONE task will actually run
def say_hello():
    print("✅ Hello from inside an Airflow task!")

# This is the DAG itself — the "race official's clipboard"
with DAG(
    dag_id="hello_world_dag",
    start_date=datetime(2026, 1, 1),
    schedule=None,   # we'll trigger it manually for now, not on a timer
    catchup=False,
) as dag:

    task1 = PythonOperator(
        task_id="say_hello_task",
        python_callable=say_hello,
    )
