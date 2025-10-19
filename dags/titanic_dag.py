from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from src import model_development

# Default arguments for the DAG
default_args = {
    'owner': 'Arpita Wagulde',
    'start_date': datetime(2025, 1, 18),
    'retries': 0,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='titanic_pipeline',
    default_args=default_args,
    description='Titanic pipeline using PythonOperator',
    catchup=False
) as dag:

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=model_development.preprocess_data,
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=model_development.train_model,
    )

    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=model_development.evaluate_model,
    )

    preprocess_task >> train_task >> evaluate_task