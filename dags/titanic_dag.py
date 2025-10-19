from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

from src import model_development

def check_accuracy(**context):
    acc = context['ti'].xcom_pull(task_ids='evaluate_model')
    print(f"Model accuracy is {acc:.4f}. Checking against threshold...")

    if acc >= 0.75:
        return 'end_success'
    return 'end_failure'

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
    
    start = EmptyOperator(task_id='start')

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

    branch_task = BranchPythonOperator(
        task_id='check_accuracy_branch',
        python_callable=check_accuracy,
    )

    end_success = EmptyOperator(task_id='end_success')
    end_failure = EmptyOperator(task_id='end_failure')

    start >> preprocess_task >> train_task >> evaluate_task >> branch_task >> [end_success, end_failure]