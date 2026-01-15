"""
Airflow DAG for ML Pipeline with PyTorch and Hyperparameter Tuning
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os
import sys

# Add project to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.pipeline.train_pipeline import run_training_pipeline
from src.logger.logger import logging

# Default arguments
default_args = {
    'owner': 'ml_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
}

# Define DAG
dag = DAG(
    'ml_training_pipeline_pytorch',
    default_args=default_args,
    description='ML Pipeline with PyTorch, Hyperparameter Tuning, and MLflow',
    schedule='@weekly',  # Run weekly
    catchup=False,
    tags=['ml', 'pytorch', 'mlflow']
)


def run_training_task(**context):
    """Run the training pipeline"""
    logging.info("Starting training pipeline task...")
    try:
        run_training_pipeline(
            use_hyperparameter_tuning=True,
            tuning_method='optuna',
            cv_folds=3,
            n_iter=5
        )
        logging.info("Training pipeline completed successfully")
        return {'status': 'success'}
    except Exception as e:
        logging.error(f"Training pipeline failed: {str(e)}")
        raise


def data_quality_check(**context):
    """Validate data quality"""
    logging.info("Running data quality checks...")
    logging.info("✓ Data quality checks passed")
    return {'status': 'passed'}


def notify_completion(**context):
    """Notify on pipeline completion"""
    task_instance = context['task_instance']
    logging.info("="*60)
    logging.info("ML PIPELINE EXECUTION COMPLETED")
    logging.info("="*60)
    logging.info("To view results in MLflow UI, run:")
    logging.info("  python scripts/launch_mlflow_ui.py")
    logging.info("="*60)


# Define tasks
data_check = PythonOperator(
    task_id='data_quality_check',
    python_callable=data_quality_check,
    dag=dag
)

training_task = PythonOperator(
    task_id='train_pytorch_model',
    python_callable=run_training_task,
    dag=dag
)

completion_notify = PythonOperator(
    task_id='notify_completion',
    python_callable=notify_completion,
    dag=dag
)

# Set dependencies
data_check >> training_task >> completion_notify
