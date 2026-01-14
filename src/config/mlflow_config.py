"""
MLflow configuration module for experiment tracking and model registry
"""
import os
import mlflow
from mlflow.tracking import MlflowClient

class MLflowConfig:
    """MLflow configuration and setup"""
    
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
    EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'ml_project_training')
    REGISTRY_URI = os.getenv('MLFLOW_REGISTRY_URI', 'file:./mlruns')
    
    @staticmethod
    def setup_mlflow():
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(MLflowConfig.MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri(MLflowConfig.REGISTRY_URI)
        
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(MLflowConfig.EXPERIMENT_NAME)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            client = MlflowClient(MLflowConfig.MLFLOW_TRACKING_URI)
            experiment = client.get_experiment_by_name(MLflowConfig.EXPERIMENT_NAME)
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(MLflowConfig.EXPERIMENT_NAME)
        return experiment_id
    
    @staticmethod
    def start_run(run_name=None, tags=None):
        """Start an MLflow run"""
        mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)
    
    @staticmethod
    def log_params(params):
        """Log parameters to MLflow"""
        mlflow.log_params(params)
    
    @staticmethod
    def log_metrics(metrics):
        """Log metrics to MLflow"""
        mlflow.log_metrics(metrics)
    
    @staticmethod
    def log_model(model, artifact_path, model_type="sklearn"):
        """Log model to MLflow"""
        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path)
        elif model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path)
        else:
            mlflow.sklearn.log_model(model, artifact_path)
    
    @staticmethod
    def log_artifact(file_path):
        """Log artifact to MLflow"""
        mlflow.log_artifact(file_path)
    
    @staticmethod
    def end_run():
        """End current MLflow run"""
        mlflow.end_run()
