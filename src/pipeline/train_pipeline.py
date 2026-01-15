import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import mlflow
import matplotlib.pyplot as plt
import numpy as np
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.pytorch_model import PyTorchModelTrainer
from src.components.hyperparameter_tuning import HyperparameterTuner
from src.logger.logger import logging
from src.exception import CustomException
from src.config.mlflow_config import MLflowConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import pandas as pd

def run_training_pipeline(use_hyperparameter_tuning=False, tuning_method='random', cv_folds=3, n_iter=5, use_pytorch=False):
    """
    Run the complete training pipeline with optional hyperparameter tuning
    
    Args:
        use_hyperparameter_tuning (bool): Enable hyperparameter tuning
        tuning_method (str): 'random', 'grid', or 'optuna' for tuning method
        cv_folds (int): Number of cross-validation folds
        n_iter (int): Number of iterations for RandomizedSearchCV / Optuna trials
        use_pytorch (bool): Use PyTorch neural network instead of traditional models
    """
    try:
        # Setup MLflow
        MLflowConfig.setup_mlflow()
        
        # Don't start a main run - let child runs be independent
        # mlflow.start_run(run_name="full_training_pipeline")
        
        logging.info("="*60)
        logging.info("TRAINING PIPELINE STARTED")
        logging.info("="*60)
        
        # Step 1: Data Ingestion
        logging.info("Step 1: Data Ingestion")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"✓ Data ingestion completed")
        logging.info(f"  Train path: {train_data_path}")
        logging.info(f"  Test path: {test_data_path}")
        mlflow.log_param("train_data_path", train_data_path)
        mlflow.log_param("test_data_path", test_data_path)
        
        # Step 2: Data Transformation
        logging.info("\nStep 2: Data Transformation")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info(f"✓ Data transformation completed")
        logging.info(f"  Train shape: {train_arr.shape}")
        logging.info(f"  Test shape: {test_arr.shape}")
        logging.info(f"  Preprocessor saved: {preprocessor_path}")
        mlflow.log_param("preprocessor_path", preprocessor_path)
        mlflow.log_metric("train_samples", train_arr.shape[0])
        mlflow.log_metric("test_samples", test_arr.shape[0])
        mlflow.log_metric("num_features", train_arr.shape[1])
        
        # Step 3: Model Training with Hyperparameter Tuning
        logging.info("\nStep 3: Model Training")
        if use_pytorch:
            logging.info(f"🔥 Using PyTorch Neural Network")
        else:
            logging.info(f"📊 Using Traditional ML Models (XGBoost, LightGBM)")
            
        if use_hyperparameter_tuning:
            logging.info(f"⚙️  Hyperparameter Tuning: ENABLED")
            logging.info(f"  Method: {tuning_method.upper()}")
            logging.info(f"  CV Folds: {cv_folds}")
            if tuning_method in ['random', 'optuna']:
                logging.info(f"  Iterations/Trials: {n_iter}")
        else:
            logging.info(f"⚙️  Hyperparameter Tuning: DISABLED")
        
        mlflow.log_param("use_pytorch", use_pytorch)
        
        # Train with PyTorch or traditional models
        if use_pytorch:
            # Convert to DataFrames for PyTorch trainer
            X_train = pd.DataFrame(train_arr[:, :-1])
            y_train = pd.Series(train_arr[:, -1])
            X_test = pd.DataFrame(test_arr[:, :-1])
            y_test = pd.Series(test_arr[:, -1])
            
            # Hyperparameter tuning with Optuna
            if use_hyperparameter_tuning and tuning_method == 'optuna':
                # Split training data for validation
                X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                
                # Tune hyperparameters
                tuner = HyperparameterTuner(n_trials=n_iter, timeout=600)
                best_params = tuner.tune(X_tr, y_tr, X_val, y_val)
                
                # Train with best parameters
                pytorch_trainer = PyTorchModelTrainer()
                pytorch_trainer.config.hidden_dim = best_params['hidden_dim']
                pytorch_trainer.config.learning_rate = best_params['learning_rate']
                pytorch_trainer.config.dropout_rate = best_params['dropout_rate']
                pytorch_trainer.config.batch_size = best_params['batch_size']
            else:
                pytorch_trainer = PyTorchModelTrainer()
            
            # Train PyTorch model
            pytorch_model = pytorch_trainer.train_pytorch_model(X_train, y_train, X_test, y_test)
            
            # Evaluate
            metrics = pytorch_trainer.evaluate_pytorch_model(pytorch_model, X_test, y_test)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            model_path = "artifacts/pytorch_model.pt"
            import torch
            torch.save(pytorch_model.state_dict(), model_path)
            logging.info(f"✓ PyTorch model training completed")
            logging.info(f"  Model saved: {model_path}")
            mlflow.log_param("model_path", model_path)
        
        else:
            # Traditional model training
            model_trainer = ModelTrainer()
            
            # Configure hyperparameter tuning
            model_trainer.model_trainer_config.use_hyperparameter_tuning = use_hyperparameter_tuning
            model_trainer.model_trainer_config.tuning_method = tuning_method
            # Configure hyperparameter tuning
            model_trainer.model_trainer_config.use_hyperparameter_tuning = use_hyperparameter_tuning
            model_trainer.model_trainer_config.tuning_method = tuning_method
            model_trainer.model_trainer_config.cv_folds = cv_folds
            model_trainer.model_trainer_config.n_iter = n_iter
            
            model_path = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"✓ Model training completed")
            logging.info(f"  Best model saved: {model_path}")
            mlflow.log_param("model_path", model_path)
        
        logging.info("\n" + "="*60)
        logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("="*60)
        logging.info("\n📊 NEXT STEPS:")
        logging.info("="*60)
        logging.info("To view training results in MLflow UI:")
        logging.info("  python scripts/launch_mlflow_ui.py")
        logging.info("\nOr run:")
        logging.info("  bash scripts/launch_mlflow.sh")
        logging.info("="*60)
        
        logging.info("\n" + "="*60)
        logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("="*60)
        logging.info("\n📊 NEXT STEPS:")
        logging.info("="*60)
        logging.info("To view training results in MLflow UI:")
        logging.info("  python scripts/launch_mlflow_ui.py")
        logging.info("\nOr run:")
        logging.info("  bash scripts/launch_mlflow.sh")
        logging.info("="*60)
        
        return {
            'train_data': train_data_path,
            'test_data': test_data_path,
            'preprocessor': preprocessor_path,
            'model': model_path,
            'status': 'success'
        }
        
    except Exception as e:
        logging.error(f"❌ Training pipeline failed: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    # ⭐ TRAIN XGBoost/LightGBM WITH VISUALIZATIONS
    logging.info("Training XGBoost and LightGBM with visualizations...")
    
    result = run_training_pipeline(
        use_hyperparameter_tuning=False,
        use_pytorch=False
    )
    
    print("\n✓ Models trained with visualizations!")
    print(f"Best model saved at: {result['model']}")
    print("\nMLflow UI will show:")
    print("  ✓ Confusion matrices for each model")
    print("  ✓ ROC curves for each model")
    print("  ✓ Metrics (AUC, Accuracy, F1)")
    print("\nTo view all results and graphs:")
    print("  python scripts/launch_mlflow_ui.py")
    print("  🌐 Opens at: http://localhost:5000")

