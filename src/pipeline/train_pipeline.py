import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import mlflow
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.logger.logger import logging
from src.exception import CustomException
from src.config.mlflow_config import MLflowConfig

def run_training_pipeline(use_hyperparameter_tuning=False, tuning_method='random', cv_folds=3, n_iter=20):
    """
    Run the complete training pipeline with optional hyperparameter tuning
    
    Args:
        use_hyperparameter_tuning (bool): Enable hyperparameter tuning
        tuning_method (str): 'random' for RandomizedSearchCV or 'grid' for GridSearchCV
        cv_folds (int): Number of cross-validation folds
        n_iter (int): Number of iterations for RandomizedSearchCV
    """
    try:
        # Setup MLflow
        MLflowConfig.setup_mlflow()
        
        # Start main training run
        mlflow.start_run(run_name="full_training_pipeline")
        mlflow.log_param("use_hyperparameter_tuning", use_hyperparameter_tuning)
        mlflow.log_param("tuning_method", tuning_method)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("n_iter", n_iter)
        
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
        if use_hyperparameter_tuning:
            logging.info(f"⚙️  Hyperparameter Tuning: ENABLED")
            logging.info(f"  Method: {tuning_method.upper()}")
            logging.info(f"  CV Folds: {cv_folds}")
            if tuning_method == 'random':
                logging.info(f"  Iterations: {n_iter}")
        else:
            logging.info(f"⚙️  Hyperparameter Tuning: DISABLED")
        
        model_trainer = ModelTrainer()
        
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
        
        # End MLflow run
        mlflow.end_run()
        
        return {
            'train_data': train_data_path,
            'test_data': test_data_path,
            'preprocessor': preprocessor_path,
            'model': model_path,
            'status': 'success'
        }
        
    except Exception as e:
        logging.error(f"❌ Training pipeline failed: {str(e)}")
        mlflow.end_run()
        raise CustomException(e, sys)

if __name__ == "__main__":
    # Option 1: Train WITHOUT hyperparameter tuning (Fast)
    # result = run_training_pipeline(use_hyperparameter_tuning=False)
    
    # Option 2: Train WITH RandomizedSearchCV (Recommended - Balanced)
    result = run_training_pipeline(
        use_hyperparameter_tuning=True,
        tuning_method='random',
        cv_folds=3,
        n_iter=20
    )
    
    # Option 3: Train WITH GridSearchCV (Slow but thorough)
    # result = run_training_pipeline(
    #     use_hyperparameter_tuning=True,
    #     tuning_method='grid',
    #     cv_folds=5
    # )
    
    print("\n✓ Pipeline completed!")
    print(f"Model saved at: {result['model']}")
