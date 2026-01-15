import os
import sys
from dataclasses import dataclass
import xgboost as xgb
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.sklearn
import matplotlib.pyplot as plt


# Add project root to sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from src.exception import CustomException
from src.logger.logger import logging
from src.utils import save_object
from src.config.mlflow_config import MLflowConfig
from src.utils_viz import plot_confusion_matrix, plot_roc_curve
# no direct dependency on DataTransformation artifacts required here

@dataclass
class ModelTrainerConfig:
    model_obj_file_path: str = os.path.join('artifacts', 'model.pkl')
    use_hyperparameter_tuning: bool = False
    tuning_method: str = 'random'  # 'grid' or 'random'
    cv_folds: int = 2  
    n_iter: int = 3  
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def _tune_xgboost(self, X_train, y_train):
        """Hyperparameter tuning for XGBoost using RandomizedSearchCV - !"""
        logging.info("Starting XGBoost hyperparameter tuning...")
        
        # MINIMAL PARAMS FOR SPEED
        param_dist = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
        }
        
        base_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        
        if self.model_trainer_config.tuning_method == 'grid':
            search = GridSearchCV(
                base_model, 
                param_dist, 
                cv=self.model_trainer_config.cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
        else:  # random
            search = RandomizedSearchCV(
                base_model, 
                param_dist, 
                n_iter=self.model_trainer_config.n_iter,
                cv=self.model_trainer_config.cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        
        search.fit(X_train, y_train)
        logging.info(f"Best XGBoost params: {search.best_params_}")
        logging.info(f"Best XGBoost CV score: {search.best_score_:.4f}")
        return search.best_estimator_

    def _tune_lightgbm(self, X_train, y_train):
        """Hyperparameter tuning for LightGBM using RandomizedSearchCV - ULTRA FAST!"""
        logging.info("Starting LightGBM hyperparameter tuning (FAST mode - 3 iterations only)...")
        
        # MINIMAL PARAMS FOR SPEED
        param_dist = {
            'n_estimators': [50, 100],
            'num_leaves': [20, 31],
            'learning_rate': [0.05, 0.1],
        }
        
        base_model = lgbm.LGBMClassifier(random_state=42, verbose=-1)
        
        if self.model_trainer_config.tuning_method == 'grid':
            search = GridSearchCV(
                base_model, 
                param_dist, 
                cv=self.model_trainer_config.cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
        else:  # random
            search = RandomizedSearchCV(
                base_model, 
                param_dist, 
                n_iter=self.model_trainer_config.n_iter,
                cv=self.model_trainer_config.cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        
        search.fit(X_train, y_train)
        logging.info(f"Best LightGBM params: {search.best_params_}")
        logging.info(f"Best LightGBM CV score: {search.best_score_:.4f}")
        return search.best_estimator_

    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Model Trainer method starts")
        try:
            model_scores = []
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # ensure arrays
            X_train = np.asarray(X_train, dtype=np.float32)
            X_test = np.asarray(X_test, dtype=np.float32)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)

            # FAST BASELINE MODELS - optimized for speed
            models = [
                ("XGB", xgb.XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, eval_metric='logloss', tree_method='hist', random_state=42)),
                ("LGBM", lgbm.LGBMClassifier(n_estimators=50, num_leaves=31, learning_rate=0.1, verbose=-1, random_state=42)),
            ]

            trained_models = {}

            # normalize/validate y to binary 0/1 (support 'yes'/'no' or 0/1)
            def _convert_y(arr):
                s = pd.Series(arr)
                # try numeric conversion first
                num = pd.to_numeric(s, errors='coerce')
                if num.notna().all():
                    return num.astype(int).to_numpy()
                # map common string labels
                mapped = s.astype(str).str.strip().str.lower().map({
                    'yes': 1, 'y': 1, 'true': 1, '1': 1,
                    'no': 0, 'n': 0, 'false': 0, '0': 0
                })
                # fallback to numeric where possible
                mapped = mapped.fillna(pd.to_numeric(s, errors='coerce'))
                mapped = mapped.fillna(0).astype(int)
                return mapped.to_numpy()

            y_train_proc = _convert_y(y_train)
            y_test_proc = _convert_y(y_test)

            # If training labels contain only one class, abort with helpful message
            if len(np.unique(y_train_proc)) < 2:
                raise CustomException(f"Training labels contain a single class: {np.unique(y_train_proc)}. Cannot train binary classifier.", sys)

            for model_name, model in models:
                logging.info(f"Training {model_name}")

                try:
                    # Start MLflow run for this model
                    mlflow.start_run(run_name=f"{model_name}_training")
                    
                    # Log model name
                    mlflow.set_tag("model_type", model_name)
                    mlflow.log_param("use_hyperparameter_tuning", self.model_trainer_config.use_hyperparameter_tuning)
                    
                    # Apply hyperparameter tuning if enabled
                    if self.model_trainer_config.use_hyperparameter_tuning:
                        mlflow.log_param("tuning_method", self.model_trainer_config.tuning_method)
                        mlflow.log_param("cv_folds", self.model_trainer_config.cv_folds)
                        mlflow.log_param("n_iter", self.model_trainer_config.n_iter)
                        
                        if model_name == "XGB":
                            model = self._tune_xgboost(X_train, y_train_proc)
                        elif model_name == "LGBM":
                            model = self._tune_lightgbm(X_train, y_train_proc)
                        
                        # Log best parameters from tuning
                        if hasattr(model, 'get_params'):
                            mlflow.log_params(model.get_params())
                    else:
                        model.fit(X_train, y_train_proc)
                        
                except Exception as me:
                    logging.error(f"Training failed for {model_name}: {me}")
                    mlflow.end_run()
                    # skip this model on failure
                    continue

                if hasattr(model, "predict_proba"):
                    y_pred_probs = model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_probs = model.predict(X_test)
                y_pred = model.predict(X_test)

                try:
                    auc = float(roc_auc_score(y_test_proc, y_pred_probs))
                except Exception:
                    auc = None
                acc = float(accuracy_score(y_test_proc, y_pred))
                f1 = float(f1_score(y_test_proc, y_pred))
                trained_models[model_name] = model
                
                # Log metrics to MLflow
                mlflow.log_metric("auc", auc if auc is not None else 0.0)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                
                # Log visualizations to MLflow
                try:
                    plot_confusion_matrix(y_test_proc, y_pred, model_name)
                    logging.info(f"✓ Logged confusion matrix for {model_name}")
                except Exception as e:
                    logging.warning(f"Could not plot confusion matrix: {e}")
                
                try:
                    plot_roc_curve(y_test_proc, y_pred_probs, model_name)
                    logging.info(f"✓ Logged ROC curve for {model_name}")
                except Exception as e:
                    logging.warning(f"Could not plot ROC curve: {e}")
                
                # Log model to MLflow
                if model_name == "XGB":
                    mlflow.xgboost.log_model(model, artifact_path=f"model_{model_name}")
                elif model_name == "LGBM":
                    mlflow.lightgbm.log_model(model, artifact_path=f"model_{model_name}")

                # use -1.0 for missing AUC so sorting works
                auc_for_sort = auc if auc is not None else -1.0
                model_scores.append({
                    "name": model_name,
                    "auc": float(auc_for_sort),
                    "accuracy": float(acc),
                    "f1": float(f1),
                    "raw_auc": auc
                })

                logging.info(f"{model_name} AUC: {auc}, Accuracy: {acc}, F1: {f1}")
                mlflow.end_run()

            # select best model by AUC
            best = sorted(model_scores, key=lambda x: x["auc"], reverse=True)[0]
            best_name = best["name"]
            best_model = trained_models[best_name]

            logging.info(f"Best model: {best_name} with AUC: {best['auc']}")

            # save best model
            save_path = os.path.join(os.path.dirname(self.model_trainer_config.model_obj_file_path), f"model.pkl")
            save_object(file_path=save_path, obj=best_model)
            logging.info(f"Saved best model to {save_path}")

            logging.info("Model Trainer method ends")
            return save_path

        except Exception as e:
            raise CustomException(e, sys)