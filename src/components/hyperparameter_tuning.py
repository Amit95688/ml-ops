import os
import sys
import optuna
from optuna.trial import Trial
import torch
import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.exception import CustomException
from src.logger.logger import logging
from src.components.pytorch_model import PyTorchModelTrainer, SimpleNN, PyTorchModelConfig
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim


class HyperparameterTuner:
    """Light hyperparameter tuning using Optuna for fast training"""
    
    def __init__(self, n_trials=5, timeout=300):
        """
        Args:
            n_trials: Number of trials for optimization (light tuning = 5 trials)
            timeout: Maximum time in seconds for optimization
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"HyperparameterTuner initialized with {n_trials} trials")
    
    def objective(self, trial: Trial, X_train, y_train, X_val, y_val):
        """Objective function for Optuna optimization"""
        try:
            # Suggest hyperparameters
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            epochs = 10  # Fixed for faster training
            
            logging.info(f"Trial {trial.number}: hidden_dim={hidden_dim}, lr={learning_rate:.5f}, dropout={dropout_rate:.3f}")
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train.values if isinstance(X_train, pd.DataFrame) else X_train)
            y_train_tensor = torch.FloatTensor(y_train.values.reshape(-1, 1) if isinstance(y_train, pd.Series) else y_train.reshape(-1, 1))
            X_val_tensor = torch.FloatTensor(X_val.values if isinstance(X_val, pd.DataFrame) else X_val)
            y_val_tensor = torch.FloatTensor(y_val.values.reshape(-1, 1) if isinstance(y_val, pd.Series) else y_val.reshape(-1, 1))
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Build model
            input_dim = X_train.shape[1]
            model = SimpleNN(input_dim, hidden_dim, dropout_rate)
            model.to(self.device)
            
            # Training setup
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Quick training
            for epoch in range(epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            logging.info(f"  Trial {trial.number} validation loss: {val_loss:.4f}")
            
            return val_loss
            
        except Exception as e:
            logging.error(f"Error in trial {trial.number}: {str(e)}")
            return float('inf')
    
    def tune(self, X_train, y_train, X_val, y_val):
        """Run hyperparameter tuning"""
        try:
            logging.info("="*60)
            logging.info("STARTING HYPERPARAMETER TUNING (Light - 5 trials)")
            logging.info("="*60)
            
            # Create study
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(
                direction='minimize',
                sampler=sampler
            )
            
            # Optimize
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True
            )
            
            # Log best trial
            best_trial = study.best_trial
            logging.info("\n" + "="*60)
            logging.info("HYPERPARAMETER TUNING COMPLETED")
            logging.info("="*60)
            logging.info(f"Best trial: {best_trial.number}")
            logging.info(f"Best value: {best_trial.value:.4f}")
            logging.info("Best hyperparameters:")
            for key, value in best_trial.params.items():
                logging.info(f"  {key}: {value}")
            
            # Log to MLflow
            mlflow.log_param("tuning_n_trials", self.n_trials)
            for key, value in best_trial.params.items():
                mlflow.log_param(f"best_{key}", value)
            mlflow.log_metric("best_validation_loss", best_trial.value)
            
            return best_trial.params
            
        except Exception as e:
            logging.error(f"Error in hyperparameter tuning: {str(e)}")
            raise CustomException(e, sys)
