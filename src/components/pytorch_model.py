import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from dataclasses import dataclass
import mlflow
import mlflow.pytorch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.exception import CustomException
from src.logger.logger import logging
from src.utils import save_object


@dataclass
class PyTorchModelConfig:
    model_obj_file_path: str = os.path.join('artifacts', 'pytorch_model.pkl')
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_dim: int = 128
    dropout_rate: float = 0.2


class SimpleNN(nn.Module):
    """Simple Neural Network for binary classification"""
    
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class PyTorchModelTrainer:
    def __init__(self):
        self.config = PyTorchModelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
    
    def train_pytorch_model(self, X_train, y_train, X_test, y_test):
        """Train PyTorch neural network model"""
        try:
            logging.info("Starting PyTorch model training...")
            
            # Convert to numpy arrays with proper dtype
            if isinstance(X_train, pd.DataFrame):
                X_train_np = X_train.values.astype(np.float32)
            else:
                X_train_np = np.asarray(X_train, dtype=np.float32)
            
            if isinstance(y_train, pd.Series):
                y_train_np = y_train.values.reshape(-1, 1).astype(np.float32)
            else:
                y_train_np = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
            
            if isinstance(X_test, pd.DataFrame):
                X_test_np = X_test.values.astype(np.float32)
            else:
                X_test_np = np.asarray(X_test, dtype=np.float32)
            
            if isinstance(y_test, pd.Series):
                y_test_np = y_test.values.astype(np.float32)
            else:
                y_test_np = np.asarray(y_test, dtype=np.float32)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_np)
            y_train_tensor = torch.FloatTensor(y_train_np)
            X_test_tensor = torch.FloatTensor(X_test_np)
            y_test_tensor = torch.FloatTensor(y_test_np)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor.squeeze())
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor.squeeze())
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
            
            # Initialize model
            input_dim = X_train.shape[1]
            model = SimpleNN(input_dim, self.config.hidden_dim, self.config.dropout_rate)
            model.to(self.device)
            
            # Loss and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            
            # Training loop
            best_test_loss = float('inf')
            patience = 5
            patience_counter = 0
            
            logging.info(f"Training for {self.config.epochs} epochs...")
            
            for epoch in range(self.config.epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation phase
                model.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        test_loss += loss.item()
                
                test_loss /= len(test_loader)
                
                if (epoch + 1) % 5 == 0:
                    logging.info(f"Epoch {epoch+1}/{self.config.epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                
                # Early stopping
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Log metrics to MLflow
            mlflow.log_param("pytorch_epochs", self.config.epochs)
            mlflow.log_param("pytorch_batch_size", self.config.batch_size)
            mlflow.log_param("pytorch_learning_rate", self.config.learning_rate)
            mlflow.log_param("pytorch_hidden_dim", self.config.hidden_dim)
            mlflow.log_param("pytorch_dropout_rate", self.config.dropout_rate)
            mlflow.log_metric("pytorch_final_train_loss", train_loss)
            mlflow.log_metric("pytorch_final_test_loss", test_loss)
            
            logging.info("✓ PyTorch model training completed")
            
            return model
            
        except Exception as e:
            logging.error(f"Error in PyTorch model training: {str(e)}")
            raise CustomException(e, sys)
    
    def evaluate_pytorch_model(self, model, X_test, y_test):
        """Evaluate PyTorch model"""
        try:
            model.eval()
            
            # Convert to numpy with proper dtype
            if isinstance(X_test, pd.DataFrame):
                X_test_np = X_test.values.astype(np.float32)
            else:
                X_test_np = np.asarray(X_test, dtype=np.float32)
            
            if isinstance(y_test, pd.Series):
                y_test_np = y_test.values.astype(np.float32)
            else:
                y_test_np = np.asarray(y_test, dtype=np.float32)
            
            X_test_tensor = torch.FloatTensor(X_test_np)
            y_test_tensor = torch.FloatTensor(y_test_np)
            
            X_test_tensor = X_test_tensor.to(self.device)
            
            with torch.no_grad():
                predictions = model(X_test_tensor).cpu().numpy()
            
            predictions = (predictions > 0.5).astype(int).flatten()
            
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(y_test_tensor, predictions)
            f1 = f1_score(y_test_tensor, predictions)
            auc = roc_auc_score(y_test_tensor, predictions)
            
            metrics = {
                'pytorch_accuracy': accuracy,
                'pytorch_f1': f1,
                'pytorch_auc': auc
            }
            
            logging.info(f"PyTorch Model Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating PyTorch model: {str(e)}")
            raise CustomException(e, sys)
