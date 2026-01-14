import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.exception import CustomException
from src.logger.logger import logging


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        
        # Load model and preprocessor
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logging.info(f"Model loaded from {self.model_path}")
            else:
                logging.warning(f"Model not found at {self.model_path}. Train the model first.")
                self.model = None
            
            if os.path.exists(self.preprocessor_path):
                with open(self.preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
                logging.info(f"Preprocessor loaded from {self.preprocessor_path}")
            else:
                logging.warning(f"Preprocessor not found at {self.preprocessor_path}")
                self.preprocessor = None
                
        except Exception as e:
            logging.error(f"Error loading model/preprocessor: {str(e)}")
            self.model = None
            self.preprocessor = None
    
    def create_engineered_features(self, df):
        """Create the same engineered features as in data_transformation.py"""
        try:
            # Convert numeric columns to float
            numeric_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Feature engineering with trigonometric features
            logging.info("Creating trigonometric features")
            df['_duration_sin'] = np.sin(2*np.pi * df['duration'] / 540).astype('float32')
            df['_duration_cos'] = np.cos(2*np.pi * df['duration'] / 540).astype('float32')
            df['_balance_log'] = (np.sign(df['balance']) * np.log1p(np.abs(df['balance']))).astype('float32')
            df['_balance_sin'] = np.sin(2*np.pi * df['balance'] / 1000).astype('float32')
            df['_balance_cos'] = np.cos(2*np.pi * df['balance'] / 1000).astype('float32')
            df['_age_sin'] = np.sin(2*np.pi * df['age'] / 10).astype('float32')
            df['_pdays_sin'] = np.sin(2*np.pi * df['pdays'] / 7).astype('float32')
            
            # Interaction features
            logging.info("Creating interaction features")
            df['_age_balance'] = (df['age'] * df['balance']).astype('float32')
            df['_duration_campaign'] = (df['duration'] * df['campaign']).astype('float32')
            
            # Ratio features
            df['_prev_campaign_ratio'] = (df['previous'] / (df['campaign'] + 1)).astype('float32')
            
            # Binning features
            df['_age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype('int32')
            
            logging.info("Feature engineering completed")
            return df
            
        except Exception as e:
            logging.error(f"Error in feature engineering: {str(e)}")
            return df
    
    def predict(self, data_dict):
        """
        Make predictions from input data dictionary
        
        Args:
            data_dict: Dictionary with keys like 'age', 'balance', 'duration', etc.
        
        Returns:
            Prediction result as string
        """
        try:
            if self.model is None:
                return "⚠️ Model not trained yet. Please run data_ingestion.py to train the model."
            
            # Get column info from preprocessor
            numerical_cols = []
            categorical_cols = []
            
            if self.preprocessor is not None and isinstance(self.preprocessor, dict):
                numerical_cols = self.preprocessor.get('numerical_cols', [])
                categorical_cols = self.preprocessor.get('categorical_cols', [])
                
                # Add missing columns with default values
                for col in numerical_cols:
                    if col not in data_dict:
                        data_dict[col] = 0  # Default for numerical
                
                for col in categorical_cols:
                    if col not in data_dict:
                        data_dict[col] = 'unknown'  # Default for categorical
            
            # Convert input dictionary to DataFrame
            df = pd.DataFrame([data_dict])
            logging.info(f"Input data columns: {df.columns.tolist()}")
            logging.info(f"Input data shape before engineering: {df.shape}")
            
            # Create engineered features BEFORE processing
            df = self.create_engineered_features(df)
            logging.info(f"Input data shape after engineering: {df.shape}")
            
            # Apply preprocessing if available
            if self.preprocessor is not None:
                try:
                    # Handle preprocessor as dictionary (from data_transformation.py)
                    if isinstance(self.preprocessor, dict):
                        preprocessor_obj = self.preprocessor['preprocessor']
                        label_encoders = self.preprocessor.get('label_encoders', {})
                        numerical_cols = self.preprocessor.get('numerical_cols', [])
                        categorical_cols = self.preprocessor.get('categorical_cols', [])
                        
                        logging.info(f"Using preprocessing pipeline with {len(numerical_cols)} numerical and {len(categorical_cols)} categorical columns")
                        
                        # Ensure correct column order and types
                        # Reorder columns to match training order
                        all_cols = numerical_cols + categorical_cols
                        df = df[all_cols]
                        
                        # Convert numerical columns to numeric type
                        for col in numerical_cols:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        
                        # Keep categorical columns as strings/objects
                        for col in categorical_cols:
                            df[col] = df[col].astype(str)
                        
                        logging.info(f"Column types after conversion - Numerical: {df[numerical_cols].dtypes.tolist()}, Categorical: {df[categorical_cols].dtypes.tolist()}")
                        
                        # Transform the data
                        df_transformed = preprocessor_obj.transform(df)
                        logging.info(f"Data transformed. Shape: {df_transformed.shape}")
                        
                        # Label encode categorical columns
                        for i, col in enumerate(categorical_cols):
                            if col in label_encoders:
                                col_idx = len(numerical_cols) + i
                                try:
                                    df_transformed[:, col_idx] = label_encoders[col].transform(df_transformed[:, col_idx].astype(str))
                                except Exception as e:
                                    logging.warning(f"Could not encode column {col}: {str(e)}")
                    else:
                        # Handle preprocessor as a single object
                        df_transformed = self.preprocessor.transform(df)
                        logging.info(f"Data transformed using preprocessor. Shape: {df_transformed.shape}")
                        
                except Exception as e:
                    error_msg = f"Preprocessing error: {str(e)}"
                    logging.error(error_msg)
                    return f"❌ {error_msg}"
            else:
                logging.warning("No preprocessor found, using raw data")
                df_transformed = df.values
            
            # Make prediction
            prediction = self.model.predict(df_transformed)
            
            # Format prediction result
            try:
                pred_proba = self.model.predict_proba(df_transformed)
                confidence = max(pred_proba[0]) * 100
                result = f"✅ Prediction: {int(prediction[0])} (Confidence: {confidence:.2f}%)"
            except:
                result = f"✅ Prediction: {int(prediction[0])}"
            
            logging.info(f"Prediction result: {result}")
            return result
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logging.error(error_msg)
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, age, balance, duration, campaign, pdays, previous, job, marital):
        self.age = age
        self.balance = balance
        self.duration = duration
        self.campaign = campaign
        self.pdays = pdays
        self.previous = previous
        self.job = job
        self.marital = marital
    
    def get_data_as_dataframe(self):
        """Convert custom data to DataFrame"""
        try:
            custom_data_input_dict = {
                'age': [self.age],
                'balance': [self.balance],
                'duration': [self.duration],
                'campaign': [self.campaign],
                'pdays': [self.pdays],
                'previous': [self.previous],
                'job': [self.job],
                'marital': [self.marital],
            }
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
