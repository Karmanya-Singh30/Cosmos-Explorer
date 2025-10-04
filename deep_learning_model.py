import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ExoplanetDeepLearningModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['period', 'planet_radius', 'star_teff', 'star_radius', 'impact']
        self.training_results = {}
        
    def load_and_clean_data(self, file_path):
        """Load and preprocess data from CSV file"""
        try:
            # Define column mappings for different data sources
            KEPLER_MAP = {
                'koi_disposition': 'disposition', 'koi_period': 'period', 'koi_prad': 'planet_radius', 
                'koi_steff': 'star_teff', 'koi_srad': 'star_radius', 'koi_impact': 'impact'
            }
            TESS_MAP = {
                'tfopwg_disp': 'disposition', 'pl_orbper': 'period', 'pl_rade': 'planet_radius', 
                'st_teff': 'star_teff', 'st_rad': 'star_radius', 'pl_imppar': 'impact'
            }
            K2_MAP = {
                'disposition': 'disposition', 'pl_orbper': 'period', 'pl_rade': 'planet_radius', 
                'st_teff': 'star_teff', 'st_rad': 'star_radius', 'pl_imppar': 'impact'
            }
            
            # Determine which mapping to use based on filename
            if 'kepler' in file_path.lower():
                column_map = KEPLER_MAP
                skiprows = 53
            elif 'toi' in file_path.lower() or 'tess' in file_path.lower():
                column_map = TESS_MAP
                skiprows = 69
            elif 'k2' in file_path.lower():
                column_map = K2_MAP
                skiprows = 98
            else:
                # Default case - try to use all mappings
                column_map = {**KEPLER_MAP, **TESS_MAP, **K2_MAP}
                skiprows = 0
            
            # Load data
            df = pd.read_csv(file_path, skiprows=skiprows)
            
            # Map columns
            existing_cols_map = {old: new for old, new in column_map.items() if old in df.columns}
            
            if not existing_cols_map:
                print(f"Warning: Could not find any required columns in {file_path}")
                return pd.DataFrame()

            df = df[list(existing_cols_map.keys())].copy()
            df.rename(columns=existing_cols_map, inplace=True)
            
            return df
            
        except FileNotFoundError:
            print(f"Error: {file_path} not found.")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path}: {e}")
            return pd.DataFrame()

    def preprocess_data(self, df):
        """Preprocess the data for training"""
        # Map disposition to binary target
        def map_disposition_to_binary(disposition):
            if pd.isna(disposition):
                return np.nan
            disposition = str(disposition).upper()
            if 'CONFIRMED' in disposition or 'CANDIDATE' in disposition or 'PC' in disposition or 'KP' in disposition:
                return 1
            elif 'FALSE POSITIVE' in disposition or 'FP' in disposition:
                return 0
            else:
                return np.nan

        df['is_exoplanet'] = df['disposition'].apply(map_disposition_to_binary)
        df.dropna(subset=['is_exoplanet'], inplace=True)
        
        # Check which feature columns are available
        available_features = [col for col in self.feature_columns if col in df.columns]
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        
        if missing_features:
            print(f"Warning: Missing feature columns: {missing_features}")
            print(f"Using available features: {available_features}")
            
            # If we don't have enough features, return empty
            if len(available_features) < 3:
                print("Error: Not enough feature columns available for training")
                return pd.DataFrame(), pd.Series()
        
        # Select features and target
        X = df[available_features]
        y = df['is_exoplanet']
        
        # Convert to numeric and handle missing values
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col].fillna(X[col].mean(), inplace=True)
            
        # If we're missing some features, we need to adjust our feature columns
        if missing_features:
            self.feature_columns = available_features
            
        return X, y

    def build_model(self, input_shape):
        """Build the deep learning model architecture"""
        model = Sequential([
            # Input Layer + First Hidden Layer
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),  # Helps prevent overfitting

            # Second Hidden Layer
            Dense(32, activation='relu'),
            Dropout(0.3),

            # Third Hidden Layer
            Dense(16, activation='relu'),

            # Output Layer for Binary Classification (1 neuron, sigmoid activation)
            Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, X, y, epochs=100, batch_size=64, validation_split=0.3):
        """Train the deep learning model"""
        # Check if we have data
        if X.empty or y.empty:
            raise ValueError("No data available for training")
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        self.model = self.build_model(X_train_scaled.shape[1])
        
        print("Model Architecture:")
        self.model.summary()
        
        # Train model
        print("\n--- Starting Model Training ---")
        history = self.model.fit(
            X_train_scaled,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        print("--- Training Complete ---")
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        # Store results
        self.training_results = {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'loss': round(loss, 4)
        }
        
        # Print classification report
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred, target_names=['Not Exoplanet (0)', 'Exoplanet (1)']))
        
        return history, X_test_scaled, y_test, y_pred

    def get_training_results(self):
        """Return the training results"""
        return self.training_results

    def save_model(self, model_path='models/deep_learning_model.h5', scaler_path='models/scaler.pkl'):
        """Save the trained model and scaler"""
        if self.model is not None:
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model and scaler
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            print(f"Model saved to {model_path}")
            print(f"Scaler saved to {scaler_path}")
        else:
            print("No model to save. Train the model first.")

    def load_model(self, model_path='models/deep_learning_model.h5', scaler_path='models/scaler.pkl'):
        """Load a trained model and scaler"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Model loaded from {model_path}")
            print(f"Scaler loaded from {scaler_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Train or load a model first.")
        
        # Check if we have the right columns
        available_features = [col for col in self.feature_columns if col in X.columns]
        if len(available_features) != len(self.feature_columns):
            raise ValueError(f"Missing feature columns. Expected: {self.feature_columns}, Available: {available_features}")
        
        # Select only the required features
        X = X[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred_proba = self.model.predict(X_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return y_pred, y_pred_proba

# Example usage function
def train_model_from_files(file_paths):
    """Train the model using multiple CSV files"""
    model = ExoplanetDeepLearningModel()
    
    # Load and combine data from all files
    all_dataframes = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            df = model.load_and_clean_data(file_path)
            if not df.empty:
                all_dataframes.append(df)
                print(f"Loaded {len(df)} rows from {file_path}")
            else:
                print(f"No data loaded from {file_path}")
        else:
            print(f"File not found: {file_path}")
    
    if not all_dataframes:
        print("No data loaded from any files.")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n--- Data Consolidation Complete. Total rows: {combined_df.shape[0]} ---")
    
    # Preprocess data
    X, y = model.preprocess_data(combined_df)
    if X.empty or y.empty:
        print("Failed to preprocess data.")
        return None
        
    print(f"Preprocessed data shape: {X.shape}")
    
    # Train model
    try:
        history, X_test, y_test, y_pred = model.train(X, y, epochs=50)  # Reduced epochs for demo
        
        # Save model
        model.save_model()
        
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    # file_paths = ['kepler_df.csv', 'toi_df.csv', 'k2_df.csv']
    # model = train_model_from_files(file_paths)
    pass