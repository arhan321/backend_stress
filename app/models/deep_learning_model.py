import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from datetime import datetime
from loguru import logger
import json

class StressDeepLearningModel:
    """
    Deep Learning model for employee stress prediction using TensorFlow
    
    This model predicts stress levels (Low, Medium, High) based on:
    - Workload
    - Work-Life Balance
    - Team Conflict
    - Management Support
    - Work Environment
    - Department (categorical)
    """
    
    def __init__(self, model_version="1.0"):
        self.model_version = model_version
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.department_encoder = LabelEncoder()
        self.feature_names = [
            'workload', 'work_life_balance', 'team_conflict', 
            'management_support', 'work_environment', 'department_encoded'
        ]
        self.stress_categories = ['Low', 'Medium', 'High']  # 0-30, 30-70, 70-100
        self.model_path = f"models/deep_learning_v{model_version}"
        self.is_trained = False
        
    def _categorize_stress(self, stress_level):
        """Convert continuous stress level to categories"""
        if stress_level < 30:
            return 0  # Low
        elif stress_level < 70:
            return 1  # Medium
        else:
            return 2  # High
    
    def _prepare_features(self, data):
        """Prepare and encode features for training/prediction"""
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Encode department if it's categorical
        if 'department' in df.columns:
            # Fit encoder only during training
            if not hasattr(self, '_department_fitted'):
                self.department_encoder.fit(df['department'])
                self._department_fitted = True
            
            df['department_encoded'] = self.department_encoder.transform(df['department'])
        else:
            df['department_encoded'] = 0  # Default value
        
        # Select only the features we need
        feature_data = df[self.feature_names].copy()
        
        return feature_data
    
    def build_model(self, input_dim=6, hidden_layers=[128, 64, 32], dropout_rate=0.3):
        """
        Build the deep neural network architecture
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers with batch normalization and dropout
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001),
                name=f'hidden_{i+1}'
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer for 3-class classification (Low, Medium, High)
        model.add(layers.Dense(3, activation='softmax', name='output'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"Deep Learning model architecture built with {len(hidden_layers)} hidden layers")
        return model
    
    def train(self, training_data, validation_split=0.2, epochs=100, batch_size=32, early_stopping_patience=15):
        """
        Train the deep learning model
        
        Args:
            training_data: DataFrame with employee data
            validation_split: Fraction of data to use for validation
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            early_stopping_patience: Early stopping patience
        """
        try:
            logger.info("Starting Deep Learning model training...")
            
            # Prepare features
            X = self._prepare_features(training_data)
            
            # Prepare target variable (convert stress level to categories)
            y = training_data['stress_level'].apply(self._categorize_stress)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Build model if not already built
            if self.model is None:
                self.build_model(input_dim=X_train_scaled.shape[1])
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=0.0001
                )
            ]
            
            # Train model
            start_time = datetime.now()
            history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate model
            train_loss, train_accuracy = self.model.evaluate(X_train_scaled, y_train, verbose=0)
            val_loss, val_accuracy = self.model.evaluate(X_val_scaled, y_val, verbose=0)
            
            # Predictions for detailed metrics
            y_pred = self.model.predict(X_val_scaled)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            self.is_trained = True
            
            # Save model and encoders
            self.save_model()
            
            training_results = {
                'model_type': 'deep_learning',
                'model_version': self.model_version,
                'training_time_seconds': training_time,
                'train_accuracy': float(train_accuracy),
                'validation_accuracy': float(val_accuracy),
                'train_loss': float(train_loss),
                'validation_loss': float(val_loss),
                'training_data_size': len(training_data),
                'epochs_trained': len(history.history['loss']),
                'classification_report': classification_report(y_val, y_pred_classes, output_dict=True)
            }
            
            logger.info(f"Deep Learning model training completed in {training_time:.2f} seconds")
            logger.info(f"Training accuracy: {train_accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error during Deep Learning model training: {str(e)}")
            raise
    
    def predict(self, employee_data):
        """
        Predict stress level for new employee data
        
        Args:
            employee_data: DataFrame or dict with employee features
        
        Returns:
            dict with prediction results
        """
        try:
            if not self.is_trained and self.model is None:
                self.load_model()
            
            # Convert to DataFrame if dict
            if isinstance(employee_data, dict):
                employee_data = pd.DataFrame([employee_data])
            
            # Prepare features
            X = self._prepare_features(employee_data)
            X_scaled = self.scaler.transform(X)
            
            # Get predictions
            predictions = self.model.predict(X_scaled)
            predicted_classes = np.argmax(predictions, axis=1)
            confidence_scores = np.max(predictions, axis=1)
            
            results = []
            for i in range(len(employee_data)):
                stress_category = self.stress_categories[predicted_classes[i]]
                confidence = float(confidence_scores[i])
                
                # Convert category back to approximate stress level
                if predicted_classes[i] == 0:  # Low
                    stress_level = 15.0
                elif predicted_classes[i] == 1:  # Medium
                    stress_level = 50.0
                else:  # High
                    stress_level = 85.0
                
                result = {
                    'predicted_stress_level': stress_level,
                    'stress_category': stress_category,
                    'confidence_score': confidence,
                    'category_probabilities': {
                        'Low': float(predictions[i][0]),
                        'Medium': float(predictions[i][1]),
                        'High': float(predictions[i][2])
                    }
                }
                results.append(result)
            
            return results if len(results) > 1 else results[0]
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def calculate_feature_importance(self, training_data, n_iterations=100):
        """
        Calculate feature importance using permutation importance
        
        Args:
            training_data: Training data used for calculating importance
            n_iterations: Number of permutation iterations
        
        Returns:
            dict with feature importance scores
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before calculating feature importance")
            
            # Prepare data
            X = self._prepare_features(training_data)
            y = training_data['stress_level'].apply(self._categorize_stress)
            X_scaled = self.scaler.transform(X)
            
            # Baseline accuracy
            baseline_pred = self.model.predict(X_scaled)
            baseline_accuracy = accuracy_score(y, np.argmax(baseline_pred, axis=1))
            
            feature_importance = {}
            
            for feature_idx, feature_name in enumerate(self.feature_names):
                importance_scores = []
                
                for _ in range(n_iterations):
                    # Permute the feature
                    X_permuted = X_scaled.copy()
                    np.random.shuffle(X_permuted[:, feature_idx])
                    
                    # Calculate accuracy with permuted feature
                    permuted_pred = self.model.predict(X_permuted)
                    permuted_accuracy = accuracy_score(y, np.argmax(permuted_pred, axis=1))
                    
                    # Importance is the decrease in accuracy
                    importance = baseline_accuracy - permuted_accuracy
                    importance_scores.append(importance)
                
                feature_importance[feature_name] = {
                    'importance': float(np.mean(importance_scores)),
                    'std': float(np.std(importance_scores))
                }
            
            # Normalize importance scores to percentages
            total_importance = sum([v['importance'] for v in feature_importance.values()])
            if total_importance > 0:
                for feature in feature_importance:
                    feature_importance[feature]['percentage'] = (
                        feature_importance[feature]['importance'] / total_importance * 100
                    )
            
            logger.info("Feature importance calculation completed")
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise
    
    def save_model(self):
        """Save the trained model and encoders"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            # Save the Keras model
            self.model.save(f"{self.model_path}/model.h5")
            
            # Save encoders and scaler
            joblib.dump(self.scaler, f"{self.model_path}/scaler.pkl")
            joblib.dump(self.department_encoder, f"{self.model_path}/department_encoder.pkl")
            
            # Save model metadata
            metadata = {
                'model_version': self.model_version,
                'feature_names': self.feature_names,
                'stress_categories': self.stress_categories,
                'is_trained': self.is_trained,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(f"{self.model_path}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Deep Learning model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self):
        """Load a trained model and encoders"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path {self.model_path} does not exist")
            
            # Load the Keras model
            self.model = keras.models.load_model(f"{self.model_path}/model.h5")
            
            # Load encoders and scaler
            self.scaler = joblib.load(f"{self.model_path}/scaler.pkl")
            self.department_encoder = joblib.load(f"{self.model_path}/department_encoder.pkl")
            self._department_fitted = True
            
            # Load metadata
            with open(f"{self.model_path}/metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.model_version = metadata['model_version']
            self.feature_names = metadata['feature_names']
            self.stress_categories = metadata['stress_categories']
            self.is_trained = metadata['is_trained']
            
            logger.info(f"Deep Learning model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built yet"
        
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines) 