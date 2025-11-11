import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix
import joblib
import os
from datetime import datetime
from loguru import logger
import json

class StressDeepLearningModel:
    """Deep Learning model for employee stress prediction using TensorFlow"""
    
    def __init__(self, model_version="1.0"):
        self.model_version = model_version
        self.model = None
        self.scaler = StandardScaler()
        self.department_encoder = LabelEncoder()
        self.feature_names = [
            'workload', 'work_life_balance', 'team_conflict', 
            'management_support', 'work_environment', 'department_encoded'
        ]
        self.stress_categories = ['Low', 'Medium', 'High']
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
        df = data.copy()
        
        # Encode department if it's categorical
        if 'department' in df.columns:
            if not hasattr(self, '_department_fitted'):
                self.department_encoder.fit(df['department'])
                self._department_fitted = True
            df['department_encoded'] = self.department_encoder.transform(df['department'])
        else:
            df['department_encoded'] = 0
        
        feature_data = df[self.feature_names].copy()
        return feature_data
    
    def build_model(self, input_dim=6, hidden_layers=[128, 64, 32], dropout_rate=0.3):
        """Build the deep neural network architecture"""
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
        
        # Output layer for 3-class classification
        model.add(layers.Dense(3, activation='softmax', name='output'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"Deep Learning model built with {len(hidden_layers)} hidden layers")
        return model
    
    def train(self, training_data, validation_split=0.2, epochs=100, batch_size=32):
        """Train the deep learning model"""
        try:
            logger.info("Starting Deep Learning model training...")
            
            # Prepare features and target
            X = self._prepare_features(training_data)
            y = training_data['stress_level'].apply(self._categorize_stress)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Build model if not built
            if self.model is None:
                self.build_model(input_dim=X_train_scaled.shape[1])
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
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
            
            self.is_trained = True
            self.save_model()
            
            return {
                'model_type': 'deep_learning',
                'model_version': self.model_version,
                'training_time_seconds': training_time,
                'train_accuracy': float(train_accuracy),
                'validation_accuracy': float(val_accuracy),
                'train_loss': float(train_loss),
                'validation_loss': float(val_loss),
                'training_data_size': len(training_data),
                'epochs_trained': len(history.history['loss'])
            }
            
        except Exception as e:
            logger.error(f"Error during Deep Learning training: {str(e)}")
            raise
    
    def predict(self, employee_data):
        """Predict stress level for new employee data"""
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
                
                # Convert category back to stress level
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
    
    def save_model(self):
        """Save the trained model and encoders"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            # Save the Keras model
            self.model.save(f"{self.model_path}/model.h5")
            
            # Save encoders and scaler
            joblib.dump(self.scaler, f"{self.model_path}/scaler.pkl")
            joblib.dump(self.department_encoder, f"{self.model_path}/department_encoder.pkl")
            
            # Save metadata
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
        """Load a trained model"""
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


class NeuralCollaborativeFiltering:
    """Neural Collaborative Filtering for employee stress pattern analysis"""
    
    def __init__(self, embedding_dim=32, hidden_dims=[128, 64], model_version="1.0"):
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.model_version = model_version
        self.model = None
        self.employee_encoder = LabelEncoder()
        self.factor_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = f"models/ncf_v{model_version}"
        
    def _prepare_interaction_data(self, data):
        """Prepare data for collaborative filtering"""
        interactions = []
        
        for _, row in data.iterrows():
            employee_id = row['employee_id']
            factors = {
                'workload': row['workload'],
                'work_life_balance': row['work_life_balance'], 
                'team_conflict': row['team_conflict'],
                'management_support': row['management_support'],
                'work_environment': row['work_environment']
            }
            
            for factor_name, factor_value in factors.items():
                interactions.append({
                    'employee_id': employee_id,
                    'factor': factor_name,
                    'rating': factor_value,
                    'stress_level': row['stress_level']
                })
        
        return pd.DataFrame(interactions)
    
    def build_model(self, n_employees, n_factors):
        """Build Neural Collaborative Filtering model using PyTorch"""
        
        class NCFModel(nn.Module):
            def __init__(self, n_employees, n_factors, embedding_dim, hidden_dims):
                super(NCFModel, self).__init__()
                
                # Embedding layers
                self.employee_embedding = nn.Embedding(n_employees, embedding_dim)
                self.factor_embedding = nn.Embedding(n_factors, embedding_dim)
                
                # MLP layers
                input_dim = embedding_dim * 2
                layers = []
                
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(input_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.2))
                    input_dim = hidden_dim
                
                # Output layer
                layers.append(nn.Linear(input_dim, 1))
                
                self.mlp = nn.Sequential(*layers)
                
            def forward(self, employee_ids, factor_ids):
                # Get embeddings
                employee_emb = self.employee_embedding(employee_ids)
                factor_emb = self.factor_embedding(factor_ids)
                
                # Concatenate embeddings
                concat_emb = torch.cat([employee_emb, factor_emb], dim=1)
                
                # Pass through MLP
                output = self.mlp(concat_emb)
                
                return output.squeeze()
        
        self.model = NCFModel(n_employees, n_factors, self.embedding_dim, self.hidden_dims)
        return self.model
    
    def train(self, training_data, epochs=100, batch_size=256, learning_rate=0.001):
        """Train the NCF model"""
        try:
            logger.info("Starting Neural Collaborative Filtering training...")
            
            # Prepare interaction data
            interactions = self._prepare_interaction_data(training_data)
            
            # Encode employees and factors
            interactions['employee_encoded'] = self.employee_encoder.fit_transform(interactions['employee_id'])
            interactions['factor_encoded'] = self.factor_encoder.fit_transform(interactions['factor'])
            
            # Normalize ratings
            ratings_scaled = self.scaler.fit_transform(interactions[['rating']]).flatten()
            
            # Build model
            n_employees = len(self.employee_encoder.classes_)
            n_factors = len(self.factor_encoder.classes_)
            
            if self.model is None:
                self.build_model(n_employees, n_factors)
            
            # Prepare data for training
            employee_ids = torch.LongTensor(interactions['employee_encoded'].values)
            factor_ids = torch.LongTensor(interactions['factor_encoded'].values)
            ratings = torch.FloatTensor(ratings_scaled)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            start_time = datetime.now()
            
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                
                # Shuffle data
                indices = torch.randperm(len(employee_ids))
                
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    
                    batch_employees = employee_ids[batch_indices]
                    batch_factors = factor_ids[batch_indices]
                    batch_ratings = ratings[batch_indices]
                    
                    # Forward pass
                    predictions = self.model(batch_employees, batch_factors)
                    loss = criterion(predictions, batch_ratings)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if epoch % 10 == 0:
                    avg_loss = total_loss / (len(indices) // batch_size)
                    logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            self.is_trained = True
            self.save_model()
            
            return {
                'model_type': 'ncf',
                'model_version': self.model_version,
                'training_time_seconds': training_time,
                'training_data_size': len(training_data),
                'epochs_trained': epochs,
                'final_loss': total_loss / (len(indices) // batch_size)
            }
            
        except Exception as e:
            logger.error(f"Error during NCF training: {str(e)}")
            raise
    
    def predict_employee_factors(self, employee_id, factor_list=None):
        """Predict factor values for a specific employee"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")
            
            if factor_list is None:
                factor_list = ['workload', 'work_life_balance', 'team_conflict', 
                             'management_support', 'work_environment']
            
            # Encode employee ID
            if employee_id not in self.employee_encoder.classes_:
                # Handle new employee (use average)
                employee_encoded = 0
            else:
                employee_encoded = self.employee_encoder.transform([employee_id])[0]
            
            predictions = {}
            
            for factor in factor_list:
                if factor in self.factor_encoder.classes_:
                    factor_encoded = self.factor_encoder.transform([factor])[0]
                    
                    # Make prediction
                    employee_tensor = torch.LongTensor([employee_encoded])
                    factor_tensor = torch.LongTensor([factor_encoded])
                    
                    self.model.eval()
                    with torch.no_grad():
                        prediction = self.model(employee_tensor, factor_tensor)
                        
                    # Inverse transform to original scale
                    prediction_scaled = prediction.item()
                    prediction_original = self.scaler.inverse_transform([[prediction_scaled]])[0][0]
                    
                    predictions[factor] = float(prediction_original)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during NCF prediction: {str(e)}")
            raise
    
    def get_employee_embeddings(self):
        """Get learned employee embeddings"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        embeddings = self.model.employee_embedding.weight.data.numpy()
        return embeddings
    
    def get_factor_embeddings(self):
        """Get learned factor embeddings"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        embeddings = self.model.factor_embedding.weight.data.numpy()
        return embeddings
    
    def save_model(self):
        """Save the NCF model"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            # Save PyTorch model
            torch.save(self.model.state_dict(), f"{self.model_path}/model.pth")
            
            # Save encoders and scaler
            joblib.dump(self.employee_encoder, f"{self.model_path}/employee_encoder.pkl")
            joblib.dump(self.factor_encoder, f"{self.model_path}/factor_encoder.pkl")
            joblib.dump(self.scaler, f"{self.model_path}/scaler.pkl")
            
            # Save model architecture info
            metadata = {
                'model_version': self.model_version,
                'embedding_dim': self.embedding_dim,
                'hidden_dims': self.hidden_dims,
                'n_employees': len(self.employee_encoder.classes_),
                'n_factors': len(self.factor_encoder.classes_),
                'is_trained': self.is_trained,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(f"{self.model_path}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"NCF model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving NCF model: {str(e)}")
            raise
    
    def load_model(self):
        """Load the NCF model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path {self.model_path} does not exist")
            
            # Load metadata first
            with open(f"{self.model_path}/metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.model_version = metadata['model_version']
            self.embedding_dim = metadata['embedding_dim']
            self.hidden_dims = metadata['hidden_dims']
            self.is_trained = metadata['is_trained']
            
            # Load encoders and scaler
            self.employee_encoder = joblib.load(f"{self.model_path}/employee_encoder.pkl")
            self.factor_encoder = joblib.load(f"{self.model_path}/factor_encoder.pkl")
            self.scaler = joblib.load(f"{self.model_path}/scaler.pkl")
            
            # Rebuild and load model
            self.build_model(metadata['n_employees'], metadata['n_factors'])
            self.model.load_state_dict(torch.load(f"{self.model_path}/model.pth"))
            
            logger.info(f"NCF model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading NCF model: {str(e)}")
            raise 