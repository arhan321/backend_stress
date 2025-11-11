import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

class AdvancedStressDeepLearningModel:
    """Advanced Deep Learning Model for Employee Stress Prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'workload', 'work_life_balance', 'team_conflict', 
            'management_support', 'work_environment'
        ]
        self.model_accuracy = 0.0
        self.training_history = None
        
    def create_model(self, input_dim):
        """Create advanced deep learning model architecture"""
        model = keras.Sequential([
            # Input layer
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            # Hidden layers with increasing complexity
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            # Output layer
            keras.layers.Dense(1, activation='sigmoid')  # Output 0-1, will be scaled to 0-100
        ])
        
        # Compile with advanced optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Extract features and target
        X = df[self.feature_columns].copy()
        y = df['stress_level'].copy() / 100.0  # Normalize to 0-1
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train(self, df):
        """Train the deep learning model"""
        try:
            logger.info("Starting advanced deep learning model training...")
            
            # Prepare data
            X, y = self.prepare_data(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create model
            self.model = self.create_model(X.shape[1])
            
            # Training callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
            
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001
            )
            
            # Train model
            self.training_history = self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate model
            y_pred = self.model.predict(X_test, verbose=0)
            self.model_accuracy = r2_score(y_test, y_pred.flatten())
            
            self.is_trained = True
            
            logger.info(f"Model training completed. R² Score: {self.model_accuracy:.4f}")
            
            return {
                'success': True,
                'accuracy': self.model_accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'final_val_loss': min(self.training_history.history['val_loss']),
                'epochs_trained': len(self.training_history.history['loss'])
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict(self, data):
        """Make predictions using the trained model"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            # Prepare input data
            if isinstance(data, dict):
                input_data = np.array([[
                    data[col] for col in self.feature_columns
                ]])
            else:
                input_data = data[self.feature_columns].values
            
            # Scale input
            input_scaled = self.scaler.transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(input_scaled, verbose=0)
            
            # Convert back to 0-100 scale
            stress_level = float(prediction[0][0] * 100)
            
            return {
                'predicted_stress_level': max(0, min(100, stress_level)),
                'confidence_score': self.model_accuracy,
                'model_type': 'deep_learning'
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None

class NeuralCollaborativeFilteringModel:
    """Neural Collaborative Filtering for Stress Factor Analysis"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.employee_encoder = LabelEncoder()
        self.factor_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model_accuracy = 0.0
        
    def create_ncf_model(self, num_employees, num_factors, embedding_dim=50):
        """Create Neural Collaborative Filtering model"""
        # Employee embedding
        employee_input = keras.layers.Input(shape=(), name='employee_id')
        employee_embedding = keras.layers.Embedding(
            num_employees, embedding_dim, name='employee_embedding'
        )(employee_input)
        employee_vec = keras.layers.Flatten()(employee_embedding)
        
        # Factor embedding
        factor_input = keras.layers.Input(shape=(), name='factor_id')
        factor_embedding = keras.layers.Embedding(
            num_factors, embedding_dim, name='factor_embedding'
        )(factor_input)
        factor_vec = keras.layers.Flatten()(factor_embedding)
        
        # Additional features
        features_input = keras.layers.Input(shape=(3,), name='features')  # department, age, position
        
        # Concatenate all features
        concat = keras.layers.Concatenate()([employee_vec, factor_vec, features_input])
        
        # Deep layers
        x = keras.layers.Dense(256, activation='relu')(concat)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        
        # Output layer
        output = keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(
            inputs=[employee_input, factor_input, features_input],
            outputs=output
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_ncf_data(self, df):
        """Prepare data for NCF training"""
        # Create interaction data
        interactions = []
        
        factor_names = ['workload', 'work_life_balance', 'team_conflict', 
                       'management_support', 'work_environment']
        
        for idx, row in df.iterrows():
            for i, factor in enumerate(factor_names):
                interactions.append({
                    'employee_id': row['employee_id'],
                    'factor_id': i,
                    'factor_value': row[factor] / 10.0,  # Normalize to 0-1
                    'department': row.get('department', 'Unknown'),
                    'age': row.get('age', 30),
                    'position': row.get('position', 'Employee'),
                    'stress_impact': row[factor] / 10.0 * (row['stress_level'] / 100.0)
                })
        
        interaction_df = pd.DataFrame(interactions)
        
        # Encode categorical variables
        interaction_df['employee_encoded'] = self.employee_encoder.fit_transform(
            interaction_df['employee_id'].astype(str)
        )
        interaction_df['factor_encoded'] = self.factor_encoder.fit_transform(
            interaction_df['factor_id']
        )
        
        # Prepare features
        dept_encoder = LabelEncoder()
        pos_encoder = LabelEncoder()
        
        interaction_df['dept_encoded'] = dept_encoder.fit_transform(
            interaction_df['department'].fillna('Unknown')
        )
        interaction_df['pos_encoded'] = pos_encoder.fit_transform(
            interaction_df['position'].fillna('Employee')
        )
        
        # Scale age
        age_scaled = StandardScaler().fit_transform(
            interaction_df[['age']].fillna(30)
        )
        
        X = {
            'employee_id': interaction_df['employee_encoded'].values,
            'factor_id': interaction_df['factor_encoded'].values,
            'features': np.column_stack([
                interaction_df['dept_encoded'].values,
                age_scaled.flatten(),
                interaction_df['pos_encoded'].values
            ])
        }
        
        y = interaction_df['stress_impact'].values
        
        return X, y, interaction_df
    
    def train(self, df):
        """Train the NCF model"""
        try:
            logger.info("Starting Neural Collaborative Filtering training...")
            
            # Prepare data
            X, y, interaction_df = self.prepare_ncf_data(df)
            
            # Get dimensions
            num_employees = len(self.employee_encoder.classes_)
            num_factors = len(self.factor_encoder.classes_)
            
            # Create model
            self.model = self.create_ncf_model(num_employees, num_factors)
            
            # Split data
            indices = np.arange(len(y))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
            
            X_train = {key: val[train_idx] for key, val in X.items()}
            X_test = {key: val[test_idx] for key, val in X.items()}
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=64,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Evaluate
            y_pred = self.model.predict(X_test, verbose=0)
            self.model_accuracy = r2_score(y_test, y_pred.flatten())
            
            self.is_trained = True
            
            logger.info(f"NCF training completed. R² Score: {self.model_accuracy:.4f}")
            
            return {
                'success': True,
                'accuracy': self.model_accuracy,
                'num_employees': num_employees,
                'num_factors': num_factors,
                'training_samples': len(train_idx),
                'test_samples': len(test_idx)
            }
            
        except Exception as e:
            logger.error(f"NCF training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_factor_recommendations(self, employee_id, top_k=5):
        """Get personalized factor recommendations for an employee"""
        if not self.is_trained:
            return None
        
        try:
            # Encode employee ID
            if str(employee_id) not in self.employee_encoder.classes_:
                return None
            
            employee_encoded = self.employee_encoder.transform([str(employee_id)])[0]
            
            # Predict for all factors
            factor_predictions = []
            for factor_id in range(len(self.factor_encoder.classes_)):
                # Create input
                X_pred = {
                    'employee_id': np.array([employee_encoded]),
                    'factor_id': np.array([factor_id]),
                    'features': np.array([[0, 0.5, 0]])  # Default features
                }
                
                prediction = self.model.predict(X_pred, verbose=0)[0][0]
                
                factor_predictions.append({
                    'factor_id': factor_id,
                    'factor_name': ['workload', 'work_life_balance', 'team_conflict', 
                                  'management_support', 'work_environment'][factor_id],
                    'stress_impact': float(prediction),
                    'priority': 'High' if prediction > 0.7 else 'Medium' if prediction > 0.4 else 'Low'
                })
            
            # Sort by stress impact
            factor_predictions.sort(key=lambda x: x['stress_impact'], reverse=True)
            
            return factor_predictions[:top_k]
            
        except Exception as e:
            logger.error(f"Factor recommendation failed: {str(e)}")
            return None

# Global model instances
advanced_dl_model = AdvancedStressDeepLearningModel()
ncf_model = NeuralCollaborativeFilteringModel() 