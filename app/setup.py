#!/usr/bin/env python3
"""
Setup script for Stress Analysis Backend
This script initializes the database, creates sample data, and sets up the environment
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import mysql.connector
from loguru import logger

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import db_manager, DatabaseConfig
from models.database_models import Base, Employee, Dataset, AnalysisResult
from ml_models import StressDeepLearningModel, NeuralCollaborativeFiltering

def check_mysql_connection():
    """Check if MySQL (XAMPP) is running and accessible"""
    try:
        connection = mysql.connector.connect(
            host=DatabaseConfig.MYSQL_HOST,
            port=DatabaseConfig.MYSQL_PORT,
            user=DatabaseConfig.MYSQL_USER,
            password=DatabaseConfig.MYSQL_PASSWORD
        )
        connection.close()
        logger.info("✅ MySQL connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ MySQL connection failed: {str(e)}")
        logger.error("Please ensure XAMPP is running and MySQL service is started")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'uploads',
        'models',
        'logs',
        'data/samples'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def generate_sample_employee_data(num_employees=500):
    """Generate sample employee dataset for testing"""
    logger.info(f"Generating sample dataset with {num_employees} employees...")
    
    np.random.seed(42)  # For reproducible results
    
    departments = ['HR', 'Finance', 'Marketing', 'IT', 'Operations']
    positions = ['Junior', 'Senior', 'Lead', 'Manager', 'Director']
    
    # Generate base employee data
    employees = []
    
    for i in range(num_employees):
        employee_id = f"EMP{str(i+1).zfill(4)}"
        department = np.random.choice(departments)
        position = np.random.choice(positions)
        age = np.random.randint(22, 65)
        
        # Generate correlated stress factors
        # Higher workload tends to increase stress
        workload = np.random.normal(6.5, 2.0)
        workload = np.clip(workload, 1, 10)
        
        # Work-life balance inversely related to workload
        work_life_balance = np.random.normal(7 - (workload - 5) * 0.3, 1.5)
        work_life_balance = np.clip(work_life_balance, 1, 10)
        
        # Team conflict affects stress
        team_conflict = np.random.normal(4, 2)
        team_conflict = np.clip(team_conflict, 1, 10)
        
        # Management support helps reduce stress
        management_support = np.random.normal(6.5, 2)
        management_support = np.clip(management_support, 1, 10)
        
        # Work environment contributes to overall wellness
        work_environment = np.random.normal(6.8, 1.8)
        work_environment = np.clip(work_environment, 1, 10)
        
        # Calculate stress level based on factors with some noise
        stress_base = (
            workload * 0.25 +
            (10 - work_life_balance) * 0.20 +
            team_conflict * 0.20 +
            (10 - management_support) * 0.15 +
            (10 - work_environment) * 0.20
        )
        
        # Add department-specific stress modifiers
        dept_modifiers = {
            'HR': -0.5,
            'Finance': 1.0,
            'Marketing': 0.5,
            'IT': 0.8,
            'Operations': 0.3
        }
        
        stress_level = stress_base * 10 + dept_modifiers[department] * 5
        stress_level += np.random.normal(0, 5)  # Add noise
        stress_level = np.clip(stress_level, 0, 100)
        
        employee = {
            'employee_id': employee_id,
            'department': department,
            'position': position,
            'age': age,
            'workload': round(workload, 1),
            'work_life_balance': round(work_life_balance, 1),
            'team_conflict': round(team_conflict, 1),
            'management_support': round(management_support, 1),
            'work_environment': round(work_environment, 1),
            'stress_level': round(stress_level, 1)
        }
        
        employees.append(employee)
    
    # Create DataFrame
    df = pd.DataFrame(employees)
    
    # Save to CSV
    csv_path = 'data/samples/employee_stress_data_500.csv'
    df.to_csv(csv_path, index=False)
    
    logger.info(f"✅ Sample dataset saved to {csv_path}")
    logger.info(f"📊 Dataset statistics:")
    logger.info(f"   - Total employees: {len(df)}")
    logger.info(f"   - Departments: {df['department'].nunique()}")
    logger.info(f"   - Average stress level: {df['stress_level'].mean():.1f}")
    logger.info(f"   - Stress level range: {df['stress_level'].min():.1f} - {df['stress_level'].max():.1f}")
    
    return csv_path, df

def setup_database():
    """Initialize database and create tables"""
    logger.info("Setting up database...")
    
    try:
        # Initialize database connection
        if not db_manager.initialize_database():
            raise Exception("Failed to initialize database")
        
        # Create all tables
        db_manager.create_tables()
        
        logger.info("✅ Database setup completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {str(e)}")
        return False

def test_ml_models(sample_data_path):
    """Test ML models with sample data"""
    logger.info("Testing ML models with sample data...")
    
    try:
        # Load sample data
        df = pd.read_csv(sample_data_path)
        
        # Test Deep Learning model
        logger.info("Testing Deep Learning model...")
        dl_model = StressDeepLearningModel(model_version="test_1.0")
        
        # Train with a subset for quick testing
        train_data = df.head(200)
        training_results = dl_model.train(train_data, epochs=10, batch_size=16)
        
        logger.info(f"✅ Deep Learning training completed:")
        logger.info(f"   - Training accuracy: {training_results['train_accuracy']:.3f}")
        logger.info(f"   - Validation accuracy: {training_results['validation_accuracy']:.3f}")
        logger.info(f"   - Training time: {training_results['training_time_seconds']:.1f}s")
        
        # Test prediction
        test_employee = {
            'department': 'IT',
            'workload': 8.0,
            'work_life_balance': 5.0,
            'team_conflict': 6.0,
            'management_support': 4.0,
            'work_environment': 7.0
        }
        
        prediction = dl_model.predict(test_employee)
        logger.info(f"✅ Test prediction: {prediction['predicted_stress_level']:.1f}% stress level")
        
        # Test NCF model
        logger.info("Testing Neural Collaborative Filtering model...")
        ncf_model = NeuralCollaborativeFiltering(model_version="test_1.0")
        
        ncf_results = ncf_model.train(train_data, epochs=20, batch_size=32)
        logger.info(f"✅ NCF training completed:")
        logger.info(f"   - Training time: {ncf_results['training_time_seconds']:.1f}s")
        logger.info(f"   - Final loss: {ncf_results['final_loss']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ML model testing failed: {str(e)}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Starting Stress Analysis Backend Setup")
    logger.info("=" * 50)
    
    # Step 1: Check MySQL connection
    logger.info("Step 1: Checking MySQL connection...")
    if not check_mysql_connection():
        logger.error("❌ Setup failed: MySQL not accessible")
        logger.info("💡 Please start XAMPP and ensure MySQL service is running")
        return False
    
    # Step 2: Create directories
    logger.info("Step 2: Creating directories...")
    create_directories()
    
    # Step 3: Setup database
    logger.info("Step 3: Setting up database...")
    if not setup_database():
        logger.error("❌ Setup failed: Database setup error")
        return False
    
    # Step 4: Generate sample data
    logger.info("Step 4: Generating sample data...")
    try:
        sample_path, sample_df = generate_sample_employee_data(500)
    except Exception as e:
        logger.error(f"❌ Sample data generation failed: {str(e)}")
        return False
    
    # Step 5: Test ML models
    logger.info("Step 5: Testing ML models...")
    if not test_ml_models(sample_path):
        logger.warning("⚠️  ML model testing failed, but setup can continue")
    
    # Final steps
    logger.info("=" * 50)
    logger.info("🎉 Setup completed successfully!")
    logger.info("")
    logger.info("📋 Next steps:")
    logger.info("1. Copy .env.example to .env and configure if needed")
    logger.info("2. Run: python app.py")
    logger.info("3. API will be available at http://localhost:5000")
    logger.info("4. Use the sample dataset at: data/samples/employee_stress_data_500.csv")
    logger.info("")
    logger.info("🔗 API Endpoints:")
    logger.info("   - Health Check: GET /api/health")
    logger.info("   - Upload Dataset: POST /api/upload-dataset")
    logger.info("   - Analyze Dataset: POST /api/analyze-dataset/<id>")
    logger.info("   - Predict Stress: POST /api/predict-stress")
    logger.info("")
    
    return True

if __name__ == "__main__":
    # Configure logging
    logger.add("logs/setup.log", rotation="10 MB")
    
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed with unexpected error: {str(e)}")
        sys.exit(1) 