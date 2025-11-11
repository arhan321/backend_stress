from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import traceback
from datetime import datetime
import io
import uuid
from werkzeug.utils import secure_filename
from loguru import logger
import json

# Import our models and database
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import db_manager, DatabaseConfig
from models.database_models import Employee, Dataset, AnalysisResult, StressPrediction, ModelTraining, SystemLog
from ml_models import StressDeepLearningModel, NeuralCollaborativeFiltering
from nan_fix_utils import enhanced_factor_importance_calculation, dynamic_correlation_analysis
from recommendation_engine import recommendation_engine

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model instances
deep_learning_model = None
ncf_model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_system_event(level, module, message, additional_data=None):
    """Log system events to database with proper JSON serialization"""
    try:
        session = db_manager.get_session()
        
        # Properly serialize additional_data to ensure JSON compatibility
        serialized_additional_data = None
        if additional_data is not None:
            try:
                # Convert to JSON-serializable format
                serialized_additional_data = _make_json_serializable(additional_data)
            except Exception as e:
                logger.error(f"Failed to serialize additional_data: {str(e)}")
                # Fallback to string representation if serialization fails
                serialized_additional_data = {"serialization_error": str(additional_data)}
        
        log_entry = SystemLog(
            level=level,
            module=module,
            message=message,
            additional_data=serialized_additional_data
        )
        session.add(log_entry)
        session.commit()
        session.close()
    except Exception as e:
        logger.error(f"Failed to log system event: {str(e)}")

def _make_json_serializable(obj):
    """Convert objects to JSON-serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'item'):  # numpy types
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif hasattr(obj, '__dict__'):  # custom objects
        return str(obj)
    else:
        return str(obj)

@app.before_first_request
def initialize_app():
    """Initialize application components"""
    global deep_learning_model, ncf_model
    
    try:
        # Initialize database
        if not db_manager.initialize_database():
            raise Exception("Failed to initialize database")
        
        # Create tables
        db_manager.create_tables()
        
        # Initialize ML models
        deep_learning_model = StressDeepLearningModel()
        ncf_model = NeuralCollaborativeFiltering()
        
        # Try to load pre-trained models
        try:
            deep_learning_model.load_model()
            logger.info("Pre-trained Deep Learning model loaded")
        except:
            logger.info("No pre-trained Deep Learning model found")
        
        try:
            ncf_model.load_model()
            logger.info("Pre-trained NCF model loaded")
        except:
            logger.info("No pre-trained NCF model found")
        
        log_system_event("INFO", "API", "Application initialized successfully")
        logger.info("Stress Analysis API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        log_system_event("ERROR", "API", f"Application initialization failed: {str(e)}")
        raise

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db_test = db_manager.test_connection()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': 'connected' if db_test else 'disconnected',
            'deep_learning_model': 'loaded' if deep_learning_model and deep_learning_model.is_trained else 'not_loaded',
            'ncf_model': 'loaded' if ncf_model and ncf_model.is_trained else 'not_loaded'
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """Upload and validate employee dataset"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload CSV or Excel files.'}), 400
        
        # Get additional parameters
        dataset_name = request.form.get('dataset_name', f'Dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        description = request.form.get('description', '')
        
        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Read and validate dataset
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Validate required columns
            required_columns = [
                'employee_id', 'department', 'workload', 'work_life_balance',
                'team_conflict', 'management_support', 'work_environment', 'stress_level'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return jsonify({
                    'error': f'Missing required columns: {", ".join(missing_columns)}',
                    'required_columns': required_columns,
                    'found_columns': list(df.columns)
                }), 400
            
            # Check minimum dataset size
            if len(df) < 50:
                return jsonify({
                    'error': f'Dataset too small. Minimum 50 records required, found {len(df)}'
                }), 400
            
            # Validate data types and ranges
            validation_errors = []
            
            # Check numeric columns
            numeric_columns = ['workload', 'work_life_balance', 'team_conflict', 
                             'management_support', 'work_environment', 'stress_level']
            
            for col in numeric_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    validation_errors.append(f'{col} must be numeric')
                else:
                    if col != 'stress_level':
                        # Check 1-10 scale for factors
                        if df[col].min() < 1 or df[col].max() > 10:
                            validation_errors.append(f'{col} must be between 1 and 10')
                    else:
                        # Check 0-100 scale for stress level
                        if df[col].min() < 0 or df[col].max() > 100:
                            validation_errors.append(f'{col} must be between 0 and 100')
            
            # Check for missing values
            if df.isnull().any().any():
                validation_errors.append('Dataset contains missing values')
            
            if validation_errors:
                return jsonify({
                    'error': 'Dataset validation failed',
                    'validation_errors': validation_errors
                }), 400
            
            # Save dataset info to database
            session = db_manager.get_session()
            
            dataset = Dataset(
                name=dataset_name,
                description=description,
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                record_count=len(df),
                status='uploaded'
            )
            
            session.add(dataset)
            session.commit()
            
            dataset_id = dataset.id
            session.close()
            
            # Log successful upload
            log_system_event("INFO", "API", f"Dataset uploaded successfully: {dataset_name}", {
                'dataset_id': dataset_id,
                'record_count': len(df),
                'file_size': os.path.getsize(file_path)
            })
            
            return jsonify({
                'success': True,
                'message': 'Dataset uploaded and validated successfully',
                'dataset_id': dataset_id,
                'dataset_name': dataset_name,
                'record_count': len(df),
                'columns': list(df.columns),
                'file_size': os.path.getsize(file_path),
                'departments': df['department'].unique().tolist(),
                'stress_level_stats': {
                    'min': float(df['stress_level'].min()),
                    'max': float(df['stress_level'].max()),
                    'mean': float(df['stress_level'].mean()),
                    'std': float(df['stress_level'].std())
                }
            }), 200
            
        except Exception as e:
            # Clean up file if validation fails
            if os.path.exists(file_path):
                os.remove(file_path)
            
            logger.error(f"Dataset validation failed: {str(e)}")
            return jsonify({
                'error': f'Failed to process dataset: {str(e)}'
            }), 400
        
    except Exception as e:
        logger.error(f"Dataset upload failed: {str(e)}")
        log_system_event("ERROR", "API", f"Dataset upload failed: {str(e)}")
        return jsonify({
            'error': f'Upload failed: {str(e)}'
        }), 500

@app.route('/api/analyze-dataset/<int:dataset_id>', methods=['POST'])
def analyze_dataset(dataset_id):
    """Analyze dataset using ML models with DYNAMIC factor calculation and recommendations"""
    try:
        session = db_manager.get_session()
        
        # Get dataset info
        dataset = session.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Update dataset status
        dataset.status = 'processing'
        dataset.processing_start_time = datetime.utcnow()
        session.commit()
        
        # Load dataset
        try:
            if dataset.file_path.endswith('.csv'):
                df = pd.read_csv(dataset.file_path)
            else:
                df = pd.read_excel(dataset.file_path)
        except Exception as e:
            dataset.status = 'failed'
            dataset.error_message = f"Failed to load dataset: {str(e)}"
            session.commit()
            session.close()
            return jsonify({'error': f'Failed to load dataset: {str(e)}'}), 500
        
        analysis_start_time = datetime.now()
        
        # Store employee data in database
        for _, row in df.iterrows():
            employee = Employee(
                employee_id=str(row['employee_id']),
                department=row['department'],
                age=row.get('age'),
                position=row.get('position'),
                workload=row['workload'],
                work_life_balance=row['work_life_balance'],
                team_conflict=row['team_conflict'],
                management_support=row['management_support'],
                work_environment=row['work_environment'],
                stress_level=row['stress_level']
            )
            session.merge(employee)  # Use merge to handle duplicates
        
        session.commit()
        
        # Perform analysis
        results = {}
        
        # Calculate overall stress level first
        overall_stress_level = float(df['stress_level'].mean())
        
        # 1. DYNAMIC Factor Importance Calculation (Using enhanced calculation)
        feature_importance = enhanced_factor_importance_calculation(df, 'stress_level', overall_stress_level)
        
        # Get correlation insights for recommendations
        correlation_insights = dynamic_correlation_analysis(df, 'stress_level')
        
        # 2. Deep Learning Analysis
        try:
            global deep_learning_model
            if not deep_learning_model.is_trained:
                # Train the model
                dl_training_results = deep_learning_model.train(df)
                results['deep_learning_training'] = dl_training_results
            
            # Get predictions for all employees
            dl_predictions = deep_learning_model.predict(df)
            if isinstance(dl_predictions, dict):
                dl_predictions = [dl_predictions]
            
            results['deep_learning'] = {
                'overall_stress_level': np.mean([p['predicted_stress_level'] for p in dl_predictions]),
                'predictions': dl_predictions
            }
            
        except Exception as e:
            logger.error(f"Deep Learning analysis failed: {str(e)}")
            results['deep_learning'] = {'error': str(e)}
        
        # 3. Neural Collaborative Filtering Analysis
        try:
            global ncf_model
            if not ncf_model.is_trained:
                # Train the NCF model
                ncf_training_results = ncf_model.train(df)
                results['ncf_training'] = ncf_training_results
            
            # Get factor analysis
            factor_analysis = {}
            unique_employees = df['employee_id'].unique()[:10]  # Sample first 10 employees
            
            for emp_id in unique_employees:
                try:
                    factor_predictions = ncf_model.predict_employee_factors(str(emp_id))
                    factor_analysis[str(emp_id)] = factor_predictions
                except:
                    continue
            
            results['ncf'] = {
                'factor_analysis': factor_analysis
            }
            
        except Exception as e:
            logger.error(f"NCF analysis failed: {str(e)}")
            results['ncf'] = {'error': str(e)}
        
        # 4. Department breakdown with dynamic thresholds
        department_stats = df.groupby('department')['stress_level'].agg(['mean', 'count', 'std']).round(2)
        department_breakdown = {}
        
        for dept in department_stats.index:
            dept_data = department_stats.loc[dept]
            dept_avg_stress = float(dept_data['mean'])
            
            # Categorize department stress level
            if dept_avg_stress >= 70:
                dept_category = 'Critical'
            elif dept_avg_stress >= 50:
                dept_category = 'High'
            elif dept_avg_stress >= 30:
                dept_category = 'Moderate'
            else:
                dept_category = 'Low'
            
            department_breakdown[dept] = {
                'average_stress': dept_avg_stress,
                'employee_count': int(dept_data['count']),
                'stress_std': float(dept_data['std']) if not pd.isna(dept_data['std']) else 0.0,
                'stress_category': dept_category,
                'needs_intervention': dept_avg_stress >= 60
            }
        
        # 5. DYNAMIC RECOMMENDATIONS using the recommendation engine
        analysis_data_for_recommendations = {
            'overall_stress_level': overall_stress_level,
            'factor_contributions': feature_importance,
            'department_breakdown': department_breakdown,
            'correlation_insights': correlation_insights,
            'total_employees': len(df)
        }
        
        # Generate dynamic recommendations
        dynamic_recommendations = recommendation_engine.generate_recommendations(
            analysis_data_for_recommendations
        )
        
        # 6. Stress Distribution Analysis with dynamic thresholds
        high_stress_count = len(df[df['stress_level'] >= 70])
        medium_stress_count = len(df[(df['stress_level'] >= 40) & (df['stress_level'] < 70)])
        low_stress_count = len(df[df['stress_level'] < 40])
        
        stress_distribution = {
            'high_risk': high_stress_count,
            'medium_risk': medium_stress_count,
            'low_risk': low_stress_count,
            'high_risk_percentage': round((high_stress_count / len(df)) * 100, 1),
            'medium_risk_percentage': round((medium_stress_count / len(df)) * 100, 1),
            'low_risk_percentage': round((low_stress_count / len(df)) * 100, 1)
        }
        
        # 7. Categorize overall stress level
        if overall_stress_level >= 80:
            stress_category = 'Critical'
            urgency_level = 'Immediate Action Required'
        elif overall_stress_level >= 60:
            stress_category = 'High'
            urgency_level = 'Action Required'
        elif overall_stress_level >= 40:
            stress_category = 'Moderate'
            urgency_level = 'Monitor Closely'
        else:
            stress_category = 'Low'
            urgency_level = 'Preventive Measures'
        
        processing_time = (datetime.now() - analysis_start_time).total_seconds()
        
        # Save analysis results with dynamic data
        analysis_result = AnalysisResult(
            dataset_id=dataset_id,
            analysis_type='dynamic_comprehensive',
            overall_stress_level=overall_stress_level,
            factor_contributions=feature_importance,
            department_breakdown=department_breakdown,
            correlation_matrix={},  # Will add if needed
            feature_importance=feature_importance,
            processing_time_seconds=processing_time
        )
        
        session.add(analysis_result)
        
        # Update dataset status
        dataset.status = 'processed'
        dataset.processing_end_time = datetime.utcnow()
        
        session.commit()
        session.close()
        
        # Log successful analysis
        log_system_event("INFO", "API", f"Dynamic analysis completed: {dataset.name}", {
            'dataset_id': dataset_id,
            'processing_time_seconds': processing_time,
            'overall_stress_level': overall_stress_level,
            'stress_category': stress_category,
            'recommendations_count': len(dynamic_recommendations)
        })
        
        # Prepare comprehensive response with dynamic data
        response = {
            'success': True,
            'dataset_id': dataset_id,
            'overall_stress_level': overall_stress_level,
            'stress_category': stress_category,
            'urgency_level': urgency_level,
            'analysis_results': {
                'factor_contributions': feature_importance,
                'department_breakdown': department_breakdown,
                'stress_distribution': stress_distribution,
                'correlation_insights': correlation_insights,
                'dynamic_recommendations': dynamic_recommendations,
                'feature_importance': feature_importance  # For backward compatibility
            },
            'dynamic_insights': {
                'highest_stress_factor': max(feature_importance.items(), key=lambda x: x[1]['importance_percentage'])[0] if feature_importance else 'Unknown',
                'departments_needing_intervention': [dept for dept, data in department_breakdown.items() if data['needs_intervention']],
                'risk_employees_count': high_stress_count,
                'intervention_priority': 'High' if overall_stress_level >= 60 else 'Medium' if overall_stress_level >= 40 else 'Low'
            },
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat(),
            'model_results': results,
            'calculation_method': 'enhanced_dynamic_synchronized'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        # Update dataset status on error
        try:
            session = db_manager.get_session()
            dataset = session.query(Dataset).filter_by(id=dataset_id).first()
            if dataset:
                dataset.status = 'failed'
                dataset.error_message = str(e)
                dataset.processing_end_time = datetime.utcnow()
                session.commit()
            session.close()
        except:
            pass
        
        logger.error(f"Dynamic dataset analysis failed: {str(e)}")
        log_system_event("ERROR", "API", f"Dynamic dataset analysis failed: {str(e)}", {
            'dataset_id': dataset_id
        })
        
        return jsonify({
            'error': f'Dynamic analysis failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/predict-stress', methods=['POST'])
def predict_stress():
    """Predict stress level for new employee data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['workload', 'work_life_balance', 'team_conflict', 
                          'management_support', 'work_environment', 'department']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Make predictions using both models
        predictions = {}
        
        # Deep Learning prediction
        try:
            global deep_learning_model
            if deep_learning_model and deep_learning_model.is_trained:
                dl_prediction = deep_learning_model.predict(data)
                predictions['deep_learning'] = dl_prediction
            else:
                predictions['deep_learning'] = {'error': 'Deep learning model not trained'}
        except Exception as e:
            predictions['deep_learning'] = {'error': str(e)}
        
        # NCF prediction
        try:
            global ncf_model
            if ncf_model and ncf_model.is_trained:
                employee_id = data.get('employee_id', 'new_employee')
                ncf_prediction = ncf_model.predict_employee_factors(str(employee_id))
                predictions['ncf'] = ncf_prediction
            else:
                predictions['ncf'] = {'error': 'NCF model not trained'}
        except Exception as e:
            predictions['ncf'] = {'error': str(e)}
        
        # Simple statistical prediction as fallback
        # This uses basic rules based on factor weights
        factor_weights = {
            'workload': 0.25,
            'work_life_balance': 0.20,
            'team_conflict': 0.20,
            'management_support': 0.15,
            'work_environment': 0.20
        }
        
        weighted_score = sum(data[factor] * weight for factor, weight in factor_weights.items())
        statistical_stress_level = min(100, max(0, weighted_score * 10))  # Scale to 0-100
        
        predictions['statistical'] = {
            'predicted_stress_level': statistical_stress_level,
            'confidence_score': 0.7,
            'method': 'weighted_average'
        }
        
        # Save prediction to database if employee_id provided
        if 'employee_id' in data:
            try:
                session = db_manager.get_session()
                
                # Get or create employee
                employee = session.query(Employee).filter_by(
                    employee_id=str(data['employee_id'])
                ).first()
                
                if employee:
                    # Use best available prediction
                    best_prediction = predictions.get('deep_learning', predictions['statistical'])
                    if 'error' in best_prediction:
                        best_prediction = predictions['statistical']
                    
                    prediction_record = StressPrediction(
                        employee_id=employee.id,
                        predicted_stress_level=best_prediction['predicted_stress_level'],
                        confidence_score=best_prediction.get('confidence_score', 0.5),
                        input_features=data,
                        factor_weights=factor_weights
                    )
                    
                    session.add(prediction_record)
                    session.commit()
                
                session.close()
                
            except Exception as e:
                logger.error(f"Failed to save prediction: {str(e)}")
        
        log_system_event("INFO", "API", "Stress prediction completed", {
            'employee_id': data.get('employee_id', 'anonymous'),
            'predicted_stress_level': predictions.get('statistical', {}).get('predicted_stress_level')
        })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Stress prediction failed: {str(e)}")
        log_system_event("ERROR", "API", f"Stress prediction failed: {str(e)}")
        
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get list of uploaded datasets"""
    try:
        session = db_manager.get_session()
        datasets = session.query(Dataset).order_by(Dataset.upload_date.desc()).all()
        
        dataset_list = []
        for dataset in datasets:
            dataset_dict = dataset.to_dict()
            
            # Get latest analysis result
            latest_analysis = session.query(AnalysisResult).filter_by(
                dataset_id=dataset.id
            ).order_by(AnalysisResult.analysis_date.desc()).first()
            
            if latest_analysis:
                dataset_dict['latest_analysis'] = latest_analysis.to_dict()
            
            dataset_list.append(dataset_dict)
        
        session.close()
        
        return jsonify({
            'success': True,
            'datasets': dataset_list
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get datasets: {str(e)}")
        return jsonify({
            'error': f'Failed to get datasets: {str(e)}'
        }), 500

@app.route('/api/analysis-results/<int:dataset_id>', methods=['GET'])
def get_analysis_results(dataset_id):
    """Get analysis results for a specific dataset"""
    try:
        session = db_manager.get_session()
        
        # Get dataset
        dataset = session.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Get analysis results
        analysis_results = session.query(AnalysisResult).filter_by(
            dataset_id=dataset_id
        ).order_by(AnalysisResult.analysis_date.desc()).all()
        
        results = []
        for result in analysis_results:
            results.append(result.to_dict())
        
        session.close()
        
        return jsonify({
            'success': True,
            'dataset': dataset.to_dict(),
            'analysis_results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get analysis results: {str(e)}")
        return jsonify({
            'error': f'Failed to get analysis results: {str(e)}'
        }), 500

@app.route('/api/dynamic-recommendations/<int:dataset_id>', methods=['GET'])
def get_dynamic_recommendations(dataset_id):
    """Get dynamic recommendations based on dataset analysis with threshold-based logic"""
    try:
        session = db_manager.get_session()
        
        # Get dataset and analysis results
        dataset = session.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        analysis = session.query(AnalysisResult).filter_by(dataset_id=dataset_id).order_by(AnalysisResult.analysis_date.desc()).first()
        if not analysis:
            return jsonify({'error': 'Analysis results not found. Please run analysis first.'}), 404
        
        # Load the actual dataset for dynamic calculations
        try:
            if dataset.file_path.endswith('.csv'):
                df = pd.read_csv(dataset.file_path)
            else:
                df = pd.read_excel(dataset.file_path)
        except Exception as e:
            session.close()
            return jsonify({'error': f'Failed to load dataset: {str(e)}'}), 500
        
        session.close()
        
        # Calculate real-time dynamic recommendations
        overall_stress_level = float(df['stress_level'].mean())
        
        # Re-calculate dynamic factor importance for fresh recommendations
        feature_importance = enhanced_factor_importance_calculation(df, 'stress_level', overall_stress_level)
        correlation_insights = dynamic_correlation_analysis(df, 'stress_level')
        
        # Department analysis for recommendations
        department_stats = df.groupby('department')['stress_level'].agg(['mean', 'count', 'std']).round(2)
        department_breakdown = {}
        
        for dept in department_stats.index:
            dept_data = department_stats.loc[dept]
            dept_avg_stress = float(dept_data['mean'])
            
            department_breakdown[dept] = {
                'average_stress': dept_avg_stress,
                'employee_count': int(dept_data['count']),
                'stress_std': float(dept_data['std']) if not pd.isna(dept_data['std']) else 0.0,
                'stress_category': 'Critical' if dept_avg_stress >= 70 else 'High' if dept_avg_stress >= 50 else 'Moderate' if dept_avg_stress >= 30 else 'Low',
                'needs_intervention': dept_avg_stress >= 60
            }
        
        # Prepare comprehensive analysis data for recommendation engine
        analysis_data_for_recommendations = {
            'overall_stress_level': overall_stress_level,
            'factor_contributions': feature_importance,
            'department_breakdown': department_breakdown,
            'correlation_insights': correlation_insights,
            'total_employees': len(df),
            'stress_distribution': {
                'high_risk': len(df[df['stress_level'] >= 70]),
                'medium_risk': len(df[(df['stress_level'] >= 40) & (df['stress_level'] < 70)]),
                'low_risk': len(df[df['stress_level'] < 40])
            }
        }
        
        # Generate dynamic recommendations using threshold logic
        dynamic_recommendations = recommendation_engine.generate_recommendations(
            analysis_data_for_recommendations
        )
        
        # Determine stress category and urgency for recommendations
        if overall_stress_level >= 80:
            stress_category = 'Critical'
            urgency_level = 'Immediate Action Required'
            priority_message = 'Urgent intervention needed across multiple factors'
        elif overall_stress_level >= 60:
            stress_category = 'High'
            urgency_level = 'Action Required'
            priority_message = 'Significant stress levels require targeted interventions'
        elif overall_stress_level >= 40:
            stress_category = 'Moderate'
            urgency_level = 'Monitor Closely'
            priority_message = 'Proactive measures recommended to prevent escalation'
        else:
            stress_category = 'Low'
            urgency_level = 'Preventive Measures'
            priority_message = 'Maintain current positive practices and build resilience'
        
        # Find top stress factors for targeted recommendations
        top_stress_factors = sorted(feature_importance.items(), key=lambda x: x[1]['importance_percentage'], reverse=True)[:3]
        
        # Add factor-specific recommendations based on thresholds
        factor_based_recommendations = []
        for factor, data in top_stress_factors:
            importance = data['importance_percentage']
            if importance > 70:  # High impact factor
                factor_based_recommendations.append({
                    'factor': factor,
                    'importance': importance,
                    'priority': 'High',
                    'action': f'Address {factor.replace("_", " ").title()} immediately - contributing {importance:.1f}% to stress levels',
                    'threshold_triggered': 'high_impact'
                })
            elif importance > 50:  # Medium impact factor
                factor_based_recommendations.append({
                    'factor': factor,
                    'importance': importance,
                    'priority': 'Medium',
                    'action': f'Improve {factor.replace("_", " ").title()} - moderate impact factor ({importance:.1f}%)',
                    'threshold_triggered': 'medium_impact'
                })
        
        # Generate department-specific recommendations
        department_recommendations = []
        for dept, data in department_breakdown.items():
            if data['needs_intervention']:
                department_recommendations.append({
                    'department': dept,
                    'stress_level': data['average_stress'],
                    'action': f'Focus intervention on {dept} department - {data["average_stress"]:.1f}% average stress',
                    'employee_count': data['employee_count'],
                    'priority': 'High' if data['average_stress'] >= 70 else 'Medium'
                })
        
        response = {
            'success': True,
            'dataset_id': dataset_id,
            'dataset_name': dataset.name,
            'overall_stress_level': overall_stress_level,
            'stress_category': stress_category,
            'urgency_level': urgency_level,
            'priority_message': priority_message,
            'dynamic_recommendations': {
                'general_recommendations': dynamic_recommendations,
                'factor_based_recommendations': factor_based_recommendations,
                'department_recommendations': department_recommendations,
                'total_recommendations': len(dynamic_recommendations) + len(factor_based_recommendations) + len(department_recommendations)
            },
            'threshold_analysis': {
                'overall_stress_threshold': 'Critical' if overall_stress_level >= 80 else 'High' if overall_stress_level >= 60 else 'Moderate' if overall_stress_level >= 40 else 'Low',
                'high_impact_factors': [f['factor'] for f in factor_based_recommendations if f['threshold_triggered'] == 'high_impact'],
                'departments_needing_intervention': [d['department'] for d in department_recommendations],
                'intervention_priority': 'Immediate' if overall_stress_level >= 70 else 'Planned' if overall_stress_level >= 40 else 'Preventive'
            },
            'factor_importance': feature_importance,
            'correlation_insights': correlation_insights,
            'timestamp': datetime.now().isoformat(),
            'calculation_method': 'dynamic_threshold_based'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Failed to generate dynamic recommendations: {str(e)}")
        return jsonify({'error': f'Failed to generate recommendations: {str(e)}'}), 500

@app.route('/api/enhanced-analysis/<int:dataset_id>', methods=['POST'])
def enhanced_analysis(dataset_id):
    """Enhanced analysis with dynamic implementation steps and personalized recommendations"""
    try:
        session = db_manager.get_session()
        
        # Get dataset
        dataset = session.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Load dataset
        try:
            if dataset.file_path.endswith('.csv'):
                df = pd.read_csv(dataset.file_path)
            else:
                df = pd.read_excel(dataset.file_path)
        except Exception as e:
            session.close()
            return jsonify({'error': f'Failed to load dataset: {str(e)}'}), 500
        
        session.close()
        
        # Perform comprehensive analysis
        overall_stress_level = float(df['stress_level'].mean())
        
        # Enhanced factor importance calculation
        feature_importance = enhanced_factor_importance_calculation(df, 'stress_level', overall_stress_level)
        correlation_insights = dynamic_correlation_analysis(df, 'stress_level')
        
        # Department breakdown
        department_stats = df.groupby('department')['stress_level'].agg(['mean', 'count', 'std']).round(2)
        department_breakdown = {}
        
        for dept in department_stats.index:
            dept_data = department_stats.loc[dept]
            dept_avg_stress = float(dept_data['mean'])
            
            department_breakdown[dept] = {
                'average_stress': dept_avg_stress,
                'employee_count': int(dept_data['count']),
                'stress_std': float(dept_data['std']) if not pd.isna(dept_data['std']) else 0.0,
                'stress_category': 'Critical' if dept_avg_stress >= 70 else 'High' if dept_avg_stress >= 50 else 'Moderate' if dept_avg_stress >= 30 else 'Low',
                'needs_intervention': dept_avg_stress >= 60
            }
        
        # Stress distribution
        high_stress_count = len(df[df['stress_level'] >= 70])
        medium_stress_count = len(df[(df['stress_level'] >= 40) & (df['stress_level'] < 70)])
        low_stress_count = len(df[df['stress_level'] < 40])
        
        stress_distribution = {
            'high_risk': high_stress_count,
            'medium_risk': medium_stress_count,
            'low_risk': low_stress_count,
            'high_risk_percentage': round((high_stress_count / len(df)) * 100, 1),
            'medium_risk_percentage': round((medium_stress_count / len(df)) * 100, 1),
            'low_risk_percentage': round((low_stress_count / len(df)) * 100, 1)
        }
        
        # Prepare data for recommendation engine
        analysis_data_for_recommendations = {
            'overall_stress_level': overall_stress_level,
            'factor_contributions': feature_importance,
            'department_breakdown': department_breakdown,
            'correlation_insights': correlation_insights,
            'total_employees': len(df),
            'stress_distribution': stress_distribution
        }
        
        # Generate dynamic recommendations with implementation steps
        dynamic_recommendations = recommendation_engine.generate_recommendations(
            analysis_data_for_recommendations
        )
        
        # Calculate confidence score based on data quality
        confidence_score = min(1.0, len(df) / 500) * 0.9  # Higher confidence with more data
        
        # Determine stress category and priority
        if overall_stress_level >= 80:
            stress_category = 'Critical'
            urgency_level = 'Immediate Action Required'
        elif overall_stress_level >= 60:
            stress_category = 'High'
            urgency_level = 'Action Required'
        elif overall_stress_level >= 40:
            stress_category = 'Moderate'
            urgency_level = 'Monitor Closely'
        else:
            stress_category = 'Low'
            urgency_level = 'Preventive Measures'
        
        # Enhanced insights
        dominant_factors = sorted(feature_importance.items(), key=lambda x: x[1]['importance_percentage'], reverse=True)[:3]
        high_risk_departments = [dept for dept, data in department_breakdown.items() if data['needs_intervention']]
        
        response = {
            'success': True,
            'dataset_id': dataset_id,
            'summary': {
                'overall_stress_level': overall_stress_level,
                'stress_category': stress_category,
                'urgency_level': urgency_level,
                'total_employees': len(df),
                'confidence_score': confidence_score,
                'dominant_factors': [f[0] for f in dominant_factors],
                'high_risk_departments': high_risk_departments,
                'intervention_needed': overall_stress_level >= 60
            },
            'analysis_results': {
                'factor_contributions': feature_importance,
                'department_breakdown': department_breakdown,
                'stress_distribution': stress_distribution,
                'correlation_insights': correlation_insights
            },
            'recommendations': dynamic_recommendations,  # These include implementation_steps
            'insights': {
                'highest_stress_factor': dominant_factors[0][0] if dominant_factors else 'Unknown',
                'factor_importance_percentage': dominant_factors[0][1]['importance_percentage'] if dominant_factors else 0,
                'departments_needing_intervention': high_risk_departments,
                'risk_employees_count': high_stress_count,
                'intervention_priority': 'High' if overall_stress_level >= 60 else 'Medium' if overall_stress_level >= 40 else 'Low',
                'personalization_applied': True,
                'dynamic_steps_generated': True
            },
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'enhanced_with_dynamic_implementation_steps'
        }
        
        # Log enhanced analysis
        log_system_event("INFO", "API", f"Enhanced analysis with dynamic steps completed: {dataset.name}", {
            'dataset_id': dataset_id,
            'overall_stress_level': overall_stress_level,
            'stress_category': stress_category,
            'recommendations_count': len(dynamic_recommendations),
            'implementation_steps_generated': sum(len(rec.get('implementation_steps', [])) for rec in dynamic_recommendations)
        })
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {str(e)}")
        log_system_event("ERROR", "API", f"Enhanced analysis failed: {str(e)}", {
            'dataset_id': dataset_id
        })
        
        return jsonify({
            'error': f'Enhanced analysis failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 