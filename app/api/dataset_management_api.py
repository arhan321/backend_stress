from flask import Blueprint, request, jsonify
from datetime import datetime
import os
import uuid
import pandas as pd
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import DatabaseManager
from models.database_models import Dataset, Employee, AnalysisResult, StressPrediction, SystemLog
import jwt
import logging

logger = logging.getLogger(__name__)

dataset_management_bp = Blueprint('dataset_management', __name__)
db_manager = DatabaseManager()

# JWT Configuration
JWT_SECRET_KEY = 'stress_analysis_secret_key_2024'
JWT_ALGORITHM = 'HS256'

def verify_jwt_token(token):
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@dataset_management_bp.route('/api/dataset/upload', methods=['POST'])
def upload_dataset():
    """Upload and process employee stress dataset"""
    try:
        # Verify authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No valid token provided'}), 401
        
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        user_id = payload['user_id']
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload CSV, XLS, or XLSX files'}), 400
        
        # Create uploads directory if it doesn't exist
        upload_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        # Save file with unique name
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        # Get additional metadata
        dataset_name = request.form.get('dataset_name', f'Dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        description = request.form.get('description', '')
        
        # Process and validate dataset
        try:
            # Read file based on extension
            if file.filename.lower().endswith('.csv'):
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
                os.remove(file_path)  # Clean up file
                return jsonify({
                    'error': f'Missing required columns: {", ".join(missing_columns)}',
                    'required_columns': required_columns,
                    'found_columns': df.columns.tolist()
                }), 400
            
            # Validate minimum dataset size
            if len(df) < 10:
                os.remove(file_path)  # Clean up file
                return jsonify({
                    'error': f'Dataset too small. Minimum 10 records required, found {len(df)}'
                }), 400
            
            # Validate data ranges
            numeric_columns = ['workload', 'work_life_balance', 'team_conflict', 'management_support', 'work_environment']
            for col in numeric_columns:
                if df[col].min() < 1 or df[col].max() > 10:
                    return jsonify({
                        'error': f'Column {col} must have values between 1-10'
                    }), 400
            
            if df['stress_level'].min() < 0 or df['stress_level'].max() > 100:
                return jsonify({
                    'error': 'Stress level must be between 0-100'
                }), 400
            
            # Save dataset info to database
            session_db = db_manager.get_session()
            
            # Create dataset record
            dataset = Dataset(
                name=dataset_name,
                description=description,
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                record_count=len(df),
                status='uploaded',
                upload_date=datetime.utcnow()
            )
            
            session_db.add(dataset)
            session_db.flush()  # Get dataset ID
            dataset_id = dataset.id
            
            # Process and store employee data
            employees_added = 0
            employees_updated = 0
            
            for _, row in df.iterrows():
                try:
                    # Check if employee exists
                    existing_employee = session_db.query(Employee).filter_by(
                        employee_id=str(row['employee_id'])
                    ).first()
                    
                    if existing_employee:
                        # Update existing employee
                        existing_employee.department = row['department']
                        existing_employee.workload = float(row['workload'])
                        existing_employee.work_life_balance = float(row['work_life_balance'])
                        existing_employee.team_conflict = float(row['team_conflict'])
                        existing_employee.management_support = float(row['management_support'])
                        existing_employee.work_environment = float(row['work_environment'])
                        existing_employee.stress_level = float(row['stress_level'])
                        existing_employee.updated_at = datetime.utcnow()
                        employees_updated += 1
                    else:
                        # Create new employee
                        employee = Employee(
                            employee_id=str(row['employee_id']),
                            department=row['department'],
                            age=row.get('age'),
                            position=row.get('position'),
                            workload=float(row['workload']),
                            work_life_balance=float(row['work_life_balance']),
                            team_conflict=float(row['team_conflict']),
                            management_support=float(row['management_support']),
                            work_environment=float(row['work_environment']),
                            stress_level=float(row['stress_level'])
                        )
                        session_db.add(employee)
                        employees_added += 1
                        
                except Exception as row_error:
                    logger.warning(f"Error processing row {row['employee_id']}: {str(row_error)}")
                    continue
            
            # Update dataset status
            dataset.status = 'processed'
            dataset.processing_end_time = datetime.utcnow()
            
            # Create system log
            system_log = SystemLog(
                level='INFO',
                module='dataset_upload',
                message=f'Dataset {dataset_name} uploaded and processed successfully',
                additional_data={
                    'dataset_id': dataset_id,
                    'user_id': user_id,
                    'employees_added': employees_added,
                    'employees_updated': employees_updated,
                    'total_records': len(df)
                }
            )
            session_db.add(system_log)
            
            session_db.commit()
            session_db.close()
            
            # Calculate basic statistics
            departments = df['department'].unique().tolist()
            avg_stress = float(df['stress_level'].mean())
            
            logger.info(f"Dataset uploaded successfully: {dataset_name} with {len(df)} records")
            
            return jsonify({
                'success': True,
                'message': 'Dataset uploaded and processed successfully',
                'dataset': {
                    'id': dataset_id,
                    'name': dataset_name,
                    'record_count': len(df),
                    'employees_added': employees_added,
                    'employees_updated': employees_updated,
                    'departments': departments,
                    'average_stress_level': round(avg_stress, 2),
                    'upload_date': dataset.upload_date.isoformat()
                }
            }), 201
            
        except Exception as processing_error:
            # Clean up file and database record if processing fails
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({
                'error': f'Failed to process dataset: {str(processing_error)}'
            }), 400
        
    except Exception as e:
        logger.error(f"Dataset upload failed: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@dataset_management_bp.route('/api/dataset/list', methods=['GET'])
def list_datasets():
    """Get list of all datasets for the user"""
    try:
        # Verify authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No valid token provided'}), 401
        
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        session_db = db_manager.get_session()
        
        # Get all datasets
        datasets = session_db.query(Dataset).order_by(Dataset.upload_date.desc()).all()
        
        dataset_list = []
        for dataset in datasets:
            dataset_info = {
                'id': dataset.id,
                'name': dataset.name,
                'description': dataset.description,
                'record_count': dataset.record_count,
                'file_size': dataset.file_size,
                'status': dataset.status,
                'upload_date': dataset.upload_date.isoformat() if dataset.upload_date else None,
                'processing_time': None
            }
            
            # Calculate processing time if available
            if dataset.processing_end_time and dataset.upload_date:
                processing_time = (dataset.processing_end_time - dataset.upload_date).total_seconds()
                dataset_info['processing_time'] = round(processing_time, 2)
            
            dataset_list.append(dataset_info)
        
        session_db.close()
        
        return jsonify({
            'success': True,
            'datasets': dataset_list,
            'total_count': len(dataset_list)
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {str(e)}")
        return jsonify({'error': f'Failed to list datasets: {str(e)}'}), 500

@dataset_management_bp.route('/api/dataset/<int:dataset_id>', methods=['DELETE'])
def delete_dataset():
    """Delete a dataset and associated data"""
    try:
        # Verify authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No valid token provided'}), 401
        
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        user_id = payload['user_id']
        dataset_id = request.view_args['dataset_id']
        
        session_db = db_manager.get_session()
        
        # Get dataset
        dataset = session_db.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            session_db.close()
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Delete associated file
        if dataset.file_path and os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        
        # Delete associated analysis results
        session_db.query(AnalysisResult).filter_by(dataset_id=dataset_id).delete()
        
        # Delete dataset record
        session_db.delete(dataset)
        
        # Create system log
        system_log = SystemLog(
            level='INFO',
            module='dataset_management',
            message=f'Dataset {dataset.name} deleted by user {user_id}',
            additional_data={'dataset_id': dataset_id, 'user_id': user_id}
        )
        session_db.add(system_log)
        
        session_db.commit()
        session_db.close()
        
        logger.info(f"Dataset {dataset_id} deleted successfully")
        
        return jsonify({
            'success': True,
            'message': 'Dataset deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to delete dataset: {str(e)}")
        return jsonify({'error': f'Failed to delete dataset: {str(e)}'}), 500 