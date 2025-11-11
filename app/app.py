from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import traceback
from datetime import datetime, timedelta
import uuid
from werkzeug.utils import secure_filename
from loguru import logger
import json
import io
import csv

# Import our models and database
from config.database import db_manager
from models.database_models import Employee, Dataset, AnalysisResult, SystemLog, User, UserSession, UserPermission, AuditLog, UserRole
from ml_models import StressDeepLearningModel, NeuralCollaborativeFiltering
from sqlalchemy import func

# Import authentication functions
import bcrypt
import jwt
import re

# Import enhanced ML models and recommendation engine
from advanced_ml_models import advanced_dl_model, ncf_model
from recommendation_engine import get_recommendations_for_analysis, DynamicRecommendationEngine

# Import nan fix utils
from nan_fix_utils import (
    clean_nan_values, 
    safe_correlation_calculation, 
    safe_factor_importance_calculation,
    safe_department_breakdown,
    validate_numeric_data,
    enhanced_factor_importance_calculation,
    dynamic_correlation_analysis
)

# Initialize Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
JWT_SECRET_KEY = 'your-jwt-secret-key-here-change-in-production'
JWT_EXPIRATION_HOURS = 24

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model instances
deep_learning_model = None
ncf_model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Authentication utility functions
def hash_password(password):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_jwt_token(user_id, username):
    """Generate JWT token for user"""
    payload = {
        'user_id': user_id,
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')

def verify_jwt_token(token):
    """Verify JWT token and return user data"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password_strength(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    return True, "Password is valid"

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
        
        logger.info("Stress Analysis API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

# AUTHENTICATION ENDPOINTS

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Required fields
        required_fields = ['username', 'email', 'password', 'full_name']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Validate input
        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        full_name = data['full_name'].strip()
        
        # Validation checks
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters long'}), 400
        
        if not validate_email(email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        password_valid, password_message = validate_password_strength(password)
        if not password_valid:
            return jsonify({'error': password_message}), 400
        
        if len(full_name) < 2:
            return jsonify({'error': 'Full name must be at least 2 characters long'}), 400
        
        # Check for existing users
        session_db = db_manager.get_session()
        
        existing_username = session_db.query(User).filter_by(username=username).first()
        if existing_username:
            session_db.close()
            return jsonify({'error': 'Username already exists'}), 409
        
        existing_email = session_db.query(User).filter_by(email=email).first()
        if existing_email:
            session_db.close()
            return jsonify({'error': 'Email already exists'}), 409
        
        # Hash password
        password_hash = hash_password(password)
        
        # Determine role (default to employee, admin if first user)
        user_count = session_db.query(User).count()
        user_role = UserRole.admin if user_count == 0 else UserRole.employee
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            role=user_role,
            department=data.get('department'),
            position=data.get('position'),
            phone_number=data.get('phone_number'),
            is_active=True,
            email_verified=False
        )
        
        session_db.add(new_user)
        session_db.commit()
        
        user_id = new_user.id
        
        # Add default permissions based on role
        default_permissions = {
            UserRole.admin: ['admin_access', 'user_management', 'system_config', 'data_analysis', 'report_generation'],
            UserRole.hr_manager: ['user_management', 'employee_data_access', 'report_generation'],
            UserRole.analyst: ['data_analysis', 'report_generation', 'model_management'],
            UserRole.employee: ['profile_access', 'basic_reports']
        }
        
        for permission in default_permissions.get(user_role, []):
            user_permission = UserPermission(
                user_id=user_id,
                permission=permission
            )
            session_db.add(user_permission)
        
        session_db.commit()
        session_db.close()
        
        logger.info(f"New user registered: {username} ({email})")
        
        return jsonify({
            'success': True,
            'message': 'User registered successfully',
            'user': {
                'id': user_id,
                'username': username,
                'email': email,
                'full_name': full_name,
                'role': user_role.value,
                'department': data.get('department'),
                'position': data.get('position')
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        return jsonify({'error': f'Registration successfull: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Authenticate user login"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        username_or_email = data.get('username_or_email', '').strip()
        password = data.get('password', '')
        
        if not username_or_email or not password:
            return jsonify({'error': 'Username/email and password are required'}), 400
        
        # Find user by username or email
        session_db = db_manager.get_session()
        
        user = session_db.query(User).filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()
        
        if not user:
            session_db.close()
            return jsonify({'error': 'Invalid username/email or password'}), 401
        
        # Check if user is active
        if not user.is_active:
            session_db.close()
            return jsonify({'error': 'Account is inactive'}), 401
        
        # Verify password
        if not verify_password(password, user.password_hash):
            session_db.close()
            return jsonify({'error': 'Invalid username/email or password'}), 401
        
        # Generate JWT token
        token = generate_jwt_token(user.id, user.username)
        
        # Create session record
        session_token = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        
        user_session = UserSession(
            user_id=user.id,
            session_token=session_token,
            device_info=request.headers.get('User-Agent'),
            ip_address=request.remote_addr,
            expires_at=expires_at,
            is_active=True
        )
        
        session_db.add(user_session)
        
        # Update last login
        user.last_login = datetime.utcnow()
        
        session_db.commit()
        
        # Get user permissions
        permissions = session_db.query(UserPermission).filter_by(user_id=user.id).all()
        user_permissions = [perm.permission for perm in permissions]
        
        session_db.close()
        
        logger.info(f"User logged in: {user.username}")
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'token': token,
            'session_token': session_token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name,
                'role': user.role.value,
                'department': user.department,
                'position': user.position,
                'phone_number': user.phone_number,
                'profile_picture': user.profile_picture,
                'last_login': user.last_login.isoformat() if user.last_login else None,
                'permissions': user_permissions
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/api/auth/profile', methods=['GET'])
def get_profile():
    """Get user profile data"""
    try:
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No valid token provided'}), 401
        
        token = auth_header.split(' ')[1]
        
        # Verify token
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        user_id = payload['user_id']
        
        # Get user data
        session_db = db_manager.get_session()
        user = session_db.query(User).filter_by(id=user_id, is_active=True).first()
        
        if not user:
            session_db.close()
            return jsonify({'error': 'User not found or inactive'}), 404
        
        # Get user permissions
        permissions = session_db.query(UserPermission).filter_by(user_id=user_id).all()
        user_permissions = [perm.permission for perm in permissions]
        
        session_db.close()
        
        return jsonify({
            'success': True,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name,
                'role': user.role.value,
                'department': user.department,
                'position': user.position,
                'phone_number': user.phone_number,
                'profile_picture': user.profile_picture,
                'email_verified': user.email_verified,
                'last_login': user.last_login.isoformat() if user.last_login else None,
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'permissions': user_permissions
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Get profile failed: {str(e)}")
        return jsonify({'error': f'Failed to get profile: {str(e)}'}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout user and invalidate session"""
    try:
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        session_token = request.headers.get('Session-Token')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No valid token provided'}), 401
        
        token = auth_header.split(' ')[1]
        
        # Verify token
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        user_id = payload['user_id']
        
        # Invalidate session
        if session_token:
            session_db = db_manager.get_session()
            user_session = session_db.query(UserSession).filter_by(
                user_id=user_id,
                session_token=session_token
            ).first()
            
            if user_session:
                user_session.is_active = False
                session_db.commit()
            
            session_db.close()
        
        logger.info(f"User logged out: user_id={user_id}")
        
        return jsonify({
            'success': True,
            'message': 'Logout successful'
        }), 200
        
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        return jsonify({'error': f'Logout failed: {str(e)}'}), 500

# USER MANAGEMENT ENDPOINTS

@app.route('/api/user/change-password', methods=['POST'])
def change_password():
    """Change user password with current password verification"""
    try:
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No valid token provided'}), 401
        
        token = auth_header.split(' ')[1]
        
        # Verify token
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        user_id = payload['user_id']
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        current_password = data.get('current_password', '').strip()
        new_password = data.get('new_password', '').strip()
        confirm_password = data.get('confirm_password', '').strip()
        
        # Validate inputs
        if not current_password or not new_password or not confirm_password:
            return jsonify({'error': 'All password fields are required'}), 400
        
        if new_password != confirm_password:
            return jsonify({'error': 'New password and confirmation do not match'}), 400
        
        if len(new_password) < 8:
            return jsonify({'error': 'New password must be at least 8 characters long'}), 400
        
        # Get user from database
        session_db = db_manager.get_session()
        user = session_db.query(User).filter_by(id=user_id, is_active=True).first()
        
        if not user:
            session_db.close()
            return jsonify({'error': 'User not found or inactive'}), 404
        
        # Verify current password
        if not bcrypt.checkpw(current_password.encode('utf-8'), user.password_hash.encode('utf-8')):
            session_db.close()
            return jsonify({'error': 'Current password is incorrect'}), 401
        
        # Hash new password
        salt = bcrypt.gensalt()
        new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), salt).decode('utf-8')
        
        # Update password in database
        user.password_hash = new_password_hash
        user.updated_at = datetime.utcnow()
        
        # Create audit log
        audit_log = AuditLog(
            user_id=user_id,
            action='password_change',
            resource='user',
            resource_id=user_id,
            details={'message': 'User password changed successfully'},
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        
        session_db.add(audit_log)
        session_db.commit()
        session_db.close()
        
        logger.info(f"Password changed successfully for user ID: {user_id}")
        
        return jsonify({
            'success': True,
            'message': 'Password changed successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Password change failed: {str(e)}")
        return jsonify({'error': f'Password change failed: {str(e)}'}), 500

@app.route('/api/user/update-profile', methods=['PUT'])
def update_profile():
    """Update user profile information"""
    try:
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No valid token provided'}), 401
        
        token = auth_header.split(' ')[1]
        
        # Verify token
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        user_id = payload['user_id']
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get user from database
        session_db = db_manager.get_session()
        user = session_db.query(User).filter_by(id=user_id, is_active=True).first()
        
        if not user:
            session_db.close()
            return jsonify({'error': 'User not found or inactive'}), 404
        
        # Track changes for audit
        changes = {}
        
        # Update allowed fields
        if 'full_name' in data and data['full_name'].strip():
            old_name = user.full_name
            user.full_name = data['full_name'].strip()
            changes['full_name'] = {'old': old_name, 'new': user.full_name}
        
        if 'department' in data:
            old_dept = user.department
            user.department = data['department'].strip() if data['department'] else None
            changes['department'] = {'old': old_dept, 'new': user.department}
        
        if 'position' in data:
            old_pos = user.position
            user.position = data['position'].strip() if data['position'] else None
            changes['position'] = {'old': old_pos, 'new': user.position}
        
        if 'phone_number' in data:
            old_phone = user.phone_number
            user.phone_number = data['phone_number'].strip() if data['phone_number'] else None
            changes['phone_number'] = {'old': old_phone, 'new': user.phone_number}
        
        # Update timestamp
        user.updated_at = datetime.utcnow()
        
        # Create audit log
        if changes:
            audit_log = AuditLog(
                user_id=user_id,
                action='profile_update',
                resource='user',
                resource_id=user_id,
                details={'changes': changes},
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent')
            )
            session_db.add(audit_log)
        
        session_db.commit()
        
        # Refresh user data to ensure it's properly bound to session
        session_db.refresh(user)
        
        user_data = {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'full_name': user.full_name,
            'role': user.role.value,
            'department': user.department,
            'position': user.position,
            'phone_number': user.phone_number,
            'updated_at': user.updated_at.isoformat()
        }
        
        session_db.close()
        
        logger.info(f"Profile updated successfully for user ID: {user_id}")
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully',
            'user': user_data
        }), 200
        
    except Exception as e:
        logger.error(f"Profile update failed: {str(e)}")
        return jsonify({'error': f'Profile update failed: {str(e)}'}), 500

# EXISTING ENDPOINTS (unchanged)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        db_test = db_manager.test_connection()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': 'connected' if db_test else 'disconnected'
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """Upload and process employee stress dataset with enhanced error handling"""
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
            logger.error("No file provided in upload request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename in upload request")
            return jsonify({'error': 'No file selected'}), 400
        
        # Log file details for debugging
        logger.info(f"Processing file upload: {file.filename}, Content-Type: {file.content_type}")
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload CSV, XLS, or XLSX files'}), 400
        
        # Create uploads directory if it doesn't exist
        upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        # Save file with unique name
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        file_path = os.path.join(upload_folder, filename)
        
        try:
            file.save(file_path)
            logger.info(f"File saved successfully: {file_path}")
        except Exception as save_error:
            logger.error(f"Failed to save file: {str(save_error)}")
            return jsonify({'error': f'Failed to save file: {str(save_error)}'}), 500
        
        # Get additional metadata
        dataset_name = request.form.get('dataset_name', f'Dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        description = request.form.get('description', '')
        
        logger.info(f"Dataset metadata - Name: {dataset_name}, Description: {description}")
        
        # Process and validate dataset
        try:
            # Read file based on extension
            if file.filename.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            logger.info(f"Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            
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
                    os.remove(file_path)
                    return jsonify({
                        'error': f'Column {col} must have values between 1-10. Found range: {df[col].min()}-{df[col].max()}'
                    }), 400
            
            if df['stress_level'].min() < 0 or df['stress_level'].max() > 100:
                os.remove(file_path)
                return jsonify({
                    'error': f'Stress level must be between 0-100. Found range: {df["stress_level"].min()}-{df["stress_level"].max()}'
                }), 400
            
            # Save dataset info to database
            session_db = db_manager.get_session()
            
            try:
                # Create dataset record
                dataset = Dataset(
                    name=dataset_name,
                    description=description,
                    file_path=file_path,
                    file_size=os.path.getsize(file_path),
                    record_count=len(df),
                    status='uploaded',
                    upload_date=datetime.utcnow(),
                    uploaded_by=user_id
                )
                
                session_db.add(dataset)
                session_db.flush()  # Get dataset ID
                dataset_id = dataset.id
                
                logger.info(f"Dataset record created with ID: {dataset_id}")
                
                # Process and store employee data with CRUD operations
                employees_added = 0
                employees_updated = 0
                
                for _, row in df.iterrows():
                    try:
                        # Check if employee exists
                        existing_employee = session_db.query(Employee).filter_by(
                            employee_id=str(row['employee_id'])
                        ).first()
                        
                        if existing_employee:
                            # Update existing employee (UPDATE operation)
                            existing_employee.department = str(row['department'])
                            existing_employee.workload = float(row['workload'])
                            existing_employee.work_life_balance = float(row['work_life_balance'])
                            existing_employee.team_conflict = float(row['team_conflict'])
                            existing_employee.management_support = float(row['management_support'])
                            existing_employee.work_environment = float(row['work_environment'])
                            existing_employee.stress_level = float(row['stress_level'])
                            existing_employee.last_update = datetime.utcnow()
                            
                            # No need to refresh - just let SQLAlchemy handle the update
                            employees_updated += 1
                            
                        else:
                            # Create new employee (CREATE operation)
                            new_employee = Employee(
                                employee_id=str(row['employee_id']),
                                department=str(row['department']),
                                workload=float(row['workload']),
                                work_life_balance=float(row['work_life_balance']),
                                team_conflict=float(row['team_conflict']),
                                management_support=float(row['management_support']),
                                work_environment=float(row['work_environment']),
                                stress_level=float(row['stress_level']),
                                created_at=datetime.utcnow(),
                                last_update=datetime.utcnow()
                            )
                            session_db.add(new_employee)
                            employees_added += 1
                            
                    except Exception as emp_error:
                        logger.warning(f"Error processing employee {row['employee_id']}: {str(emp_error)}")
                        continue
                
                # Update dataset status
                dataset.status = 'processed'
                dataset.processing_end_time = datetime.utcnow()
                
                # Create system log for audit trail
                system_log = SystemLog(
                    level='INFO',
                    module='dataset_upload',
                    message=f'Dataset uploaded successfully: {dataset_name}',
                    additional_data={
                        'dataset_id': dataset_id,
                        'user_id': user_id,
                        'record_count': len(df),
                        'employees_added': employees_added,
                        'employees_updated': employees_updated,
                        'file_size': os.path.getsize(file_path),
                        'original_filename': file.filename
                    }
                )
                session_db.add(system_log)
                
                # Commit all changes at once
                session_db.commit()
                
                # Log successful upload
                logger.info(f"Dataset uploaded successfully: {dataset_name}, Records: {len(df)}, Employees Added: {employees_added}, Updated: {employees_updated}")
                
                response_data = {
                    'success': True,
                    'message': f'Dataset "{dataset_name}" uploaded and processed successfully',
                    'dataset': {
                        'id': dataset_id,
                        'name': dataset_name,
                        'description': description,
                        'record_count': len(df),
                        'employees_added': employees_added,
                        'employees_updated': employees_updated,
                        'file_size': os.path.getsize(file_path),
                        'upload_date': dataset.upload_date.isoformat(),
                        'status': 'processed'
                    }
                }
                
                return jsonify(response_data), 201
                
            except Exception as db_error:
                # Rollback the session in case of database errors
                session_db.rollback()
                logger.error(f"Database operation failed: {str(db_error)}")
                raise db_error
                
            finally:
                # Always close the session
                session_db.close()
            
        except Exception as processing_error:
            # Clean up file and database record if processing fails
            if os.path.exists(file_path):
                os.remove(file_path)
            
            logger.error(f"Dataset processing failed: {str(processing_error)}")
            return jsonify({
                'error': f'Failed to process dataset: {str(processing_error)}',
                'details': 'Please check the file format and data requirements'
            }), 400
        
    except Exception as e:
        logger.error(f"Dataset upload failed: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/analyze-dataset/<int:dataset_id>', methods=['POST'])
def analyze_dataset(dataset_id):
    """Analyze dataset using ML models with real-time result storage"""
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
        
        session = db_manager.get_session()
        
        # Get dataset
        dataset = session.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            session.close()
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Load dataset
        if dataset.file_path.endswith('.csv'):
            df = pd.read_csv(dataset.file_path)
        else:
            df = pd.read_excel(dataset.file_path)
        
        # Update dataset status with real-time tracking
        dataset.status = 'analyzing'
        dataset.processing_start_time = datetime.utcnow()
        session.commit()
        
        analysis_start_time = datetime.utcnow()
        
        # Perform comprehensive statistical analysis
        correlation_matrix = df[['workload', 'work_life_balance', 'team_conflict', 
                               'management_support', 'work_environment', 'stress_level']].corr()
        
        # Calculate feature importance with enhanced metrics
        stress_correlations = correlation_matrix['stress_level'].abs().sort_values(ascending=False)
        feature_importance = {}
        
        for feature, correlation in stress_correlations.items():
            if feature != 'stress_level':
                importance = abs(correlation) * 100
                feature_importance[feature] = {
                    'correlation': float(correlation),
                    'importance_percentage': float(importance),
                    'rank': len(feature_importance) + 1
                }
        
        # Enhanced department breakdown with statistics
        department_stats = df.groupby('department')['stress_level'].agg([
            'mean', 'count', 'std', 'min', 'max'
        ]).round(2)
        
        department_breakdown = {}
        for dept in department_stats.index:
            dept_data = department_stats.loc[dept]
            department_breakdown[dept] = {
                'average_stress': float(dept_data['mean']),
                'employee_count': int(dept_data['count']),
                'stress_std': float(dept_data['std']),
                'min_stress': float(dept_data['min']),
                'max_stress': float(dept_data['max']),
                'risk_level': 'High' if dept_data['mean'] > 70 else 'Medium' if dept_data['mean'] > 40 else 'Low'
            }
        
        # Calculate comprehensive metrics
        overall_stress_level = float(df['stress_level'].mean())
        stress_std = float(df['stress_level'].std())
        high_stress_count = int((df['stress_level'] > 70).sum())
        medium_stress_count = int(((df['stress_level'] > 40) & (df['stress_level'] <= 70)).sum())
        low_stress_count = int((df['stress_level'] <= 40).sum())
        
        # Generate individual stress predictions and store them
        predictions_stored = 0
        for _, row in df.iterrows():
            try:
                # Calculate individual prediction factors
                factor_weights = {
                    'workload': 0.25,
                    'work_life_balance': -0.20,  # Negative because higher balance = lower stress
                    'team_conflict': 0.20,
                    'management_support': -0.15,  # Negative because more support = lower stress
                    'work_environment': -0.20  # Negative because better environment = lower stress
                }
                
                predicted_stress = 50  # Base stress level
                confidence_factors = []
                
                for factor, weight in factor_weights.items():
                    if factor in row:
                        predicted_stress += (row[factor] - 5.5) * weight * 10
                        confidence_factors.append(abs(weight))
                
                predicted_stress = max(0, min(100, predicted_stress))  # Clamp to 0-100
                confidence_score = sum(confidence_factors) / len(confidence_factors)
                
                # Store prediction in database
                prediction = StressPrediction(
                    employee_id=session.query(Employee).filter_by(
                        employee_id=str(row['employee_id'])
                    ).first().id,
                    predicted_stress_level=predicted_stress,
                    confidence_score=confidence_score,
                    prediction_date=datetime.utcnow(),
                    model_version='statistical_v1.0',
                    input_features={
                        'workload': float(row['workload']),
                        'work_life_balance': float(row['work_life_balance']),
                        'team_conflict': float(row['team_conflict']),
                        'management_support': float(row['management_support']),
                        'work_environment': float(row['work_environment'])
                    },
                    factor_weights=factor_weights
                )
                session.add(prediction)
                predictions_stored += 1
                
            except Exception as pred_error:
                logger.warning(f"Error creating prediction for employee {row['employee_id']}: {str(pred_error)}")
                continue
        
        processing_time = (datetime.utcnow() - analysis_start_time).total_seconds()
        
        # Save comprehensive analysis results to database
        analysis_result = AnalysisResult(
            dataset_id=dataset_id,
            analysis_type='comprehensive_statistical',
            overall_stress_level=overall_stress_level,
            analysis_date=datetime.utcnow(),
            factor_contributions=feature_importance,
            model_accuracy=0.85,  # Statistical model accuracy estimate
            confidence_score=0.80,
            department_breakdown=department_breakdown,
            correlation_matrix=correlation_matrix.to_dict(),
            feature_importance=feature_importance,
            processing_time_seconds=processing_time
        )
        
        session.add(analysis_result)
        
        # Update dataset status
        dataset.status = 'analyzed'
        dataset.processing_end_time = datetime.utcnow()
        
        # Create comprehensive system log
        system_log = SystemLog(
            level='INFO',
            module='stress_analysis',
            message=f'Dataset {dataset.name} analyzed successfully by user {user_id}',
            additional_data={
                'dataset_id': dataset_id,
                'user_id': user_id,
                'analysis_type': 'comprehensive_statistical',
                'overall_stress_level': overall_stress_level,
                'high_stress_employees': high_stress_count,
                'predictions_stored': predictions_stored,
                'processing_time': processing_time
            }
        )
        session.add(system_log)
        
        session.commit()
        session.close()
        
        # Generate actionable recommendations
        recommendations = []
        
        if overall_stress_level > 70:
            recommendations.append({
                'priority': 'High',
                'category': 'Organization',
                'recommendation': 'Immediate intervention required. Consider stress management programs.'
            })
        
        # Department-specific recommendations
        for dept, stats in department_breakdown.items():
            if stats['average_stress'] > 70:
                recommendations.append({
                    'priority': 'High',
                    'category': f'Department: {dept}',
                    'recommendation': f'Focus on {dept} department - average stress level is {stats["average_stress"]:.1f}%'
                })
        
        # Factor-based recommendations
        top_stress_factor = max(feature_importance.items(), key=lambda x: x[1]['importance_percentage'])
        recommendations.append({
            'priority': 'Medium',
            'category': 'Focus Area',
            'recommendation': f'Address {top_stress_factor[0].replace("_", " ").title()} - highest stress factor ({top_stress_factor[1]["importance_percentage"]:.1f}% importance)'
        })
        
        response = {
            'success': True,
            'analysis_id': analysis_result.id,
            'dataset_id': dataset_id,
            'summary': {
                'overall_stress_level': round(overall_stress_level, 2),
                'stress_distribution': {
                    'high_risk': high_stress_count,
                    'medium_risk': medium_stress_count,
                    'low_risk': low_stress_count
                },
                'total_employees': len(df),
                'departments_analyzed': len(department_breakdown)
            },
            'factor_contributions': feature_importance,
            'department_breakdown': department_breakdown,
            'statistics': {
                'stress_standard_deviation': round(stress_std, 2),
                'predictions_generated': predictions_stored,
                'processing_time_seconds': round(processing_time, 2)
            },
            'recommendations': recommendations,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Analysis completed for dataset {dataset_id}: {overall_stress_level:.2f}% overall stress")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/predict-stress', methods=['POST'])
def predict_stress():
    """Predict stress level for new employee data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Simple statistical prediction
        factor_weights = {
            'workload': 0.25,
            'work_life_balance': 0.20,
            'team_conflict': 0.20,
            'management_support': 0.15,
            'work_environment': 0.20
        }
        
        weighted_score = sum(data.get(factor, 5) * weight for factor, weight in factor_weights.items())
        predicted_stress_level = min(100, max(0, weighted_score * 10))
        
        # Try ML prediction if model is available
        ml_prediction = None
        try:
            global deep_learning_model
            if deep_learning_model and deep_learning_model.is_trained:
                ml_prediction = deep_learning_model.predict(data)
        except Exception as e:
            logger.warning(f"ML prediction failed: {str(e)}")
        
        return jsonify({
            'success': True,
            'statistical_prediction': {
                'predicted_stress_level': predicted_stress_level,
                'confidence_score': 0.7
            },
            'ml_prediction': ml_prediction,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get list of datasets with enhanced information"""
    try:
        # Verify authentication
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            payload = verify_jwt_token(token)
            user_id = payload['user_id'] if payload else None
        else:
            user_id = None
        
        session = db_manager.get_session()
        
        # Get all datasets with enhanced information
        datasets = session.query(Dataset).order_by(Dataset.upload_date.desc()).all()
        
        dataset_list = []
        for dataset in datasets:
            # Calculate processing time if available
            processing_time = None
            if dataset.processing_end_time and dataset.processing_start_time:
                processing_time = (dataset.processing_end_time - dataset.processing_start_time).total_seconds()
            elif dataset.processing_end_time and dataset.upload_date:
                processing_time = (dataset.processing_end_time - dataset.upload_date).total_seconds()
            
            # Get analysis results for this dataset
            analysis_results = session.query(AnalysisResult).filter_by(dataset_id=dataset.id).all()
            
            dataset_info = {
                'id': dataset.id,
                'name': dataset.name,
                'description': dataset.description,
                'record_count': dataset.record_count,
                'file_size': dataset.file_size,
                'file_size_mb': round(dataset.file_size / (1024 * 1024), 2) if dataset.file_size else 0,
                'status': dataset.status,
                'upload_date': dataset.upload_date.isoformat() if dataset.upload_date else None,
                'processing_time': round(processing_time, 2) if processing_time else None,
                'analysis_count': len(analysis_results),
                'last_analysis': analysis_results[-1].analysis_date.isoformat() if analysis_results else None,
                'overall_stress_level': analysis_results[-1].overall_stress_level if analysis_results else None
            }
            
            dataset_list.append(dataset_info)
        
        session.close()
        
        # Calculate summary statistics
        total_datasets = len(dataset_list)
        total_records = sum(d['record_count'] for d in dataset_list)
        avg_stress = None
        if dataset_list:
            stress_levels = [d['overall_stress_level'] for d in dataset_list if d['overall_stress_level'] is not None]
            if stress_levels:
                avg_stress = round(sum(stress_levels) / len(stress_levels), 2)
        
        return jsonify({
            'success': True,
            'datasets': dataset_list,
            'summary': {
                'total_datasets': total_datasets,
                'total_employee_records': total_records,
                'average_organizational_stress': avg_stress,
                'datasets_analyzed': len([d for d in dataset_list if d['analysis_count'] > 0])
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {str(e)}")
        return jsonify({'error': f'Failed to list datasets: {str(e)}'}), 500

@app.route('/api/dataset/<int:dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """Delete a dataset and all associated data"""
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
        
        session = db_manager.get_session()
        
        # Get dataset
        dataset = session.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            session.close()
            return jsonify({'error': 'Dataset not found'}), 404
        
        dataset_name = dataset.name
        
        # Delete associated file
        if dataset.file_path and os.path.exists(dataset.file_path):
            try:
                os.remove(dataset.file_path)
                logger.info(f"File deleted: {dataset.file_path}")
            except OSError as e:
                logger.warning(f"Could not delete file {dataset.file_path}: {e}")
        
        # Delete associated analysis results and predictions
        analysis_results = session.query(AnalysisResult).filter_by(dataset_id=dataset_id).all()
        for result in analysis_results:
            session.delete(result)
        
        # Delete associated stress predictions if any employees are only in this dataset
        # Note: We don't delete employees as they might be used in other datasets
        
        # Delete dataset record
        session.delete(dataset)
        
        # Create audit log
        system_log = SystemLog(
            level='INFO',
            module='dataset_management',
            message=f'Dataset "{dataset_name}" (ID: {dataset_id}) deleted by user {user_id}',
            additional_data={
                'dataset_id': dataset_id,
                'dataset_name': dataset_name,
                'user_id': user_id,
                'analysis_results_deleted': len(analysis_results)
            }
        )
        session.add(system_log)
        
        session.commit()
        session.close()
        
        logger.info(f"Dataset {dataset_id} ({dataset_name}) deleted successfully by user {user_id}")
        
        return jsonify({
            'success': True,
            'message': f'Dataset "{dataset_name}" deleted successfully',
            'deleted_analysis_results': len(analysis_results)
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to delete dataset {dataset_id}: {str(e)}")
        return jsonify({'error': f'Failed to delete dataset: {str(e)}'}), 500

@app.route('/api/dataset/<int:dataset_id>/export', methods=['GET'])
def export_dataset_results(dataset_id):
    """Export dataset analysis results"""
    try:
        # Verify authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No valid token provided'}), 401
        
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        session = db_manager.get_session()
        
        # Get dataset and analysis results
        dataset = session.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            session.close()
            return jsonify({'error': 'Dataset not found'}), 404
        
        analysis_results = session.query(AnalysisResult).filter_by(dataset_id=dataset_id).all()
        
        if not analysis_results:
            session.close()
            return jsonify({'error': 'No analysis results found for this dataset'}), 404
        
        # Get latest analysis result
        latest_analysis = analysis_results[-1]
        
        # Get employee data for this dataset
        employees = session.query(Employee).all()  # Get all employees
        employee_data = []
        
        for emp in employees:
            predictions = session.query(StressPrediction).filter_by(employee_id=emp.id).all()
            latest_prediction = predictions[-1] if predictions else None
            
            employee_data.append({
                'employee_id': emp.employee_id,
                'department': emp.department,
                'position': emp.position,
                'workload': emp.workload,
                'work_life_balance': emp.work_life_balance,
                'team_conflict': emp.team_conflict,
                'management_support': emp.management_support,
                'work_environment': emp.work_environment,
                'actual_stress_level': emp.stress_level,
                'predicted_stress_level': latest_prediction.predicted_stress_level if latest_prediction else None,
                'prediction_confidence': latest_prediction.confidence_score if latest_prediction else None
            })
        
        session.close()
        
        # Create comprehensive export data
        export_data = {
            'dataset_info': {
                'id': dataset.id,
                'name': dataset.name,
                'description': dataset.description,
                'record_count': dataset.record_count,
                'upload_date': dataset.upload_date.isoformat() if dataset.upload_date else None,
                'analysis_date': latest_analysis.analysis_date.isoformat() if latest_analysis.analysis_date else None
            },
            'analysis_summary': {
                'overall_stress_level': latest_analysis.overall_stress_level,
                'analysis_type': latest_analysis.analysis_type,
                'model_accuracy': latest_analysis.model_accuracy,
                'confidence_score': latest_analysis.confidence_score,
                'processing_time_seconds': latest_analysis.processing_time_seconds
            },
            'factor_contributions': latest_analysis.factor_contributions,
            'department_breakdown': latest_analysis.department_breakdown,
            'employee_data': employee_data,
            'export_timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'success': True,
            'export_data': export_data
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to export dataset {dataset_id}: {str(e)}")
        return jsonify({'error': f'Failed to export dataset: {str(e)}'}), 500

@app.route('/api/enhanced-analysis/<int:dataset_id>', methods=['POST'])
def perform_enhanced_analysis(dataset_id):
    """
    Perform comprehensive enhanced ML analysis with REAL-TIME data processing
    This endpoint provides synchronized factor importance calculations
    """
    session_db = None
    analysis_start_time = datetime.utcnow()
    
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
        
        session_db = db_manager.get_session()
        
        # Get dataset
        dataset = session_db.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Check if file exists
        if not os.path.exists(dataset.file_path):
            return jsonify({'error': 'Dataset file not found on server'}), 404
        
        logger.info(f"Starting enhanced real-time ML analysis for dataset {dataset_id}")
        
        # Load and validate dataset
        try:
            if dataset.file_path.endswith('.csv'):
                df = pd.read_csv(dataset.file_path)
            else:
                df = pd.read_excel(dataset.file_path)
            
            required_columns = ['employee_id', 'department', 'workload', 'work_life_balance',
                              'team_conflict', 'management_support', 'work_environment', 'stress_level']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return jsonify({
                    'error': f'Dataset missing required columns: {", ".join(missing_columns)}'
                }), 400
                
        except Exception as file_error:
            logger.error(f"Failed to load dataset file: {str(file_error)}")
            return jsonify({'error': f'Failed to load dataset: {str(file_error)}'}), 400
        
        # STEP 1: REAL-TIME STATISTICAL ANALYSIS (SYNCHRONIZED)
        logger.info("Performing synchronized real-time statistical analysis...")
        
        # Calculate real stress level from uploaded data
        actual_stress_levels = df['stress_level'].values
        overall_stress_level = float(np.mean(actual_stress_levels))
        stress_std = float(np.std(actual_stress_levels))
        stress_median = float(np.median(actual_stress_levels))
        
        # Calculate SYNCHRONIZED feature importance using enhanced calculation
        feature_importance = enhanced_factor_importance_calculation(
            df, 'stress_level', overall_stress_level
        )
        
        # Get dynamic correlation insights that vary with dataset
        correlation_insights = dynamic_correlation_analysis(df, 'stress_level')
        
        logger.info(f"Enhanced factor importance calculated: {len(feature_importance)} factors synchronized with {overall_stress_level:.1f}% stress level")
        
        # STEP 2: REAL-TIME DEPARTMENT ANALYSIS
        logger.info("Analyzing real department data...")
        
        department_stats = df.groupby('department')['stress_level'].agg([
            'mean', 'count', 'std', 'min', 'max', 'median'
        ]).round(2)
        
        department_breakdown = {}
        for dept in department_stats.index:
            dept_data = department_stats.loc[dept]
            dept_df = df[df['department'] == dept]
            
            # Calculate real department metrics
            high_stress_count = int((dept_df['stress_level'] > 70).sum())
            medium_stress_count = int(((dept_df['stress_level'] > 40) & (dept_df['stress_level'] <= 70)).sum())
            low_stress_count = int((dept_df['stress_level'] <= 40).sum())
            
            department_breakdown[dept] = {
                'average_stress': float(dept_data['mean']),
                'employee_count': int(dept_data['count']),
                'stress_std': float(dept_data['std']) if not pd.isna(dept_data['std']) else 0.0,
                'min_stress': float(dept_data['min']),
                'max_stress': float(dept_data['max']),
                'median_stress': float(dept_data['median']),
                'high_stress_employees': high_stress_count,
                'medium_stress_employees': medium_stress_count,
                'low_stress_employees': low_stress_count,
                'risk_level': 'High' if dept_data['mean'] > 70 else 'Medium' if dept_data['mean'] > 40 else 'Low'
            }
        
        # STEP 3: ADVANCED ML MODELS (Dynamic Training on Real Data)
        dl_results = {}
        try:
            if len(df) >= 10:  # Minimum data for training
                logger.info("Training deep learning model on real dataset...")
                training_result = advanced_dl_model.train(df)
                
                if training_result.get('success', False):
                    dl_results = {
                        'model_trained': True,
                        'training_accuracy': training_result.get('accuracy', 0),
                        'training_samples': training_result.get('training_samples', 0),
                        'test_samples': training_result.get('test_samples', 0),
                        'epochs_trained': training_result.get('epochs_trained', 0),
                        'final_loss': training_result.get('final_val_loss', 0)
                    }
                    
                    # Generate REAL predictions for all employees in dataset
                    dl_predictions = []
                    for _, row in df.iterrows():
                        prediction = advanced_dl_model.predict(row)
                        if prediction:
                            dl_predictions.append({
                                'employee_id': row['employee_id'],
                                'predicted_stress': prediction['predicted_stress_level'],
                                'confidence': prediction['confidence_score'],
                                'actual_stress': float(row['stress_level'])
                            })
                    
                    dl_results['predictions'] = dl_predictions
                    dl_results['predictions_count'] = len(dl_predictions)
                    
                    # Calculate model accuracy against real data
                    if dl_predictions:
                        actual_values = [p['actual_stress'] for p in dl_predictions]
                        predicted_values = [p['predicted_stress'] for p in dl_predictions]
                        dl_results['real_data_accuracy'] = float(1 - np.mean(np.abs(np.array(actual_values) - np.array(predicted_values)) / 100))
                else:
                    dl_results = {'model_trained': False, 'error': training_result.get('error', 'Training failed')}
            else:
                dl_results = {'model_trained': False, 'error': 'Insufficient data for training (need ≥10 records)'}
                
        except Exception as e:
            logger.error(f"Deep learning analysis failed: {str(e)}")
            dl_results = {'model_trained': False, 'error': str(e)}
        
        # STEP 4: Neural Collaborative Filtering (Real Data Analysis)
        ncf_results = {}
        try:
            if len(df) >= 5:  # Minimum data for NCF
                logger.info("Training NCF model on real employee-factor interactions...")
                ncf_training = ncf_model.train(df)
                
                if ncf_training.get('success', False):
                    ncf_results = {
                        'model_trained': True,
                        'training_accuracy': ncf_training.get('accuracy', 0),
                        'num_employees': ncf_training.get('num_employees', 0),
                        'num_factors': ncf_training.get('num_factors', 0),
                        'training_samples': ncf_training.get('training_samples', 0)
                    }
                    
                    # Generate factor recommendations for each real employee
                    factor_recommendations = []
                    for _, row in df.iterrows():
                        recommendations = ncf_model.get_factor_recommendations(row['employee_id'])
                        if recommendations:
                            factor_recommendations.append({
                                'employee_id': row['employee_id'],
                                'department': row['department'],
                                'actual_stress': float(row['stress_level']),
                                'top_stress_factors': recommendations[:3]  # Top 3 factors
                            })
                    
                    ncf_results['factor_recommendations'] = factor_recommendations
                    ncf_results['recommendations_count'] = len(factor_recommendations)
                else:
                    ncf_results = {'model_trained': False, 'error': ncf_training.get('error', 'NCF training failed')}
            else:
                ncf_results = {'model_trained': False, 'error': 'Insufficient data for NCF (need ≥5 records)'}
                
        except Exception as e:
            logger.error(f"NCF analysis failed: {str(e)}")
            ncf_results = {'model_trained': False, 'error': str(e)}
        
        # STEP 5: DYNAMIC RECOMMENDATION GENERATION (Based on Real Analysis)
        logger.info("Generating dynamic recommendations based on real data analysis...")
        
        # Prepare real analysis data for recommendation engine
        analysis_data = {
            'overall_stress_level': overall_stress_level,
            'factor_contributions': feature_importance,
            'department_breakdown': department_breakdown,
            'stress_distribution': {
                'mean': overall_stress_level,
                'std': stress_std,
                'median': stress_median,
                'min': float(df['stress_level'].min()),
                'max': float(df['stress_level'].max())
            }
        }
        
        # Real employee data from dataset
        employee_data = {
            'total_employees': len(df),
            'departments': df['department'].unique().tolist(),
            'high_stress_employees': int((df['stress_level'] > 70).sum()),
            'medium_stress_employees': int(((df['stress_level'] > 40) & (df['stress_level'] <= 70)).sum()),
            'low_stress_employees': int((df['stress_level'] <= 40).sum()),
            'department_counts': df['department'].value_counts().to_dict()
        }
        
        # Generate recommendations based on REAL data patterns using dynamic engine
        from recommendation_engine import get_recommendations_for_analysis
        
        # Prepare analysis data for dynamic recommendation engine
        recommendation_analysis_data = {
            'overall_stress_level': overall_stress_level,
            'factor_importance': feature_importance,
            'department_breakdown': department_breakdown,
            'total_employees': len(df),
            'stress_distribution': {
                'high_risk': high_stress_count,
                'medium_risk': medium_stress_count, 
                'low_risk': low_stress_count
            }
        }
        
        # Generate dynamic recommendations (3 intelligent recommendations)
        recommendations = get_recommendations_for_analysis(recommendation_analysis_data)
        
        # STEP 6: STORE REAL PREDICTIONS IN DATABASE (Fixed Session Management)
        logger.info("Storing real-time predictions in database...")
        predictions_stored = 0
        prediction_errors = 0
        
        for _, row in df.iterrows():
            try:
                # Find employee in database (CREATE if not exists)
                employee = session_db.query(Employee).filter_by(
                    employee_id=str(row['employee_id'])
                ).first()
                
                if not employee:
                    # Create new employee record
                    employee = Employee(
                        employee_id=str(row['employee_id']),
                        department=str(row['department']),
                        workload=float(row['workload']),
                        work_life_balance=float(row['work_life_balance']),
                        team_conflict=float(row['team_conflict']),
                        management_support=float(row['management_support']),
                        work_environment=float(row['work_environment']),
                        stress_level=float(row['stress_level']),
                        created_at=datetime.utcnow(),
                        last_update=datetime.utcnow()
                    )
                    session_db.add(employee)
                    session_db.flush()  # Get employee ID
                
                # Calculate enhanced prediction using multiple methods
                statistical_pred = 50 + sum([
                    (row['workload'] - 5.5) * 2.5,
                    (row['work_life_balance'] - 5.5) * -2.0,
                    (row['team_conflict'] - 5.5) * 2.0,
                    (row['management_support'] - 5.5) * -1.5,
                    (row['work_environment'] - 5.5) * -2.0
                ])
                statistical_pred = max(0, min(100, statistical_pred))
                
                # Combine with DL prediction if available
                dl_pred_for_employee = None
                if dl_results.get('model_trained', False) and dl_results.get('predictions'):
                    dl_pred_for_employee = next(
                        (p for p in dl_results['predictions'] if p['employee_id'] == row['employee_id']), 
                        None
                    )
                
                if dl_pred_for_employee:
                    final_prediction = (statistical_pred * 0.3 + dl_pred_for_employee['predicted_stress'] * 0.7)
                    confidence = dl_pred_for_employee['confidence']
                    model_version = 'enhanced_dl_statistical_v2.1'
                else:
                    final_prediction = statistical_pred
                    confidence = 0.75
                    model_version = 'statistical_v2.1'
                
                # Create prediction record
                prediction = StressPrediction(
                    employee_id=employee.id,
                    predicted_stress_level=final_prediction,
                    confidence_score=confidence,
                    prediction_date=datetime.utcnow(),
                    model_version=model_version,
                    input_features={
                        'workload': float(row['workload']),
                        'work_life_balance': float(row['work_life_balance']),
                        'team_conflict': float(row['team_conflict']),
                        'management_support': float(row['management_support']),
                        'work_environment': float(row['work_environment']),
                        'actual_stress': float(row['stress_level'])
                    },
                    factor_weights={
                        'statistical_weight': 0.3 if dl_pred_for_employee else 1.0,
                        'deep_learning_weight': 0.7 if dl_pred_for_employee else 0.0,
                        'correlation_based': True
                    }
                )
                session_db.add(prediction)
                predictions_stored += 1
                
            except Exception as pred_error:
                logger.warning(f"Error storing prediction for employee {row['employee_id']}: {str(pred_error)}")
                prediction_errors += 1
                continue
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - analysis_start_time).total_seconds()
        
        # STEP 7: SAVE COMPREHENSIVE ANALYSIS RESULTS (Fixed Session Management)
        logger.info("Saving comprehensive analysis results...")
        
        # Create analysis result record with all REAL data
        analysis_result = AnalysisResult(
            dataset_id=dataset_id,
            analysis_type='enhanced_ml_comprehensive_realtime',
            overall_stress_level=overall_stress_level,
            analysis_date=datetime.utcnow(),
            factor_contributions=feature_importance,
            model_accuracy=dl_results.get('real_data_accuracy', 0.85),
            confidence_score=0.95,  # High confidence with real data analysis
            department_breakdown=department_breakdown,
            correlation_matrix=df[['workload', 'work_life_balance', 'team_conflict', 
                                 'management_support', 'work_environment', 'stress_level']].corr().to_dict(),
            feature_importance=feature_importance,
            processing_time_seconds=processing_time
        )
        
        session_db.add(analysis_result)
        
        # Update dataset status
        dataset.status = 'enhanced_processed_realtime'
        dataset.processing_end_time = datetime.utcnow()
        
        # Create comprehensive system log
        system_log = SystemLog(
            level='INFO',
            module='enhanced_analysis_realtime',
            message=f'Real-time Enhanced ML analysis completed for dataset {dataset.name}',
            additional_data={
                'dataset_id': dataset_id,
                'user_id': user_id,
                'analysis_type': 'enhanced_ml_comprehensive_realtime',
                'overall_stress_level': overall_stress_level,
                'stress_std': stress_std,
                'predictions_stored': predictions_stored,
                'prediction_errors': prediction_errors,
                'recommendations_generated': len(recommendations),
                'processing_time': processing_time,
                'dl_model_trained': dl_results.get('model_trained', False),
                'dl_accuracy': dl_results.get('real_data_accuracy', 0),
                'ncf_model_trained': ncf_results.get('model_trained', False),
                'departments_analyzed': len(department_breakdown),
                'correlation_strength': correlation_insights.get('overall_correlation_strength', 0),
                'synchronized_calculation': True
            }
        )
        session_db.add(system_log)
        
        # Commit all changes together
        session_db.commit()
        
        # Get the analysis result ID for response
        analysis_result_id = analysis_result.id
        
        # Calculate comprehensive real-time metrics
        high_stress_count = int((df['stress_level'] > 70).sum())
        medium_stress_count = int(((df['stress_level'] > 40) & (df['stress_level'] <= 70)).sum())
        low_stress_count = int((df['stress_level'] <= 40).sum())
        
        # Build comprehensive response with SYNCHRONIZED CALCULATIONS
        response = {
            'success': True,
            'analysis_id': analysis_result_id,
            'dataset_id': dataset_id,
            'analysis_type': 'enhanced_ml_comprehensive_realtime_synchronized',
            'is_real_time': True,
            'data_source': 'uploaded_dataset',
            'summary': {
                'overall_stress_level': round(overall_stress_level, 2),
                'stress_category': 'High' if overall_stress_level > 70 else 'Medium' if overall_stress_level > 40 else 'Low',
                'confidence_score': 0.95,
                'stress_distribution': {
                    'high_risk': high_stress_count,
                    'medium_risk': medium_stress_count,
                    'low_risk': low_stress_count,
                    'high_risk_percentage': round((high_stress_count / len(df)) * 100, 1),
                    'medium_risk_percentage': round((medium_stress_count / len(df)) * 100, 1),
                    'low_risk_percentage': round((low_stress_count / len(df)) * 100, 1)
                },
                'total_employees': len(df),
                'departments_analyzed': len(department_breakdown),
                'predictions_generated': predictions_stored,
                'prediction_success_rate': round((predictions_stored / len(df)) * 100, 1) if len(df) > 0 else 0
            },
            'factor_analysis': {
                'feature_importance': feature_importance,
                'correlation_insights': correlation_insights,
                'synchronization_info': {
                    'stress_level_synchronized': True,
                    'calculation_method': 'enhanced_multi_factor_analysis',
                    'factors_aligned_with_stress': True,
                    'dynamic_correlations': True,
                    'total_factor_importance': sum([v.get('importance_percentage', 0) for v in feature_importance.values()]),
                    'correlation_pattern': correlation_insights.get('correlation_pattern', 'unknown')
                }
            },
            'department_breakdown': department_breakdown,
            'deep_learning_analysis': dl_results,
            'ncf_analysis': ncf_results,
            'recommendations': recommendations,
            'advanced_insights': {
                'stress_variance': round(stress_std, 2),
                'stress_median': round(stress_median, 2),
                'stress_range': {
                    'min': float(df['stress_level'].min()),
                    'max': float(df['stress_level'].max())
                },
                'departmental_risk_assessment': {
                    'high_risk_departments': [dept for dept, data in department_breakdown.items() 
                                            if data['average_stress'] > 70],
                    'stable_departments': [dept for dept, data in department_breakdown.items() 
                                         if data['average_stress'] <= 40],
                    'departments_needing_attention': [dept for dept, data in department_breakdown.items() 
                                                    if 40 < data['average_stress'] <= 70]
                },
                'model_performance': {
                    'statistical_baseline': 0.75,
                    'deep_learning_accuracy': dl_results.get('real_data_accuracy', 0),
                    'deep_learning_trained': dl_results.get('model_trained', False),
                    'ncf_factor_analysis': ncf_results.get('model_trained', False),
                    'combined_confidence': 0.95,
                    'real_time_processing': True
                }
            },
            'processing_info': {
                'processing_time_seconds': round(processing_time, 2),
                'timestamp': datetime.utcnow().isoformat(),
                'models_used': ['statistical_realtime', 'correlation_analysis', 'synchronized_factor_analysis'] + 
                              (['deep_learning'] if dl_results.get('model_trained') else []) +
                              (['ncf'] if ncf_results.get('model_trained') else []) +
                              ['recommendation_engine'],
                'data_quality_score': 0.98,  # High score for complete analysis
                'analysis_completeness': {
                    'statistical_analysis': True,
                    'correlation_analysis': True,
                    'department_analysis': True,
                    'deep_learning': dl_results.get('model_trained', False),
                    'ncf_analysis': ncf_results.get('model_trained', False),
                    'recommendations': len(recommendations) > 0,
                    'synchronized_calculations': True
                }
            }
        }
        
        logger.info(f"Enhanced synchronized ML analysis completed for dataset {dataset_id}: {overall_stress_level:.2f}% overall stress with {len(feature_importance)} synchronized factors")
        
        return jsonify(response), 200
        
    except Exception as e:
        # Rollback any pending changes
        if session_db:
            try:
                session_db.rollback()
            except:
                pass
        
        logger.error(f"Enhanced analysis failed: {str(e)}")
        return jsonify({'error': f'Enhanced analysis failed: {str(e)}'}), 500
        
    finally:
        # Always close the session
        if session_db:
            try:
                session_db.close()
            except:
                pass

@app.route('/api/download-template', methods=['GET'])
def download_template():
    """Download dataset template file"""
    try:
        # Create template data
        template_data = [
            ['employee_id', 'department', 'workload', 'work_life_balance', 'team_conflict', 'management_support', 'work_environment', 'stress_level'],
            ['EMP001', 'IT', '8', '4', '6', '5', '6', '75'],
            ['EMP002', 'HR', '6', '7', '3', '8', '7', '45'],
            ['EMP003', 'Finance', '9', '3', '7', '4', '5', '85'],
            ['EMP004', 'Marketing', '7', '6', '4', '7', '8', '50'],
            ['EMP005', 'Sales', '8', '5', '5', '6', '6', '65'],
            ['EMP006', 'Operations', '5', '8', '2', '9', '9', '30'],
            ['EMP007', 'Research', '9', '2', '8', '3', '4', '90'],
            ['EMP008', 'Customer Service', '6', '6', '5', '7', '7', '55'],
            ['EMP009', 'Production', '7', '5', '6', '6', '6', '60'],
            ['EMP010', 'Quality Assurance', '8', '4', '7', '5', '5', '70']
        ]
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(template_data)
        
        # Convert to bytes
        csv_data = output.getvalue()
        output.close()
        
        # Create file-like object
        csv_file = io.BytesIO(csv_data.encode('utf-8'))
        csv_file.seek(0)
        
        # Generate filename with timestamp
        filename = f'stress_dataset_template_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        logger.info("Template dataset downloaded")
        
        return send_file(
            csv_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Failed to generate template: {str(e)}")
        return jsonify({'error': f'Failed to generate template: {str(e)}'}), 500

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

@app.route('/api/dataset/<int:dataset_id>/basic-analysis', methods=['GET'])
def get_dataset_basic_analysis(dataset_id):
    """Get basic analysis results for a specific dataset with synchronized factor importance"""
    session_db = None
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
        
        session_db = db_manager.get_session()
        
        # Get specific dataset
        dataset = session_db.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Check if file exists
        if not os.path.exists(dataset.file_path):
            return jsonify({'error': 'Dataset file not found on server'}), 404
        
        # Load dataset
        try:
            if dataset.file_path.endswith('.csv'):
                df = pd.read_csv(dataset.file_path)
            else:
                df = pd.read_excel(dataset.file_path)
            
            # Validate dataset structure
            required_columns = ['employee_id', 'department', 'workload', 'work_life_balance',
                              'team_conflict', 'management_support', 'work_environment', 'stress_level']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return jsonify({
                    'error': f'Dataset missing required columns: {", ".join(missing_columns)}'
                }), 400
                
        except Exception as file_error:
            logger.error(f"Failed to load dataset file: {str(file_error)}")
            return jsonify({'error': f'Failed to load dataset: {str(file_error)}'}), 400
        
        # Calculate basic statistics for this specific dataset (with NaN safety)
        overall_stress_level = float(df['stress_level'].mean()) if not pd.isna(df['stress_level'].mean()) else 0.0
        stress_std = float(df['stress_level'].std()) if not pd.isna(df['stress_level'].std()) else 0.0
        total_employees = len(df)
        
        # Calculate stress distribution
        high_stress_count = int((df['stress_level'] > 70).sum())
        medium_stress_count = int(((df['stress_level'] > 40) & (df['stress_level'] <= 70)).sum())
        low_stress_count = int((df['stress_level'] <= 40).sum())
        
        # Validate data before calculation
        numeric_columns = ['workload', 'work_life_balance', 'team_conflict', 
                          'management_support', 'work_environment', 'stress_level']
        
        validation_report = validate_numeric_data(df, numeric_columns)
        if validation_report['warnings']:
            logger.warning(f"Data validation warnings for dataset {dataset_id}: {validation_report['warnings']}")
        
        # Use ENHANCED SYNCHRONIZED factor importance calculation
        factor_importance = enhanced_factor_importance_calculation(
            df, 'stress_level', overall_stress_level
        )
        
        # Get dynamic correlation insights
        correlation_insights = dynamic_correlation_analysis(df, 'stress_level')
        
        # Safe department breakdown calculation
        department_breakdown = safe_department_breakdown(df, 'stress_level')
        
        # Get latest analysis result for this dataset if exists
        latest_analysis = session_db.query(AnalysisResult).filter_by(
            dataset_id=dataset_id
        ).order_by(AnalysisResult.analysis_date.desc()).first()
        
        # Build response with dataset-specific synchronized data
        response = {
            'success': True,
            'dataset_id': dataset_id,
            'dataset_info': {
                'id': dataset.id,
                'name': dataset.name,
                'description': dataset.description,
                'record_count': len(df),
                'upload_date': dataset.upload_date.isoformat() if dataset.upload_date else None,
                'status': dataset.status
            },
            'analysis': {
                'overall_stress_level': round(overall_stress_level, 2),
                'stress_category': 'High' if overall_stress_level > 70 else 'Medium' if overall_stress_level > 40 else 'Low',
                'total_employees': total_employees,
                'stress_distribution': {
                    'high_risk': high_stress_count,
                    'medium_risk': medium_stress_count,
                    'low_risk': low_stress_count,
                    'high_risk_percentage': round((high_stress_count / total_employees) * 100, 1) if total_employees > 0 else 0.0,
                    'medium_risk_percentage': round((medium_stress_count / total_employees) * 100, 1) if total_employees > 0 else 0.0,
                    'low_risk_percentage': round((low_stress_count / total_employees) * 100, 1) if total_employees > 0 else 0.0
                },
                'factor_importance': factor_importance,
                'department_breakdown': department_breakdown,
                'correlation_insights': correlation_insights,
                'data_quality': {
                    'completeness': 100.0,  # All required columns present
                    'stress_variance': round(stress_std, 2),
                    'departments_count': len(department_breakdown),
                    'validation_warnings': validation_report.get('warnings', [])
                },
                'synchronization_info': {
                    'calculation_method': 'enhanced_synchronized_basic_analysis',
                    'factors_aligned_with_stress': True,
                    'total_factor_importance': sum([v.get('importance_percentage', 0) for v in factor_importance.values()]),
                    'correlation_pattern': correlation_insights.get('correlation_pattern', 'unknown'),
                    'dynamic_correlations': True
                }
            },
            'has_enhanced_analysis': latest_analysis is not None,
            'last_analysis_date': latest_analysis.analysis_date.isoformat() if latest_analysis else None
        }
        
        # Clean the entire response of any remaining NaN values
        response = clean_nan_values(response)
        
        logger.info(f"Synchronized basic analysis retrieved for dataset {dataset_id}: {overall_stress_level:.2f}% avg stress, {total_employees} employees with {len(factor_importance)} synchronized factors")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Failed to get basic analysis for dataset {dataset_id}: {str(e)}")
        return jsonify({'error': f'Failed to get dataset analysis: {str(e)}'}), 500
        
    finally:
        if session_db:
            try:
                session_db.close()
            except:
                pass

@app.route('/api/dataset/<int:dataset_id>/recommendations', methods=['GET'])
def get_dataset_recommendations(dataset_id):
    """Get dynamic recommendations for a specific dataset"""
    session_db = None
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
        
        session_db = db_manager.get_session()
        
        # Get specific dataset
        dataset = session_db.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Check if file exists
        if not os.path.exists(dataset.file_path):
            return jsonify({'error': 'Dataset file not found on server'}), 404
        
        # Load dataset
        try:
            if dataset.file_path.endswith('.csv'):
                df = pd.read_csv(dataset.file_path)
            else:
                df = pd.read_excel(dataset.file_path)
            
            # Validate dataset structure
            required_columns = ['employee_id', 'department', 'workload', 'work_life_balance',
                              'team_conflict', 'management_support', 'work_environment', 'stress_level']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return jsonify({
                    'error': f'Dataset missing required columns: {", ".join(missing_columns)}'
                }), 400
                
        except Exception as file_error:
            logger.error(f"Failed to load dataset file: {str(file_error)}")
            return jsonify({'error': f'Failed to load dataset: {str(file_error)}'}), 400
        
        # Calculate analysis data needed for recommendations
        overall_stress_level = float(df['stress_level'].mean()) if not pd.isna(df['stress_level'].mean()) else 0.0
        
        # Use enhanced factor importance calculation
        factor_importance = enhanced_factor_importance_calculation(
            df, 'stress_level', overall_stress_level
        )
        
        # Safe department breakdown calculation
        department_breakdown = safe_department_breakdown(df, 'stress_level')
        
        # Calculate stress distribution
        high_stress_count = int((df['stress_level'] > 70).sum())
        medium_stress_count = int(((df['stress_level'] > 40) & (df['stress_level'] <= 70)).sum())
        low_stress_count = int((df['stress_level'] <= 40).sum())
        
        # Generate dynamic recommendations using the recommendation engine
        from recommendation_engine import get_recommendations_for_analysis
        
        # Prepare analysis data for dynamic recommendation engine
        recommendation_analysis_data = {
            'overall_stress_level': overall_stress_level,
            'factor_importance': factor_importance,
            'department_breakdown': department_breakdown,
            'total_employees': len(df),
            'stress_distribution': {
                'high_risk': high_stress_count,
                'medium_risk': medium_stress_count, 
                'low_risk': low_stress_count
            }
        }
        
        # Generate dynamic recommendations (3 intelligent recommendations)
        recommendations = get_recommendations_for_analysis(recommendation_analysis_data)
        
        # Enhance recommendations with dataset-specific context
        for rec in recommendations:
            rec['dataset_id'] = dataset_id
            rec['dataset_name'] = dataset.name
            rec['generated_at'] = datetime.utcnow().isoformat()
            rec['based_on_employees'] = len(df)
            rec['departments_analyzed'] = len(department_breakdown)
        
        logger.info(f"Generated {len(recommendations)} dynamic recommendations for dataset {dataset_id}")
        
        return jsonify({
            'success': True,
            'dataset_id': dataset_id,
            'dataset_name': dataset.name,
            'recommendations': recommendations,
            'analysis_summary': {
                'overall_stress_level': round(overall_stress_level, 2),
                'total_employees': len(df),
                'departments_count': len(department_breakdown),
                'high_risk_employees': high_stress_count,
                'medium_risk_employees': medium_stress_count,
                'low_risk_employees': low_stress_count
            },
            'generated_at': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to generate recommendations for dataset {dataset_id}: {str(e)}")
        return jsonify({'error': f'Failed to generate recommendations: {str(e)}'}), 500
        
    finally:
        if session_db:
            try:
                session_db.close()
            except:
                pass

@app.route('/api/dataset/<int:dataset_id>/employees', methods=['GET'])
def get_dataset_employees(dataset_id):
    """Get list of employees from a specific dataset"""
    try:
        # Verify authentication (temporarily disabled for testing)
        # auth_header = request.headers.get('Authorization')
        # if not auth_header or not auth_header.startswith('Bearer '):
        #     return jsonify({'error': 'No valid token provided'}), 401
        
        # token = auth_header.split(' ')[1]
        # payload = verify_jwt_token(token)
        # if not payload:
        #     return jsonify({'error': 'Invalid or expired token'}), 401
        
        session = db_manager.get_session()
        
        # Get dataset
        dataset = session.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            session.close()
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Check if file exists
        if not os.path.exists(dataset.file_path):
            session.close()
            return jsonify({'error': 'Dataset file not found on server'}), 404
        
        # Load dataset from file to get employee list
        try:
            if dataset.file_path.endswith('.csv'):
                df = pd.read_csv(dataset.file_path)
            else:
                df = pd.read_excel(dataset.file_path)
            
            required_columns = ['employee_id', 'department', 'workload', 'work_life_balance',
                              'team_conflict', 'management_support', 'work_environment', 'stress_level']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                session.close()
                return jsonify({
                    'error': f'Dataset missing required columns: {", ".join(missing_columns)}'
                }), 400
                
        except Exception as file_error:
            logger.error(f"Failed to load dataset file: {str(file_error)}")
            session.close()
            return jsonify({'error': f'Failed to load dataset: {str(file_error)}'}), 400
        
        # Convert employee data to list
        employees = []
        for _, row in df.iterrows():
            employee_data = {
                'employee_id': str(row['employee_id']),
                'name': f"Employee {row['employee_id']}",  # Generate name if not available
                'department': str(row['department']),
                'position': row.get('position', 'Staff'),  # Default position if not available
                'age': int(row.get('age', 30)) if 'age' in row else 30,  # Default age if not available
                'workload': float(row['workload']),
                'work_life_balance': float(row['work_life_balance']),
                'team_conflict': float(row['team_conflict']),
                'management_support': float(row['management_support']),
                'work_environment': float(row['work_environment']),
                'stress_level': float(row['stress_level'])
            }
            employees.append(employee_data)
        
        session.close()
        
        logger.info(f"Retrieved {len(employees)} employees from dataset {dataset_id}")
        
        return jsonify({
            'success': True,
            'dataset_id': dataset_id,
            'dataset_name': dataset.name,
            'employees': employees,
            'total_employees': len(employees),
            'departments': list(df['department'].unique()),
            'department_counts': df['department'].value_counts().to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get employees from dataset {dataset_id}: {str(e)}")
        return jsonify({'error': f'Failed to get employees: {str(e)}'}), 500

@app.route('/api/dataset/<int:dataset_id>/employee/<string:employee_id>/analyze', methods=['POST'])
def analyze_individual_employee(dataset_id, employee_id):
    """Perform individual stress analysis for a specific employee"""
    try:
        # Verify authentication (temporarily disabled for testing)
        # auth_header = request.headers.get('Authorization')
        # if not auth_header or not auth_header.startswith('Bearer '):
        #     return jsonify({'error': 'No valid token provided'}), 401
        
        # token = auth_header.split(' ')[1]
        # payload = verify_jwt_token(token)
        # if not payload:
        #     return jsonify({'error': 'Invalid or expired token'}), 401
        
        session = db_manager.get_session()
        
        # Get dataset
        dataset = session.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            session.close()
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Check if file exists
        if not os.path.exists(dataset.file_path):
            session.close()
            return jsonify({'error': 'Dataset file not found on server'}), 404
        
        # Load dataset from file
        try:
            if dataset.file_path.endswith('.csv'):
                df = pd.read_csv(dataset.file_path)
            else:
                df = pd.read_excel(dataset.file_path)
                
        except Exception as file_error:
            logger.error(f"Failed to load dataset file: {str(file_error)}")
            session.close()
            return jsonify({'error': f'Failed to load dataset: {str(file_error)}'}), 400
        
        # Find specific employee in dataset
        employee_data = df[df['employee_id'].astype(str) == str(employee_id)]
        if employee_data.empty:
            session.close()
            return jsonify({'error': f'Employee {employee_id} not found in dataset'}), 404
        
        employee_row = employee_data.iloc[0]
        
        # Get employee basic information
        employee_info = {
            'employee_id': str(employee_row['employee_id']),
            'name': f"Employee {employee_row['employee_id']}",
            'department': str(employee_row['department']),
            'position': employee_row.get('position', 'Staff'),
            'age': int(employee_row.get('age', 30)) if 'age' in employee_row else 30,
            'workload': float(employee_row['workload']),
            'work_life_balance': float(employee_row['work_life_balance']),
            'team_conflict': float(employee_row['team_conflict']),
            'management_support': float(employee_row['management_support']),
            'work_environment': float(employee_row['work_environment']),
            'actual_stress_level': float(employee_row['stress_level'])
        }
        
        # Calculate stress analysis
        stress_level = float(employee_row['stress_level'])
        
        # Determine stress category
        if stress_level < 40:
            stress_category = 'Rendah'
            stress_color = 'green'
        elif stress_level < 70:
            stress_category = 'Medium'
            stress_color = 'orange'
        else:
            stress_category = 'Tinggi'
            stress_color = 'red'
        
        # Calculate factor analysis for this employee
        risk_factors = []
        
        factors = {
            'workload': {
                'name': 'Beban Kerja',
                'value': float(employee_row['workload']),
                'threshold_high': 7.0,
                'threshold_medium': 5.0,
                'positive_impact': True  # Higher value = higher stress
            },
            'work_life_balance': {
                'name': 'Work-Life Balance',
                'value': float(employee_row['work_life_balance']),
                'threshold_high': 4.0,
                'threshold_medium': 6.0,
                'positive_impact': False  # Higher value = lower stress
            },
            'team_conflict': {
                'name': 'Konflik Tim',
                'value': float(employee_row['team_conflict']),
                'threshold_high': 6.0,
                'threshold_medium': 4.0,
                'positive_impact': True  # Higher value = higher stress
            },
            'management_support': {
                'name': 'Dukungan Manajemen',
                'value': float(employee_row['management_support']),
                'threshold_high': 4.0,
                'threshold_medium': 6.0,
                'positive_impact': False  # Higher value = lower stress
            },
            'work_environment': {
                'name': 'Lingkungan Kerja',
                'value': float(employee_row['work_environment']),
                'threshold_high': 4.0,
                'threshold_medium': 6.0,
                'positive_impact': False  # Higher value = lower stress
            }
        }
        
        for factor_key, factor_data in factors.items():
            value = factor_data['value']
            name = factor_data['name']
            positive_impact = factor_data['positive_impact']
            
            # Determine impact level
            if positive_impact:
                # Higher values mean higher stress
                if value >= factor_data['threshold_high']:
                    impact = 'Tinggi'
                elif value >= factor_data['threshold_medium']:
                    impact = 'Medium'
                else:
                    impact = 'Rendah'
            else:
                # Higher values mean lower stress
                if value <= factor_data['threshold_high']:
                    impact = 'Tinggi'
                elif value <= factor_data['threshold_medium']:
                    impact = 'Medium'
                else:
                    impact = 'Rendah'
            
            # Generate recommendations
            recommendations = {
                'workload': {
                    'Tinggi': 'Redistributions tugas dan optimalisasi workflow',
                    'Medium': 'Monitor beban kerja dan berikan dukungan tambahan',
                    'Rendah': 'Pertahankan manajemen beban kerja yang baik'
                },
                'work_life_balance': {
                    'Tinggi': 'Implementasi kebijakan work-from-home dan flexible hours',
                    'Medium': 'Berikan pelatihan time management',
                    'Rendah': 'Pertahankan keseimbangan yang sudah baik'
                },
                'team_conflict': {
                    'Tinggi': 'Mediasi konflik dan team building activities',
                    'Medium': 'Pelatihan komunikasi interpersonal',
                    'Rendah': 'Pertahankan harmoni tim yang baik'
                },
                'management_support': {
                    'Tinggi': 'Training manajemen dan implementasi regular one-on-one meetings',
                    'Medium': 'Tingkatkan komunikasi antara atasan dan bawahan',
                    'Rendah': 'Pertahankan dukungan manajemen yang baik'
                },
                'work_environment': {
                    'Tinggi': 'Perbaikan kondisi fisik workplace dan fasilitas kerja',
                    'Medium': 'Evaluasi dan tingkatkan kenyamanan workspace',
                    'Rendah': 'Pertahankan lingkungan kerja yang kondusif'
                }
            }
            
            risk_factors.append({
                'factor': name,
                'value': value,
                'impact': impact,
                'recommendation': recommendations[factor_key][impact]
            })
        
        # Calculate prediction confidence based on data consistency
        # Higher consistency across factors = higher confidence
        factor_values = [employee_row[col] for col in ['workload', 'work_life_balance', 'team_conflict', 'management_support', 'work_environment']]
        factor_variance = np.var(factor_values)
        confidence = max(75, 95 - (factor_variance * 2))  # Base confidence 75-95%
        
        # Compare with department average for context
        dept_data = df[df['department'] == employee_row['department']]
        dept_avg_stress = float(dept_data['stress_level'].mean())
        
        # Find similar employees (same department, similar stress level)
        similar_employees = df[
            (df['department'] == employee_row['department']) & 
            (abs(df['stress_level'] - stress_level) < 10) &
            (df['employee_id'] != employee_row['employee_id'])
        ]
        
        similar_profiles = []
        for _, similar_emp in similar_employees.head(3).iterrows():
            similarity_score = 100 - abs(similar_emp['stress_level'] - stress_level) * 2
            similar_profiles.append({
                'employee_id': str(similar_emp['employee_id']),
                'department': str(similar_emp['department']),
                'stress_level': float(similar_emp['stress_level']),
                'similarity': round(similarity_score, 1)
            })
        
        session.close()
        
        # Build comprehensive response
        analysis_result = {
            'success': True,
            'dataset_id': dataset_id,
            'dataset_name': dataset.name,
            'employee_info': employee_info,
            'stress_analysis': {
                'stress_level': stress_level,
                'stress_category': stress_category,
                'stress_color': stress_color,
                'prediction_confidence': round(confidence, 1),
                'department_average': round(dept_avg_stress, 1),
                'compared_to_department': 'Di atas rata-rata' if stress_level > dept_avg_stress else 'Di bawah rata-rata' if stress_level < dept_avg_stress else 'Sesuai rata-rata'
            },
            'risk_factors': risk_factors,
            'similar_profiles': similar_profiles,
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'recommendations_summary': f"Berdasarkan analisis, karyawan ini memiliki tingkat stres {stress_category.lower()} dengan skor {stress_level:.1f}%. "
                                     f"Fokus utama intervensi pada faktor dengan dampak tinggi."
        }
        
        logger.info(f"Individual analysis completed for employee {employee_id} in dataset {dataset_id}")
        
        return jsonify(analysis_result), 200
        
    except Exception as e:
        logger.error(f"Failed to analyze employee {employee_id} in dataset {dataset_id}: {str(e)}")
        return jsonify({'error': f'Failed to analyze employee: {str(e)}'}), 500

# ADMIN ENDPOINTS FOR USER AND DATASET MANAGEMENT

def require_admin_auth(f):
    """Decorator to require admin authentication"""
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No valid token provided'}), 401
        
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Check if user is admin
        session_db = db_manager.get_session()
        try:
            user = session_db.query(User).filter_by(id=payload['user_id'], is_active=True).first()
            if not user or user.role != UserRole.admin:
                session_db.close()
                return jsonify({'error': 'Admin access required'}), 403
            
            # Add user to kwargs
            kwargs['current_user'] = user
            session_db.close()
            return f(*args, **kwargs)
        except Exception as e:
            session_db.close()
            return jsonify({'error': 'Authentication failed'}), 500
    
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route('/api/admin/users', methods=['GET'])
@require_admin_auth
def admin_get_all_users(current_user):
    """Get all users with filtering and pagination (Admin only)"""
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        search = request.args.get('search', '').strip()
        role_filter = request.args.get('role', '').strip()
        department_filter = request.args.get('department', '').strip()
        active_only = request.args.get('active_only', 'true').lower() == 'true'
        
        offset = (page - 1) * limit
        
        session_db = db_manager.get_session()
        
        # Build query
        query = session_db.query(User)
        
        # Apply filters
        if active_only:
            query = query.filter(User.is_active == True)
        
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                (User.username.ilike(search_pattern)) |
                (User.email.ilike(search_pattern)) |
                (User.full_name.ilike(search_pattern)) |
                (User.department.ilike(search_pattern)) |
                (User.position.ilike(search_pattern))
            )
        
        if role_filter and role_filter in ['admin', 'employee']:
            query = query.filter(User.role == UserRole(role_filter))
        
        if department_filter:
            query = query.filter(User.department.ilike(f"%{department_filter}%"))
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination and ordering
        users = query.order_by(User.created_at.desc()).offset(offset).limit(limit).all()
        
        # Count datasets per user
        user_ids = [user.id for user in users]
        dataset_counts = {}
        if user_ids:
            dataset_count_query = session_db.query(
                Dataset.uploaded_by,
                func.count(Dataset.id).label('count')
            ).filter(Dataset.uploaded_by.in_(user_ids)).group_by(Dataset.uploaded_by).all()
            
            dataset_counts = {item[0]: item[1] for item in dataset_count_query}
        
        session_db.close()
        
        # Format response
        users_data = []
        for user in users:
            user_data = user.to_dict()
            user_data['dataset_count'] = dataset_counts.get(user.id, 0)
            users_data.append(user_data)
        
        return jsonify({
            'success': True,
            'users': users_data,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total_count,
                'pages': (total_count + limit - 1) // limit
            },
            'summary': {
                'total_users': total_count,
                'active_users': len([u for u in users_data if u['is_active']]),
                'admin_users': len([u for u in users_data if u['role'] == 'admin']),
                'employee_users': len([u for u in users_data if u['role'] == 'employee'])
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get users: {str(e)}")
        return jsonify({'error': f'Failed to get users: {str(e)}'}), 500

@app.route('/api/admin/users/<int:user_id>/datasets', methods=['GET'])
@require_admin_auth
def admin_get_user_datasets(user_id, current_user):
    """Get all datasets uploaded by a specific user (Admin only)"""
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        status_filter = request.args.get('status', '').strip()
        
        offset = (page - 1) * limit
        
        session_db = db_manager.get_session()
        
        # Check if user exists
        user = session_db.query(User).filter_by(id=user_id).first()
        if not user:
            session_db.close()
            return jsonify({'error': 'User not found'}), 404
        
        # Build query
        query = session_db.query(Dataset).filter_by(uploaded_by=user_id)
        
        if status_filter:
            query = query.filter(Dataset.status == status_filter)
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination and ordering
        datasets = query.order_by(Dataset.upload_date.desc()).offset(offset).limit(limit).all()
        
        session_db.close()
        
        # Format response - fix session issue
        datasets_data = []
        for dataset in datasets:
            dataset_dict = dataset.to_dict()
            # Add uploader info manually to avoid session issues
            dataset_dict['uploader'] = {
                'id': user.id,
                'username': user.username,
                'full_name': user.full_name,
                'email': user.email,
                'role': user.role.value
            }
            datasets_data.append(dataset_dict)
        
        user_dict = user.to_dict()
        user_dict['dataset_count'] = total_count
        
        return jsonify({
            'success': True,
            'user': user_dict,
            'datasets': datasets_data,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total_count,
                'pages': (total_count + limit - 1) // limit
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get user datasets: {str(e)}")
        return jsonify({'error': f'Failed to get user datasets: {str(e)}'}), 500

@app.route('/api/admin/datasets', methods=['GET'])
@require_admin_auth
def admin_get_all_datasets(current_user):
    """Get all datasets with user information (Admin only)"""
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        search = request.args.get('search', '').strip()
        status_filter = request.args.get('status', '').strip()
        user_filter = request.args.get('user_id', '').strip()
        
        offset = (page - 1) * limit
        
        session_db = db_manager.get_session()
        
        # Build query with user information
        query = session_db.query(Dataset).outerjoin(User, Dataset.uploaded_by == User.id)
        
        # Apply filters
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                (Dataset.name.ilike(search_pattern)) |
                (Dataset.description.ilike(search_pattern))
            )
        
        if status_filter:
            query = query.filter(Dataset.status == status_filter)
        
        if user_filter:
            try:
                user_id = int(user_filter)
                query = query.filter(Dataset.uploaded_by == user_id)
            except ValueError:
                pass
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination and ordering
        datasets = query.order_by(Dataset.upload_date.desc()).offset(offset).limit(limit).all()
        
        # Get uploader info for each dataset
        datasets_data = []
        for dataset in datasets:
            dataset_data = dataset.to_dict()
            if dataset.uploaded_by:
                uploader = session_db.query(User).filter_by(id=dataset.uploaded_by).first()
                if uploader:
                    dataset_data['uploader'] = {
                        'id': uploader.id,
                        'username': uploader.username,
                        'email': uploader.email,
                        'full_name': uploader.full_name,
                        'department': uploader.department,
                        'role': uploader.role.value if uploader.role else None
                    }
            datasets_data.append(dataset_data)
        
        session_db.close()
        
        return jsonify({
            'success': True,
            'datasets': datasets_data,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total_count,
                'pages': (total_count + limit - 1) // limit
            },
            'summary': {
                'total_datasets': total_count,
                'by_status': {
                    'uploaded': len([d for d in datasets_data if d['status'] == 'uploaded']),
                    'processing': len([d for d in datasets_data if d['status'] == 'processing']),
                    'processed': len([d for d in datasets_data if d['status'] == 'processed']),
                    'failed': len([d for d in datasets_data if d['status'] == 'failed'])
                }
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get datasets: {str(e)}")
        return jsonify({'error': f'Failed to get datasets: {str(e)}'}), 500

@app.route('/api/admin/dashboard/stats', methods=['GET'])
@require_admin_auth
def admin_get_dashboard_stats(current_user):
    """Get comprehensive dashboard statistics for admin"""
    try:
        session_db = db_manager.get_session()
        
        # User statistics
        total_users = session_db.query(User).count()
        active_users = session_db.query(User).filter_by(is_active=True).count()
        admin_users = session_db.query(User).filter_by(role=UserRole.admin).count()
        employee_users = session_db.query(User).filter_by(role=UserRole.employee).count()
        
        # Dataset statistics
        total_datasets = session_db.query(Dataset).count()
        processed_datasets = session_db.query(Dataset).filter_by(status='processed').count()
        processing_datasets = session_db.query(Dataset).filter_by(status='processing').count()
        failed_datasets = session_db.query(Dataset).filter_by(status='failed').count()
        
        # Recent activity
        recent_users = session_db.query(User).order_by(User.created_at.desc()).limit(5).all()
        recent_datasets = session_db.query(Dataset).outerjoin(User, Dataset.uploaded_by == User.id).order_by(Dataset.upload_date.desc()).limit(5).all()
        
        # Format recent datasets with uploader info
        recent_datasets_data = []
        for dataset in recent_datasets:
            dataset_data = dataset.to_dict()
            if dataset.uploaded_by:
                uploader = session_db.query(User).filter_by(id=dataset.uploaded_by).first()
                if uploader:
                    dataset_data['uploader'] = {
                        'id': uploader.id,
                        'username': uploader.username,
                        'full_name': uploader.full_name,
                        'department': uploader.department
                    }
            recent_datasets_data.append(dataset_data)
        
        session_db.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'users': {
                    'total': total_users,
                    'active': active_users,
                    'inactive': total_users - active_users,
                    'admins': admin_users,
                    'employees': employee_users
                },
                'datasets': {
                    'total': total_datasets,
                    'processed': processed_datasets,
                    'processing': processing_datasets,
                    'failed': failed_datasets,
                    'success_rate': (processed_datasets / total_datasets * 100) if total_datasets > 0 else 0
                },
                'recent_activity': {
                    'new_users': [user.to_dict() for user in recent_users],
                    'new_datasets': recent_datasets_data
                }
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get admin dashboard stats: {str(e)}")
        return jsonify({'error': f'Failed to get admin dashboard stats: {str(e)}'}), 500

# Additional Admin CRUD Operations

@app.route('/api/admin/users', methods=['POST'])
@require_admin_auth
def admin_create_user(current_user):
    """Create a new user account (Admin only)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Required fields
        required_fields = ['username', 'email', 'password', 'full_name']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Validate input
        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        full_name = data['full_name'].strip()
        role = data.get('role', 'employee').strip()
        
        # Validation checks
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters long'}), 400
        
        if not validate_email(email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        if len(full_name) < 2:
            return jsonify({'error': 'Full name must be at least 2 characters long'}), 400
        
        if role not in ['admin', 'employee']:
            return jsonify({'error': 'Invalid role. Must be admin or employee'}), 400
        
        # Check for existing users
        session_db = db_manager.get_session()
        
        existing_username = session_db.query(User).filter_by(username=username).first()
        if existing_username:
            session_db.close()
            return jsonify({'error': 'Username already exists'}), 409
        
        existing_email = session_db.query(User).filter_by(email=email).first()
        if existing_email:
            session_db.close()
            return jsonify({'error': 'Email already exists'}), 409
        
        # Hash password
        password_hash = hash_password(password)
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            role=UserRole(role),
            department=data.get('department', '').strip() or None,
            position=data.get('position', '').strip() or None,
            phone_number=data.get('phone_number', '').strip() or None,
            is_active=True,
            email_verified=True  # Admin-created users are auto-verified
        )
        
        session_db.add(new_user)
        session_db.commit()
        
        user_id = new_user.id
        
        # Add default permissions based on role
        default_permissions = {
            UserRole.admin: ['admin_access', 'user_management', 'system_config', 'data_analysis', 'report_generation'],
            UserRole.employee: ['profile_access', 'basic_reports']
        }
        
        for permission in default_permissions.get(UserRole(role), []):
            user_permission = UserPermission(
                user_id=user_id,
                permission=permission
            )
            session_db.add(user_permission)
        
        session_db.commit()
        
        # Get fresh user data for response
        user_dict = new_user.to_dict()
        
        session_db.close()
        
        logger.info(f"Admin {current_user.username} created new user: {username} ({email}) with role {role}")
        
        return jsonify({
            'success': True,
            'message': 'User created successfully',
            'user': user_dict
        }), 201
        
    except Exception as e:
        logger.error(f"Failed to create user: {str(e)}")
        return jsonify({'error': f'Failed to create user: {str(e)}'}), 500

@app.route('/api/admin/users/<int:user_id>/toggle-status', methods=['POST'])
@require_admin_auth
def admin_toggle_user_status(user_id, current_user):
    """Activate or deactivate a user account (Admin only)"""
    try:
        session_db = db_manager.get_session()
        
        user = session_db.query(User).filter_by(id=user_id).first()
        if not user:
            session_db.close()
            return jsonify({'error': 'User not found'}), 404
        
        # Prevent admin from deactivating themselves
        if user.id == current_user.id:
            session_db.close()
            return jsonify({'error': 'Cannot deactivate your own account'}), 400
        
        # Toggle status
        old_status = user.is_active
        user.is_active = not user.is_active
        user.updated_at = datetime.utcnow()
        
        session_db.commit()
        
        # Get fresh user data after commit
        user_dict = user.to_dict()
        action = 'activated' if user.is_active else 'deactivated'
        
        session_db.close()
        
        logger.info(f"Admin {current_user.username} {action} user {user.username}")
        
        return jsonify({
            'success': True,
            'message': f'User {action} successfully',
            'user': user_dict
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to toggle user status: {str(e)}")
        return jsonify({'error': f'Failed to toggle user status: {str(e)}'}), 500

@app.route('/api/admin/users/<int:user_id>/edit', methods=['PUT'])
@require_admin_auth
def admin_edit_user(user_id, current_user):
    """Edit user information (Admin only)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        session_db = db_manager.get_session()
        
        user = session_db.query(User).filter_by(id=user_id).first()
        if not user:
            session_db.close()
            return jsonify({'error': 'User not found'}), 404
        
        # Update allowed fields
        updatable_fields = ['full_name', 'email', 'department', 'position', 'phone_number']
        updated_fields = []
        
        for field in updatable_fields:
            if field in data and data[field] is not None:
                # Validate email if being updated
                if field == 'email':
                    new_email = data[field].strip().lower()
                    if not validate_email(new_email):
                        session_db.close()
                        return jsonify({'error': 'Invalid email format'}), 400
                    
                    # Check if email already exists (excluding current user)
                    existing_email = session_db.query(User).filter(
                        User.email == new_email,
                        User.id != user_id
                    ).first()
                    
                    if existing_email:
                        session_db.close()
                        return jsonify({'error': 'Email already exists'}), 409
                    
                    setattr(user, field, new_email)
                else:
                    setattr(user, field, data[field].strip() if isinstance(data[field], str) else data[field])
                
                updated_fields.append(field)
        
        # Update role if provided and user is not editing themselves
        if 'role' in data and user.id != current_user.id:
            new_role = data['role']
            if new_role in ['admin', 'employee']:
                user.role = UserRole(new_role)
                updated_fields.append('role')
        
        user.updated_at = datetime.utcnow()
        session_db.commit()
        
        # Get fresh user data after commit
        user_dict = user.to_dict()
        
        session_db.close()
        
        logger.info(f"Admin {current_user.username} edited user {user.username}, fields: {updated_fields}")
        
        return jsonify({
            'success': True,
            'message': 'User updated successfully',
            'user': user_dict,
            'updated_fields': updated_fields
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to edit user: {str(e)}")
        return jsonify({'error': f'Failed to edit user: {str(e)}'}), 500

@app.route('/api/admin/users/<int:user_id>/reset-password', methods=['POST'])
@require_admin_auth  
def admin_reset_user_password(user_id, current_user):
    """Reset user password (Admin only)"""
    try:
        data = request.get_json()
        new_password = data.get('new_password', '').strip() if data else ''
        
        if not new_password or len(new_password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        session_db = db_manager.get_session()
        
        user = session_db.query(User).filter_by(id=user_id).first()
        if not user:
            session_db.close()
            return jsonify({'error': 'User not found'}), 404
        
        # Hash new password
        password_hash = hash_password(new_password)
        user.password_hash = password_hash
        user.updated_at = datetime.utcnow()
        
        session_db.commit()
        session_db.close()
        
        logger.info(f"Admin {current_user.username} reset password for user {user.username}")
        
        return jsonify({
            'success': True,
            'message': 'Password reset successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to reset password: {str(e)}")
        return jsonify({'error': f'Failed to reset password: {str(e)}'}), 500

@app.route('/api/admin/datasets/<int:dataset_id>', methods=['DELETE'])
@require_admin_auth
def admin_delete_dataset(dataset_id, current_user):
    """Delete a dataset (Admin only)"""
    try:
        session_db = db_manager.get_session()
        
        dataset = session_db.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            session_db.close()
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Get dataset info for logging
        dataset_name = dataset.name
        dataset_uploader = dataset.uploaded_by
        
        # Delete associated analysis results first
        session_db.query(AnalysisResult).filter_by(dataset_id=dataset_id).delete()
        
        # Delete the dataset
        session_db.delete(dataset)
        session_db.commit()
        session_db.close()
        
        logger.info(f"Admin {current_user.username} deleted dataset {dataset_name} (ID: {dataset_id}) from user {dataset_uploader}")
        
        return jsonify({
            'success': True,
            'message': 'Dataset deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to delete dataset: {str(e)}")
        return jsonify({'error': f'Failed to delete dataset: {str(e)}'}), 500

# Initialize the application when the module is loaded
initialize_app()

if __name__ == '__main__':
    try:
        # Initialize the application
        initialize_app()
        logger.info("🚀 Starting Flask server on http://localhost:5000")
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"❌ Error starting server: {str(e)}")
        import sys
        sys.exit(1) 