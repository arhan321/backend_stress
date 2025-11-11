from flask import Flask, request, jsonify, session
from flask_cors import CORS
import bcrypt
import jwt
import uuid
from datetime import datetime, timedelta
import re
from loguru import logger
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import db_manager
from models.database_models import User, UserSession, UserPermission, AuditLog, UserRole

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
JWT_SECRET_KEY = 'your-jwt-secret-key-here-change-in-production'
JWT_EXPIRATION_HOURS = 24
JWT_REFRESH_EXPIRATION_DAYS = 30  # Refresh token expiration
PERSISTENT_SESSION_DAYS = 30  # Persistent session duration

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

def generate_refresh_token(user_id, username):
    """Generate refresh token for persistent sessions"""
    payload = {
        'user_id': user_id,
        'username': username,
        'type': 'refresh',
        'exp': datetime.utcnow() + timedelta(days=JWT_REFRESH_EXPIRATION_DAYS),
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

def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    return True, "Password is valid"

def log_user_activity(user_id, action, resource=None, resource_id=None, details=None, ip_address=None):
    """Log user activity for audit trail"""
    try:
        session_db = db_manager.get_session()
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=request.headers.get('User-Agent')
        )
        session_db.add(audit_log)
        session_db.commit()
        session_db.close()
    except Exception as e:
        logger.error(f"Failed to log user activity: {str(e)}")

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
        
        password_valid, password_message = validate_password(password)
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
        user_role = UserRole.ADMIN if user_count == 0 else UserRole.EMPLOYEE
        
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
            UserRole.ADMIN: ['admin_access', 'user_management', 'system_config', 'data_analysis', 'report_generation'],
            UserRole.HR_MANAGER: ['user_management', 'employee_data_access', 'report_generation'],
            UserRole.ANALYST: ['data_analysis', 'report_generation', 'model_management'],
            UserRole.EMPLOYEE: ['profile_access', 'basic_reports']
        }
        
        for permission in default_permissions.get(user_role, []):
            user_permission = UserPermission(
                user_id=user_id,
                permission=permission
            )
            session_db.add(user_permission)
        
        session_db.commit()
        session_db.close()
        
        # Log registration
        log_user_activity(
            user_id=user_id,
            action='user_registered',
            details={'role': user_role.value},
            ip_address=request.remote_addr
        )
        
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
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Authenticate user login"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        username_or_email = data.get('username_or_email', '').strip()
        password = data.get('password', '')
        remember_me = data.get('remember_me', False)
        
        if not username_or_email or not password:
            return jsonify({'error': 'Username/email and password are required'}), 400
        
        # Find user by username or email
        session_db = db_manager.get_session()
        
        user = session_db.query(User).filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()
        
        if not user:
            session_db.close()
            log_user_activity(
                user_id=None,
                action='login_failed',
                details={'reason': 'user_not_found', 'username_or_email': username_or_email},
                ip_address=request.remote_addr
            )
            return jsonify({'error': 'Invalid username/email or password'}), 401
        
        # Check if user is active
        if not user.is_active:
            session_db.close()
            log_user_activity(
                user_id=user.id,
                action='login_failed',
                details={'reason': 'account_inactive'},
                ip_address=request.remote_addr
            )
            return jsonify({'error': 'Account is inactive'}), 401
        
        # Verify password
        if not verify_password(password, user.password_hash):
            session_db.close()
            log_user_activity(
                user_id=user.id,
                action='login_failed',
                details={'reason': 'invalid_password'},
                ip_address=request.remote_addr
            )
            return jsonify({'error': 'Invalid username/email or password'}), 401
        
        # Generate JWT token
        token = generate_jwt_token(user.id, user.username)
        
        # Generate refresh token if remember me is enabled
        refresh_token = None
        if remember_me:
            refresh_token = generate_refresh_token(user.id, user.username)
        
        # Create session record
        session_token = str(uuid.uuid4())
        session_duration = timedelta(days=PERSISTENT_SESSION_DAYS) if remember_me else timedelta(hours=JWT_EXPIRATION_HOURS)
        expires_at = datetime.utcnow() + session_duration
        
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
        
        # Log successful login
        log_user_activity(
            user_id=user.id,
            action='login_success',
            details={'session_token': session_token},
            ip_address=request.remote_addr
        )
        
        logger.info(f"User logged in: {user.username}")
        
        response_data = {
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
        }
        
        # Add refresh token if remember me is enabled
        if refresh_token:
            response_data['refresh_token'] = refresh_token
            response_data['expires_in'] = int(session_duration.total_seconds())
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/api/auth/refresh', methods=['POST'])
def refresh_token():
    """Refresh JWT token using refresh token"""
    try:
        data = request.get_json()
        
        if not data or 'refresh_token' not in data:
            return jsonify({'error': 'Refresh token is required'}), 400
        
        refresh_token = data['refresh_token']
        
        # Verify refresh token
        try:
            payload = jwt.decode(refresh_token, JWT_SECRET_KEY, algorithms=['HS256'])
            
            if payload.get('type') != 'refresh':
                return jsonify({'error': 'Invalid token type'}), 401
                
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Refresh token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid refresh token'}), 401
        
        user_id = payload['user_id']
        username = payload['username']
        
        # Verify user still exists and is active
        session_db = db_manager.get_session()
        user = session_db.query(User).filter_by(id=user_id, is_active=True).first()
        
        if not user:
            session_db.close()
            return jsonify({'error': 'User not found or inactive'}), 404
        
        # Generate new access token
        new_token = generate_jwt_token(user.id, user.username)
        
        # Get user permissions
        permissions = session_db.query(UserPermission).filter_by(user_id=user.id).all()
        user_permissions = [perm.permission for perm in permissions]
        
        session_db.close()
        
        # Log token refresh
        log_user_activity(
            user_id=user_id,
            action='token_refreshed',
            ip_address=request.remote_addr
        )
        
        return jsonify({
            'success': True,
            'token': new_token,
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
                'permissions': user_permissions
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}")
        return jsonify({'error': f'Token refresh failed: {str(e)}'}), 500

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
        
        # Log logout
        log_user_activity(
            user_id=user_id,
            action='logout',
            details={'session_token': session_token},
            ip_address=request.remote_addr
        )
        
        logger.info(f"User logged out: user_id={user_id}")
        
        return jsonify({
            'success': True,
            'message': 'Logout successful'
        }), 200
        
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        return jsonify({'error': f'Logout failed: {str(e)}'}), 500

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

@app.route('/api/auth/profile', methods=['PUT'])
def update_profile():
    """Update user profile data"""
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
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get user
        session_db = db_manager.get_session()
        user = session_db.query(User).filter_by(id=user_id, is_active=True).first()
        
        if not user:
            session_db.close()
            return jsonify({'error': 'User not found or inactive'}), 404
        
        # Update allowed fields
        updatable_fields = ['full_name', 'department', 'position', 'phone_number']
        updated_fields = []
        
        for field in updatable_fields:
            if field in data and data[field] is not None:
                setattr(user, field, data[field].strip() if isinstance(data[field], str) else data[field])
                updated_fields.append(field)
        
        # Handle email update (with validation)
        if 'email' in data and data['email']:
            new_email = data['email'].strip().lower()
            if not validate_email(new_email):
                session_db.close()
                return jsonify({'error': 'Invalid email format'}), 400
            
            # Check if email already exists
            existing_email = session_db.query(User).filter(
                User.email == new_email,
                User.id != user_id
            ).first()
            
            if existing_email:
                session_db.close()
                return jsonify({'error': 'Email already exists'}), 409
            
            user.email = new_email
            user.email_verified = False  # Reset verification status
            updated_fields.append('email')
        
        # Update timestamp
        user.updated_at = datetime.utcnow()
        
        session_db.commit()
        session_db.close()
        
        # Log profile update
        log_user_activity(
            user_id=user_id,
            action='profile_updated',
            details={'updated_fields': updated_fields},
            ip_address=request.remote_addr
        )
        
        logger.info(f"Profile updated for user: {user.username}")
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully',
            'updated_fields': updated_fields
        }), 200
        
    except Exception as e:
        logger.error(f"Profile update failed: {str(e)}")
        return jsonify({'error': f'Failed to update profile: {str(e)}'}), 500

@app.route('/api/auth/change-password', methods=['POST'])
def change_password():
    """Change user password"""
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
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            return jsonify({'error': 'Current password and new password are required'}), 400
        
        # Validate new password
        password_valid, password_message = validate_password(new_password)
        if not password_valid:
            return jsonify({'error': password_message}), 400
        
        # Get user
        session_db = db_manager.get_session()
        user = session_db.query(User).filter_by(id=user_id, is_active=True).first()
        
        if not user:
            session_db.close()
            return jsonify({'error': 'User not found or inactive'}), 404
        
        # Verify current password
        if not verify_password(current_password, user.password_hash):
            session_db.close()
            log_user_activity(
                user_id=user_id,
                action='password_change_failed',
                details={'reason': 'invalid_current_password'},
                ip_address=request.remote_addr
            )
            return jsonify({'error': 'Current password is incorrect'}), 401
        
        # Update password
        user.password_hash = hash_password(new_password)
        user.updated_at = datetime.utcnow()
        
        session_db.commit()
        session_db.close()
        
        # Log password change
        log_user_activity(
            user_id=user_id,
            action='password_changed',
            ip_address=request.remote_addr
        )
        
        logger.info(f"Password changed for user: {user.username}")
        
        return jsonify({
            'success': True,
            'message': 'Password changed successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Password change failed: {str(e)}")
        return jsonify({'error': f'Failed to change password: {str(e)}'}), 500

@app.route('/api/auth/verify-token', methods=['POST'])
def verify_token():
    """Verify if token is valid"""
    try:
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'valid': False, 'error': 'No valid token provided'}), 401
        
        token = auth_header.split(' ')[1]
        
        # Verify token
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'valid': False, 'error': 'Invalid or expired token'}), 401
        
        # Check if user still exists and is active
        session_db = db_manager.get_session()
        user = session_db.query(User).filter_by(id=payload['user_id'], is_active=True).first()
        session_db.close()
        
        if not user:
            return jsonify({'valid': False, 'error': 'User not found or inactive'}), 404
        
        return jsonify({
            'valid': True,
            'user_id': payload['user_id'],
            'username': payload['username']
        }), 200
        
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        return jsonify({'valid': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 