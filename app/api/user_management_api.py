from flask import Blueprint, request, jsonify
from datetime import datetime
import bcrypt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import DatabaseManager
from models.database_models import User, AuditLog
import jwt
import logging

logger = logging.getLogger(__name__)

user_management_bp = Blueprint('user_management', __name__)
db_manager = DatabaseManager()

# JWT Configuration (should match auth_api.py)
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

@user_management_bp.route('/api/user/change-password', methods=['POST'])
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

@user_management_bp.route('/api/user/update-profile', methods=['PUT'])
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
        session_db.close()
        
        logger.info(f"Profile updated successfully for user ID: {user_id}")
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully',
            'user': {
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
        }), 200
        
    except Exception as e:
        logger.error(f"Profile update failed: {str(e)}")
        return jsonify({'error': f'Profile update failed: {str(e)}'}), 500

@user_management_bp.route('/api/user/delete-account', methods=['DELETE'])
def delete_account():
    """Soft delete user account (deactivate)"""
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
        password = data.get('password', '').strip() if data else ''
        
        if not password:
            return jsonify({'error': 'Password confirmation required'}), 400
        
        # Get user from database
        session_db = db_manager.get_session()
        user = session_db.query(User).filter_by(id=user_id, is_active=True).first()
        
        if not user:
            session_db.close()
            return jsonify({'error': 'User not found or inactive'}), 404
        
        # Verify password
        if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            session_db.close()
            return jsonify({'error': 'Incorrect password'}), 401
        
        # Soft delete (deactivate)
        user.is_active = False
        user.updated_at = datetime.utcnow()
        
        # Create audit log
        audit_log = AuditLog(
            user_id=user_id,
            action='account_deletion',
            resource='user',
            resource_id=user_id,
            details={'message': 'User account deactivated'},
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        
        session_db.add(audit_log)
        session_db.commit()
        session_db.close()
        
        logger.info(f"Account deactivated for user ID: {user_id}")
        
        return jsonify({
            'success': True,
            'message': 'Account deactivated successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Account deletion failed: {str(e)}")
        return jsonify({'error': f'Account deletion failed: {str(e)}'}), 500 