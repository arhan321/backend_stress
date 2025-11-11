from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import bcrypt
import sys
import os
from sqlalchemy import and_, or_, desc, func
from sqlalchemy.orm import joinedload

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import DatabaseManager
from models.database_models import User, Dataset, AnalysisResult, AuditLog, UserPermission, UserRole
import jwt
import logging

logger = logging.getLogger(__name__)

admin_user_mgmt_bp = Blueprint('admin_user_management', __name__)
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
                return jsonify({'error': 'Admin access required'}), 403
            
            # Add user to request context
            request.current_user = user
            return f(*args, **kwargs)
        finally:
            session_db.close()
    
    decorated_function.__name__ = f.__name__
    return decorated_function

@admin_user_mgmt_bp.route('/api/admin/users', methods=['GET'])
@require_admin_auth
def get_all_users():
    """Get all users with filtering and pagination"""
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
                or_(
                    User.username.ilike(search_pattern),
                    User.email.ilike(search_pattern),
                    User.full_name.ilike(search_pattern),
                    User.department.ilike(search_pattern),
                    User.position.ilike(search_pattern)
                )
            )
        
        if role_filter and role_filter in ['admin', 'employee']:
            query = query.filter(User.role == UserRole(role_filter))
        
        if department_filter:
            query = query.filter(User.department.ilike(f"%{department_filter}%"))
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination and ordering
        users = query.order_by(desc(User.created_at)).offset(offset).limit(limit).all()
        
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

@admin_user_mgmt_bp.route('/api/admin/users/<int:user_id>', methods=['GET'])
@require_admin_auth
def get_user_details(user_id):
    """Get detailed information about a specific user"""
    try:
        session_db = db_manager.get_session()
        
        # Get user with relationships
        user = session_db.query(User).options(
            joinedload(User.datasets),
            joinedload(User.permissions),
            joinedload(User.audit_logs)
        ).filter_by(id=user_id).first()
        
        if not user:
            session_db.close()
            return jsonify({'error': 'User not found'}), 404
        
        # Get user statistics
        dataset_count = session_db.query(Dataset).filter_by(uploaded_by=user_id).count()
        processed_datasets = session_db.query(Dataset).filter_by(
            uploaded_by=user_id, status='processed'
        ).count()
        
        # Get recent activity
        recent_logs = session_db.query(AuditLog).filter_by(user_id=user_id).order_by(
            desc(AuditLog.timestamp)
        ).limit(10).all()
        
        session_db.close()
        
        # Format response
        user_data = user.to_dict()
        user_data['statistics'] = {
            'total_datasets': dataset_count,
            'processed_datasets': processed_datasets,
            'success_rate': (processed_datasets / dataset_count * 100) if dataset_count > 0 else 0
        }
        
        user_data['recent_datasets'] = [
            dataset.to_dict() for dataset in user.datasets[-5:]  # Last 5 datasets
        ]
        
        user_data['permissions'] = [
            perm.to_dict() for perm in user.permissions
        ]
        
        user_data['recent_activity'] = [
            log.to_dict() for log in recent_logs
        ]
        
        return jsonify({
            'success': True,
            'user': user_data
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get user details: {str(e)}")
        return jsonify({'error': f'Failed to get user details: {str(e)}'}), 500

@admin_user_mgmt_bp.route('/api/admin/users/<int:user_id>/datasets', methods=['GET'])
@require_admin_auth
def get_user_datasets(user_id):
    """Get all datasets uploaded by a specific user"""
    try:
        # Get query parameters
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
        datasets = query.order_by(desc(Dataset.upload_date)).offset(offset).limit(limit).all()
        
        session_db.close()
        
        # Format response
        datasets_data = [dataset.to_dict(include_uploader=True) for dataset in datasets]
        
        return jsonify({
            'success': True,
            'user': user.to_dict(),
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

@admin_user_mgmt_bp.route('/api/admin/datasets', methods=['GET'])
@require_admin_auth
def get_all_datasets():
    """Get all datasets with user information"""
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        search = request.args.get('search', '').strip()
        status_filter = request.args.get('status', '').strip()
        user_filter = request.args.get('user_id', '').strip()
        
        offset = (page - 1) * limit
        
        session_db = db_manager.get_session()
        
        # Build query with user information
        query = session_db.query(Dataset).options(joinedload(Dataset.uploader))
        
        # Apply filters
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    Dataset.name.ilike(search_pattern),
                    Dataset.description.ilike(search_pattern)
                )
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
        datasets = query.order_by(desc(Dataset.upload_date)).offset(offset).limit(limit).all()
        
        session_db.close()
        
        # Format response
        datasets_data = [dataset.to_dict(include_uploader=True) for dataset in datasets]
        
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

@admin_user_mgmt_bp.route('/api/admin/datasets/<int:dataset_id>', methods=['GET'])
@require_admin_auth
def get_dataset_details(dataset_id):
    """Get detailed information about a specific dataset"""
    try:
        session_db = db_manager.get_session()
        
        # Get dataset with relationships
        dataset = session_db.query(Dataset).options(
            joinedload(Dataset.uploader),
            joinedload(Dataset.analysis_results)
        ).filter_by(id=dataset_id).first()
        
        if not dataset:
            session_db.close()
            return jsonify({'error': 'Dataset not found'}), 404
        
        session_db.close()
        
        # Format response
        dataset_data = dataset.to_dict(include_uploader=True)
        dataset_data['analysis_results'] = [
            result.to_dict() for result in dataset.analysis_results
        ]
        
        return jsonify({
            'success': True,
            'dataset': dataset_data
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get dataset details: {str(e)}")
        return jsonify({'error': f'Failed to get dataset details: {str(e)}'}), 500

@admin_user_mgmt_bp.route('/api/admin/users/<int:user_id>/toggle-status', methods=['POST'])
@require_admin_auth
def toggle_user_status(user_id):
    """Activate or deactivate a user account"""
    try:
        session_db = db_manager.get_session()
        
        user = session_db.query(User).filter_by(id=user_id).first()
        if not user:
            session_db.close()
            return jsonify({'error': 'User not found'}), 404
        
        # Prevent admin from deactivating themselves
        if user.id == request.current_user.id:
            session_db.close()
            return jsonify({'error': 'Cannot deactivate your own account'}), 400
        
        # Toggle status
        user.is_active = not user.is_active
        user.updated_at = datetime.utcnow()
        
        # Create audit log
        action = 'user_activated' if user.is_active else 'user_deactivated'
        audit_log = AuditLog(
            user_id=request.current_user.id,
            action=action,
            resource='user',
            resource_id=user_id,
            details={
                'target_user': user.email,
                'new_status': 'active' if user.is_active else 'inactive'
            },
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        
        session_db.add(audit_log)
        session_db.commit()
        session_db.close()
        
        return jsonify({
            'success': True,
            'message': f'User {"activated" if user.is_active else "deactivated"} successfully',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to toggle user status: {str(e)}")
        return jsonify({'error': f'Failed to toggle user status: {str(e)}'}), 500

@admin_user_mgmt_bp.route('/api/admin/users/<int:user_id>/reset-password', methods=['POST'])
@require_admin_auth
def reset_user_password(user_id):
    """Reset a user's password (admin only)"""
    try:
        data = request.get_json()
        new_password = data.get('new_password', '').strip()
        
        if not new_password or len(new_password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters long'}), 400
        
        session_db = db_manager.get_session()
        
        user = session_db.query(User).filter_by(id=user_id).first()
        if not user:
            session_db.close()
            return jsonify({'error': 'User not found'}), 404
        
        # Hash new password
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(new_password.encode('utf-8'), salt).decode('utf-8')
        
        user.password_hash = password_hash
        user.updated_at = datetime.utcnow()
        
        # Create audit log
        audit_log = AuditLog(
            user_id=request.current_user.id,
            action='password_reset_by_admin',
            resource='user',
            resource_id=user_id,
            details={'target_user': user.email},
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        
        session_db.add(audit_log)
        session_db.commit()
        session_db.close()
        
        return jsonify({
            'success': True,
            'message': 'Password reset successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to reset password: {str(e)}")
        return jsonify({'error': f'Failed to reset password: {str(e)}'}), 500

@admin_user_mgmt_bp.route('/api/admin/datasets/<int:dataset_id>/delete', methods=['DELETE'])
@require_admin_auth
def delete_dataset(dataset_id):
    """Delete a dataset (admin only)"""
    try:
        session_db = db_manager.get_session()
        
        dataset = session_db.query(Dataset).filter_by(id=dataset_id).first()
        if not dataset:
            session_db.close()
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Store dataset info for audit log
        dataset_info = {
            'name': dataset.name,
            'uploader_id': dataset.uploaded_by,
            'file_path': dataset.file_path
        }
        
        # Delete file if exists
        if dataset.file_path and os.path.exists(dataset.file_path):
            try:
                os.remove(dataset.file_path)
            except Exception as e:
                logger.warning(f"Failed to delete file {dataset.file_path}: {e}")
        
        # Delete dataset (will cascade to analysis results)
        session_db.delete(dataset)
        
        # Create audit log
        audit_log = AuditLog(
            user_id=request.current_user.id,
            action='dataset_deleted_by_admin',
            resource='dataset',
            resource_id=dataset_id,
            details=dataset_info,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        
        session_db.add(audit_log)
        session_db.commit()
        session_db.close()
        
        return jsonify({
            'success': True,
            'message': 'Dataset deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to delete dataset: {str(e)}")
        return jsonify({'error': f'Failed to delete dataset: {str(e)}'}), 500

@admin_user_mgmt_bp.route('/api/admin/dashboard/stats', methods=['GET'])
@require_admin_auth
def get_admin_dashboard_stats():
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
        recent_users = session_db.query(User).order_by(desc(User.created_at)).limit(5).all()
        recent_datasets = session_db.query(Dataset).options(
            joinedload(Dataset.uploader)
        ).order_by(desc(Dataset.upload_date)).limit(5).all()
        
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
                    'new_datasets': [dataset.to_dict(include_uploader=True) for dataset in recent_datasets]
                }
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get admin dashboard stats: {str(e)}")
        return jsonify({'error': f'Failed to get admin dashboard stats: {str(e)}'}), 500 