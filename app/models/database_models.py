from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON, Index, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import json
import enum

Base = declarative_base()

class UserRole(enum.Enum):
    """User role enumeration"""
    admin = "admin"
    employee = "employee"

class Employee(Base):
    """Employee data model"""
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(50), unique=True, nullable=False, index=True)
    department = Column(String(100), nullable=False)
    age = Column(Integer, nullable=True)
    position = Column(String(200), nullable=True)
    workload = Column(Float, nullable=False)  # Scale 1-10
    work_life_balance = Column(Float, nullable=False)  # Scale 1-10
    team_conflict = Column(Float, nullable=False)  # Scale 1-10
    management_support = Column(Float, nullable=False)  # Scale 1-10
    work_environment = Column(Float, nullable=False)  # Scale 1-10
    stress_level = Column(Float, nullable=False)  # Scale 0-100
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    predictions = relationship("StressPrediction", back_populates="employee")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_employee_department', 'department'),
        Index('idx_employee_stress_level', 'stress_level'),
        Index('idx_employee_created_at', 'created_at'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'department': self.department,
            'age': self.age,
            'position': self.position,
            'workload': self.workload,
            'work_life_balance': self.work_life_balance,
            'team_conflict': self.team_conflict,
            'management_support': self.management_support,
            'work_environment': self.work_environment,
            'stress_level': self.stress_level,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Dataset(Base):
    """Dataset information model"""
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)  # in bytes
    record_count = Column(Integer, nullable=False, default=0)
    upload_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default='uploaded')  # uploaded, processing, processed, failed
    processing_start_time = Column(DateTime, nullable=True)
    processing_end_time = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # User relationship
    uploaded_by = Column(Integer, ForeignKey('users.id'), nullable=True)
    
    # Relationships
    analysis_results = relationship("AnalysisResult", back_populates="dataset")
    uploader = relationship("User", back_populates="datasets")
    
    def to_dict(self, include_uploader=False):
        data = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'record_count': self.record_count,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'status': self.status,
            'processing_start_time': self.processing_start_time.isoformat() if self.processing_start_time else None,
            'processing_end_time': self.processing_end_time.isoformat() if self.processing_end_time else None,
            'error_message': self.error_message,
            'uploaded_by': self.uploaded_by
        }
        
        if include_uploader and self.uploader:
            data['uploader'] = {
                'id': self.uploader.id,
                'username': self.uploader.username,
                'email': self.uploader.email,
                'full_name': self.uploader.full_name,
                'department': self.uploader.department,
                'role': self.uploader.role.value if self.uploader.role else None
            }
        
        return data

class AnalysisResult(Base):
    """Analysis results model for storing ML model outputs"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    analysis_type = Column(String(100), nullable=False)  # deep_learning, ncf, combined
    overall_stress_level = Column(Float, nullable=False)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    
    # Factor contributions as JSON
    factor_contributions = Column(JSON, nullable=True)
    
    # Model performance metrics
    model_accuracy = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Detailed results
    department_breakdown = Column(JSON, nullable=True)
    correlation_matrix = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    
    # Processing time
    processing_time_seconds = Column(Float, nullable=True)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="analysis_results")
    
    # Indexes
    __table_args__ = (
        Index('idx_analysis_dataset_id', 'dataset_id'),
        Index('idx_analysis_date', 'analysis_date'),
        Index('idx_analysis_type', 'analysis_type'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'analysis_type': self.analysis_type,
            'overall_stress_level': self.overall_stress_level,
            'analysis_date': self.analysis_date.isoformat() if self.analysis_date else None,
            'factor_contributions': self.factor_contributions,
            'model_accuracy': self.model_accuracy,
            'confidence_score': self.confidence_score,
            'department_breakdown': self.department_breakdown,
            'correlation_matrix': self.correlation_matrix,
            'feature_importance': self.feature_importance,
            'processing_time_seconds': self.processing_time_seconds
        }

class StressPrediction(Base):
    """Individual stress predictions model"""
    __tablename__ = 'stress_predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey('employees.id'), nullable=False)
    predicted_stress_level = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(50), nullable=True)
    
    # Input features used for prediction
    input_features = Column(JSON, nullable=True)
    
    # Prediction breakdown
    factor_weights = Column(JSON, nullable=True)
    
    # Relationships
    employee = relationship("Employee", back_populates="predictions")
    
    # Indexes
    __table_args__ = (
        Index('idx_prediction_employee_id', 'employee_id'),
        Index('idx_prediction_date', 'prediction_date'),
        Index('idx_prediction_stress_level', 'predicted_stress_level'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'predicted_stress_level': self.predicted_stress_level,
            'confidence_score': self.confidence_score,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'model_version': self.model_version,
            'input_features': self.input_features,
            'factor_weights': self.factor_weights
        }

class ModelTraining(Base):
    """Model training history and metrics"""
    __tablename__ = 'model_training'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(100), nullable=False)  # deep_learning, ncf
    model_version = Column(String(50), nullable=False)
    training_date = Column(DateTime, default=datetime.utcnow)
    
    # Training parameters
    training_parameters = Column(JSON, nullable=True)
    
    # Performance metrics
    train_accuracy = Column(Float, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    train_loss = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)
    test_loss = Column(Float, nullable=True)
    
    # Model info
    model_file_path = Column(String(500), nullable=True)
    training_data_size = Column(Integer, nullable=True)
    training_time_seconds = Column(Float, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'model_type': self.model_type,
            'model_version': self.model_version,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'training_parameters': self.training_parameters,
            'train_accuracy': self.train_accuracy,
            'validation_accuracy': self.validation_accuracy,
            'test_accuracy': self.test_accuracy,
            'train_loss': self.train_loss,
            'validation_loss': self.validation_loss,
            'test_loss': self.test_loss,
            'model_file_path': self.model_file_path,
            'training_data_size': self.training_data_size,
            'training_time_seconds': self.training_time_seconds,
            'is_active': self.is_active
        }

class SystemLog(Base):
    """System logs for monitoring and debugging"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, DEBUG
    module = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    additional_data = Column(JSON, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_log_timestamp', 'timestamp'),
        Index('idx_log_level', 'level'),
        Index('idx_log_module', 'module'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'level': self.level,
            'module': self.module,
            'message': self.message,
            'additional_data': self.additional_data
        }

class User(Base):
    """User authentication model"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.employee)
    department = Column(String(100), nullable=True)
    position = Column(String(200), nullable=True)
    phone_number = Column(String(20), nullable=True)
    profile_picture = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)
    email_verified = Column(Boolean, default=False)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    reset_token = Column(String(255), nullable=True)
    reset_token_expires = Column(DateTime, nullable=True)
    
    # Relationships
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    permissions = relationship("UserPermission", back_populates="user", foreign_keys="UserPermission.user_id", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")
    granted_permissions = relationship("UserPermission", foreign_keys="UserPermission.granted_by")
    datasets = relationship("Dataset", back_populates="uploader")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_role', 'role'),
        Index('idx_user_department', 'department'),
        Index('idx_user_active', 'is_active'),
        Index('idx_user_email_verified', 'email_verified'),
    )
    
    def to_dict(self, include_sensitive=False):
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role.value if self.role else None,
            'department': self.department,
            'position': self.position,
            'phone_number': self.phone_number,
            'profile_picture': self.profile_picture,
            'is_active': self.is_active,
            'email_verified': self.email_verified,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_sensitive:
            data['reset_token'] = self.reset_token
            data['reset_token_expires'] = self.reset_token_expires.isoformat() if self.reset_token_expires else None
        
        return data

class UserSession(Base):
    """User session management model"""
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    device_info = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)  # Supports IPv6
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    # Indexes
    __table_args__ = (
        Index('idx_session_user_id', 'user_id'),
        Index('idx_session_expires', 'expires_at'),
        Index('idx_session_active', 'is_active'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'session_token': self.session_token,
            'device_info': self.device_info,
            'ip_address': self.ip_address,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'is_active': self.is_active
        }

class UserPermission(Base):
    """User permissions model for role-based access control"""
    __tablename__ = 'user_permissions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    permission = Column(String(100), nullable=False)
    granted_by = Column(Integer, ForeignKey('users.id'), nullable=True)
    granted_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="permissions", foreign_keys=[user_id])
    granted_by_user = relationship("User", foreign_keys=[granted_by])
    
    # Indexes
    __table_args__ = (
        Index('idx_permission_user_id', 'user_id'),
        Index('idx_permission_name', 'permission'),
        Index('idx_permission_granted_by', 'granted_by'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'permission': self.permission,
            'granted_by': self.granted_by,
            'granted_at': self.granted_at.isoformat() if self.granted_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }

class AuditLog(Base):
    """Audit log model for tracking user activities"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    action = Column(String(100), nullable=False)
    resource = Column(String(100), nullable=True)
    resource_id = Column(Integer, nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_user_id', 'user_id'),
        Index('idx_audit_action', 'action'),
        Index('idx_audit_resource', 'resource'),
        Index('idx_audit_timestamp', 'timestamp'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'action': self.action,
            'resource': self.resource,
            'resource_id': self.resource_id,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        } 