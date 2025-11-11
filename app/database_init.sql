-- Employee Stress Analysis Database Schema
-- This script creates the database and tables for the stress analysis system
-- To run in XAMPP: Open phpMyAdmin -> SQL tab -> paste and execute this script

-- Create database
CREATE DATABASE IF NOT EXISTS `stress_analysis` 
CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE `stress_analysis`;

-- Table: employees
-- Stores employee data and stress factors
CREATE TABLE `employees` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `employee_id` varchar(50) NOT NULL,
  `department` varchar(100) NOT NULL,
  `age` int(11) DEFAULT NULL,
  `position` varchar(200) DEFAULT NULL,
  `workload` decimal(3,1) NOT NULL,
  `work_life_balance` decimal(3,1) NOT NULL,
  `team_conflict` decimal(3,1) NOT NULL,
  `management_support` decimal(3,1) NOT NULL,
  `work_environment` decimal(3,1) NOT NULL,
  `stress_level` decimal(5,2) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `employee_id` (`employee_id`),
  KEY `idx_employee_department` (`department`),
  KEY `idx_employee_stress_level` (`stress_level`),
  KEY `idx_employee_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: datasets
-- Stores information about uploaded datasets
CREATE TABLE `datasets` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `description` text DEFAULT NULL,
  `file_path` varchar(500) DEFAULT NULL,
  `file_size` int(11) DEFAULT NULL,
  `record_count` int(11) NOT NULL DEFAULT 0,
  `upload_date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `status` varchar(50) DEFAULT 'uploaded',
  `processing_start_time` timestamp NULL DEFAULT NULL,
  `processing_end_time` timestamp NULL DEFAULT NULL,
  `error_message` text DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_dataset_upload_date` (`upload_date`),
  KEY `idx_dataset_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: analysis_results
-- Stores ML analysis results
CREATE TABLE `analysis_results` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `dataset_id` int(11) NOT NULL,
  `analysis_type` varchar(100) NOT NULL,
  `overall_stress_level` decimal(5,2) NOT NULL,
  `analysis_date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `factor_contributions` json DEFAULT NULL,
  `model_accuracy` decimal(5,4) DEFAULT NULL,
  `confidence_score` decimal(5,4) DEFAULT NULL,
  `department_breakdown` json DEFAULT NULL,
  `correlation_matrix` json DEFAULT NULL,
  `feature_importance` json DEFAULT NULL,
  `processing_time_seconds` decimal(8,2) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_analysis_dataset_id` (`dataset_id`),
  KEY `idx_analysis_date` (`analysis_date`),
  KEY `idx_analysis_type` (`analysis_type`),
  CONSTRAINT `fk_analysis_dataset` FOREIGN KEY (`dataset_id`) REFERENCES `datasets` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: stress_predictions
-- Stores individual stress predictions
CREATE TABLE `stress_predictions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `employee_id` int(11) NOT NULL,
  `predicted_stress_level` decimal(5,2) NOT NULL,
  `confidence_score` decimal(5,4) NOT NULL,
  `prediction_date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `model_version` varchar(50) DEFAULT NULL,
  `input_features` json DEFAULT NULL,
  `factor_weights` json DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_prediction_employee_id` (`employee_id`),
  KEY `idx_prediction_date` (`prediction_date`),
  KEY `idx_prediction_stress_level` (`predicted_stress_level`),
  CONSTRAINT `fk_prediction_employee` FOREIGN KEY (`employee_id`) REFERENCES `employees` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: model_training
-- Stores model training history and metrics
CREATE TABLE `model_training` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `model_type` varchar(100) NOT NULL,
  `model_version` varchar(50) NOT NULL,
  `training_date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `training_parameters` json DEFAULT NULL,
  `train_accuracy` decimal(5,4) DEFAULT NULL,
  `validation_accuracy` decimal(5,4) DEFAULT NULL,
  `test_accuracy` decimal(5,4) DEFAULT NULL,
  `train_loss` decimal(8,6) DEFAULT NULL,
  `validation_loss` decimal(8,6) DEFAULT NULL,
  `test_loss` decimal(8,6) DEFAULT NULL,
  `model_file_path` varchar(500) DEFAULT NULL,
  `training_data_size` int(11) DEFAULT NULL,
  `training_time_seconds` decimal(8,2) DEFAULT NULL,
  `is_active` tinyint(1) DEFAULT 1,
  PRIMARY KEY (`id`),
  KEY `idx_training_date` (`training_date`),
  KEY `idx_model_type` (`model_type`),
  KEY `idx_model_version` (`model_version`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: system_logs
-- Stores application logs
CREATE TABLE `system_logs` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `level` varchar(20) NOT NULL,
  `module` varchar(100) NOT NULL,
  `message` text NOT NULL,
  `additional_data` json DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_log_timestamp` (`timestamp`),
  KEY `idx_log_level` (`level`),
  KEY `idx_log_module` (`module`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert sample data
-- Sample departments for reference
INSERT INTO `system_logs` (`level`, `module`, `message`, `additional_data`) VALUES
('INFO', 'DATABASE', 'Database schema initialized successfully', '{"version": "1.0", "tables_created": 6}');

-- Sample employee data (optional - remove if you want to start clean)
INSERT INTO `employees` (`employee_id`, `department`, `age`, `position`, `workload`, `work_life_balance`, `team_conflict`, `management_support`, `work_environment`, `stress_level`) VALUES
('EMP0001', 'IT', 28, 'Developer', 7.5, 6.2, 4.0, 6.8, 7.0, 45.2),
('EMP0002', 'Finance', 35, 'Analyst', 8.2, 5.5, 5.2, 5.8, 6.5, 62.3),
('EMP0003', 'HR', 31, 'Specialist', 6.0, 7.8, 3.5, 7.5, 8.0, 32.1),
('EMP0004', 'Marketing', 29, 'Manager', 8.5, 5.0, 6.0, 6.0, 6.8, 58.7),
('EMP0005', 'Operations', 42, 'Supervisor', 7.0, 6.5, 4.8, 6.2, 7.2, 48.9);

-- Create database user (optional - for production use)
-- GRANT ALL PRIVILEGES ON stress_analysis.* TO 'stress_user'@'localhost' IDENTIFIED BY 'secure_password';
-- FLUSH PRIVILEGES;

-- Show tables to confirm creation
SHOW TABLES; 

-- ===========================================
-- Authentication Tables
-- ===========================================

-- Table: users
-- Stores user authentication and profile data
CREATE TABLE `users` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `email` varchar(100) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `full_name` varchar(100) NOT NULL,
  `role` enum('admin','employee') DEFAULT 'employee',
  `department` varchar(100) DEFAULT NULL,
  `position` varchar(200) DEFAULT NULL,
  `phone_number` varchar(20) DEFAULT NULL,
  `profile_picture` varchar(500) DEFAULT NULL,
  `is_active` tinyint(1) DEFAULT 1,
  `email_verified` tinyint(1) DEFAULT 0,
  `last_login` timestamp NULL DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `reset_token` varchar(255) DEFAULT NULL,
  `reset_token_expires` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`),
  UNIQUE KEY `email` (`email`),
  KEY `idx_user_role` (`role`),
  KEY `idx_user_department` (`department`),
  KEY `idx_user_active` (`is_active`),
  KEY `idx_user_email_verified` (`email_verified`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: user_sessions
-- Stores active user sessions
CREATE TABLE `user_sessions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `session_token` varchar(255) NOT NULL,
  `device_info` text DEFAULT NULL,
  `ip_address` varchar(45) DEFAULT NULL,
  `expires_at` timestamp NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `last_activity` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `is_active` tinyint(1) DEFAULT 1,
  PRIMARY KEY (`id`),
  UNIQUE KEY `session_token` (`session_token`),
  KEY `idx_session_user_id` (`user_id`),
  KEY `idx_session_expires` (`expires_at`),
  KEY `idx_session_active` (`is_active`),
  CONSTRAINT `fk_session_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: user_permissions
-- Stores user permissions for role-based access control
CREATE TABLE `user_permissions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `permission` varchar(100) NOT NULL,
  `granted_by` int(11) DEFAULT NULL,
  `granted_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `expires_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_permission_user_id` (`user_id`),
  KEY `idx_permission_name` (`permission`),
  KEY `idx_permission_granted_by` (`granted_by`),
  CONSTRAINT `fk_permission_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_permission_granted_by` FOREIGN KEY (`granted_by`) REFERENCES `users` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: audit_logs
-- Stores audit trail of user activities
CREATE TABLE `audit_logs` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) DEFAULT NULL,
  `action` varchar(100) NOT NULL,
  `resource` varchar(100) DEFAULT NULL,
  `resource_id` int(11) DEFAULT NULL,
  `details` json DEFAULT NULL,
  `ip_address` varchar(45) DEFAULT NULL,
  `user_agent` text DEFAULT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_audit_user_id` (`user_id`),
  KEY `idx_audit_action` (`action`),
  KEY `idx_audit_resource` (`resource`),
  KEY `idx_audit_timestamp` (`timestamp`),
  CONSTRAINT `fk_audit_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===========================================
-- Insert Default Admin User
-- ===========================================

-- Insert default admin user
-- Password: admin123 (hashed using bcrypt)
INSERT INTO `users` (`username`, `email`, `password_hash`, `full_name`, `role`, `department`, `position`, `is_active`, `email_verified`) VALUES
('admin', 'admin@example.com', '$2b$12$YgJteZ7X5z/mGnPOaLKYAOJPyINkCRm9zLMfYGhgBQ8zchWVYYsIe', 'System Administrator', 'admin', 'IT', 'Administrator', 1, 1);

-- Insert admin permissions
SET @admin_id = LAST_INSERT_ID();
INSERT INTO `user_permissions` (`user_id`, `permission`) VALUES
(@admin_id, 'admin_access'),
(@admin_id, 'user_management'),
(@admin_id, 'system_config'),
(@admin_id, 'data_analysis'),
(@admin_id, 'report_generation');

-- Log the admin user creation
INSERT INTO `system_logs` (`level`, `module`, `message`, `additional_data`) VALUES
('INFO', 'DATABASE', 'Default admin user created', '{"username": "admin", "email": "admin@example.com"}');

-- Show all tables including new ones
SHOW TABLES; 