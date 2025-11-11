#!/usr/bin/env python3
"""
Utility functions to handle NaN values in JSON responses
"""

import pandas as pd
import numpy as np
import json
from typing import Any, Dict, List, Union

def clean_nan_values(obj: Any) -> Any:
    """
    Recursively clean NaN and infinity values from data structures
    before JSON serialization.
    
    Args:
        obj: The object to clean (dict, list, or primitive)
        
    Returns:
        Cleaned object with NaN/inf values replaced
    """
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(v) for v in obj]
    elif isinstance(obj, (int, float)):
        if pd.isna(obj) or np.isnan(obj):
            return 0.0  # Replace NaN with 0
        elif np.isinf(obj):
            return 0.0  # Replace infinity with 0
        else:
            return float(obj)  # Ensure it's a standard float
    elif pd.isna(obj):
        return None  # Replace pandas NaN with None
    else:
        return obj

def safe_correlation_calculation(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
    """
    Safely calculate correlation matrix and handle NaN values
    
    Args:
        df: DataFrame to calculate correlations for
        numeric_columns: List of numeric column names
        
    Returns:
        Dictionary with cleaned correlation data
    """
    try:
        # Calculate correlation matrix
        correlation_matrix = df[numeric_columns].corr()
        
        # Clean the correlation matrix
        correlation_dict = correlation_matrix.to_dict()
        cleaned_correlation = clean_nan_values(correlation_dict)
        
        return {
            'correlation_matrix': cleaned_correlation,
            'correlation_calculated': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'correlation_matrix': {},
            'correlation_calculated': False,
            'error': str(e)
        }

def enhanced_factor_importance_calculation(df: pd.DataFrame, target_column: str = 'stress_level', 
                                         overall_stress_level: float = None) -> Dict[str, Any]:
    """
    Calculate SYNCHRONIZED factor importance that aligns with overall stress level
    
    This function implements a comprehensive approach that ensures:
    1. Factor importance percentages are meaningful and proportional
    2. They align with the overall stress level
    3. Correlations are dynamic and vary with different datasets
    4. Factors contribute to a coherent total importance
    
    Args:
        df: DataFrame with factor and stress data
        target_column: Target column to calculate importance against
        overall_stress_level: Overall stress level to synchronize with
        
    Returns:
        Dictionary with synchronized factor importance data
    """
    try:
        # Calculate overall stress level if not provided
        if overall_stress_level is None:
            overall_stress_level = float(df[target_column].mean())
        
        # Get factor columns (exclude target)
        factor_columns = [col for col in df.columns if col != target_column and 
                         col not in ['employee_id', 'department']]
        
        if not factor_columns:
            return {}
        
        # Calculate correlation matrix
        correlation_matrix = df[factor_columns + [target_column]].corr()
        stress_correlations = correlation_matrix[target_column].abs()
        
        # STEP 1: Calculate variance-based importance
        variance_importance = {}
        for factor in factor_columns:
            factor_variance = df[factor].var()
            stress_variance = df[target_column].var()
            if stress_variance > 0:
                variance_contribution = factor_variance / stress_variance
            else:
                variance_contribution = 1.0
            variance_importance[factor] = min(variance_contribution, 2.0)  # Cap at 2.0
        
        # STEP 2: Calculate contribution-based importance (how much each factor affects stress)
        contribution_importance = {}
        for factor in factor_columns:
            # Calculate how much this factor deviates from neutral (5.5 for 1-10 scale)
            factor_mean = df[factor].mean()
            neutral_point = 5.5
            deviation_from_neutral = abs(factor_mean - neutral_point)
            max_possible_deviation = 4.5  # Maximum deviation from neutral
            
            # Calculate factor's contribution to overall stress
            contribution_score = (deviation_from_neutral / max_possible_deviation) * 100
            contribution_importance[factor] = min(contribution_score, 100.0)
        
        # STEP 3: Combine correlation, variance, and contribution into unified importance
        combined_importance = {}
        for factor in factor_columns:
            # Get safe correlation value
            correlation = stress_correlations.get(factor, 0.0)
            if pd.isna(correlation) or np.isinf(correlation):
                correlation = 0.1  # Default small correlation
            
            # Combine all three measures with weights
            correlation_weight = abs(correlation) * 100  # 0-100
            variance_weight = variance_importance.get(factor, 1.0) * 50  # 0-100  
            contribution_weight = contribution_importance.get(factor, 50.0)  # 0-100
            
            # Weighted combination (ensure minimum 5% for each factor)
            combined_score = (
                correlation_weight * 0.4 +
                variance_weight * 0.3 + 
                contribution_weight * 0.3
            )
            
            # Apply minimum threshold and stress level scaling
            base_importance = max(combined_score, 5.0)  # Minimum 5%
            
            # Scale based on overall stress level to ensure synchronization
            stress_multiplier = overall_stress_level / 50.0  # Scale relative to 50% baseline
            synchronized_importance = base_importance * stress_multiplier
            
            combined_importance[factor] = synchronized_importance
        
        # STEP 4: Normalize to ensure coherent total while maintaining proportions
        total_raw_importance = sum(combined_importance.values())
        
        # Calculate target total based on stress level (higher stress = higher total factor impact)
        if overall_stress_level > 70:
            target_total = 400  # High stress: factors have major impact
        elif overall_stress_level > 40:
            target_total = 300  # Medium stress: moderate factor impact  
        else:
            target_total = 200  # Low stress: lower factor impact
        
        # Normalize while maintaining relative proportions
        normalization_factor = target_total / total_raw_importance if total_raw_importance > 0 else 1.0
        
        # Build final result with all measures
        factor_importance = {}
        for i, (factor, importance) in enumerate(sorted(combined_importance.items(), 
                                                       key=lambda x: x[1], reverse=True)):
            # Apply normalization
            final_importance = importance * normalization_factor
            
            # Get original correlation for relationship determination
            original_correlation = correlation_matrix[target_column].get(factor, 0.0)
            if pd.isna(original_correlation):
                original_correlation = 0.0
            
            factor_importance[factor] = {
                'correlation': float(original_correlation),
                'importance_percentage': float(final_importance),
                'rank': i + 1,
                'absolute_correlation': float(abs(original_correlation)),
                'relationship': 'positive' if original_correlation > 0 else 'negative' if original_correlation < 0 else 'neutral',
                'variance_contribution': float(variance_importance.get(factor, 1.0)),
                'contribution_score': float(contribution_importance.get(factor, 50.0)),
                'synchronized_with_stress_level': True
            }
        
        return factor_importance
        
    except Exception as e:
        print(f"Error calculating enhanced factor importance: {e}")
        # Fallback to safe calculation
        return safe_factor_importance_calculation(
            df[factor_columns + [target_column]].corr() if factor_columns else pd.DataFrame(), 
            target_column
        )

def safe_factor_importance_calculation(correlation_matrix: pd.DataFrame, target_column: str = 'stress_level') -> Dict[str, Any]:
    """
    Safely calculate factor importance from correlation matrix
    
    Args:
        correlation_matrix: Pandas correlation matrix
        target_column: Target column to calculate importance against
        
    Returns:
        Dictionary with cleaned factor importance data
    """
    try:
        if target_column not in correlation_matrix.columns:
            return {}
        
        # Get correlations with stress level
        stress_correlations = correlation_matrix[target_column].abs().sort_values(ascending=False)
        
        factor_importance = {}
        for i, (feature, correlation) in enumerate(stress_correlations.items()):
            if feature != target_column:
                # Clean correlation value
                clean_correlation = correlation if not pd.isna(correlation) else 0.0
                clean_correlation = clean_correlation if not np.isinf(clean_correlation) else 0.0
                
                factor_importance[feature] = {
                    'correlation': float(clean_correlation),
                    'importance_percentage': float(abs(clean_correlation) * 100),
                    'rank': i,
                    'relationship': 'positive' if clean_correlation > 0 else 'negative'
                }
        
        return factor_importance
        
    except Exception as e:
        print(f"Error calculating factor importance: {e}")
        return {}

def dynamic_correlation_analysis(df: pd.DataFrame, target_column: str = 'stress_level') -> Dict[str, Any]:
    """
    Perform dynamic correlation analysis that varies with different datasets
    
    Args:
        df: DataFrame with factor and stress data
        target_column: Target column for correlation analysis
        
    Returns:
        Dictionary with dynamic correlation insights
    """
    try:
        # Get factor columns
        factor_columns = [col for col in df.columns if col != target_column and 
                         col not in ['employee_id', 'department']]
        
        if not factor_columns:
            return {}
        
        # Calculate correlations
        correlation_matrix = df[factor_columns + [target_column]].corr()
        stress_correlations = correlation_matrix[target_column]
        
        # Find strongest positive and negative correlations
        positive_correlations = {k: v for k, v in stress_correlations.items() 
                               if k != target_column and v > 0}
        negative_correlations = {k: v for k, v in stress_correlations.items() 
                               if k != target_column and v < 0}
        
        strongest_positive = max(positive_correlations.items(), 
                               key=lambda x: x[1]) if positive_correlations else ('workload', 0.5)
        strongest_negative = min(negative_correlations.items(), 
                               key=lambda x: x[1]) if negative_correlations else ('work_life_balance', -0.3)
        
        # Calculate correlation strength distribution
        correlation_strengths = [abs(v) for k, v in stress_correlations.items() if k != target_column]
        avg_correlation_strength = np.mean(correlation_strengths) if correlation_strengths else 0.0
        max_correlation_strength = max(correlation_strengths) if correlation_strengths else 0.0
        
        # Determine correlation pattern
        if avg_correlation_strength > 0.6:
            pattern = "strong_correlations"
        elif avg_correlation_strength > 0.3:
            pattern = "moderate_correlations"
        else:
            pattern = "weak_correlations"
        
        return {
            'strongest_positive_factor': {
                'name': strongest_positive[0],
                'correlation': float(strongest_positive[1]),
                'importance': float(abs(strongest_positive[1]) * 100)
            },
            'strongest_negative_factor': {
                'name': strongest_negative[0], 
                'correlation': float(strongest_negative[1]),
                'importance': float(abs(strongest_negative[1]) * 100)
            },
            'overall_correlation_strength': float(avg_correlation_strength),
            'max_correlation_strength': float(max_correlation_strength),
            'correlation_pattern': pattern,
            'correlation_distribution': {
                'mean': float(avg_correlation_strength),
                'std': float(np.std(correlation_strengths)) if correlation_strengths else 0.0,
                'range': float(max_correlation_strength - min(correlation_strengths)) if correlation_strengths else 0.0
            }
        }
        
    except Exception as e:
        print(f"Error in dynamic correlation analysis: {e}")
        return {}

def safe_department_breakdown(df: pd.DataFrame, stress_column: str = 'stress_level') -> Dict[str, Any]:
    """
    Safely calculate department breakdown statistics
    
    Args:
        df: DataFrame with department and stress data
        stress_column: Name of the stress level column
        
    Returns:
        Dictionary with cleaned department statistics
    """
    try:
        department_stats = df.groupby('department')[stress_column].agg([
            'mean', 'count', 'std', 'min', 'max'
        ]).round(2)
        
        department_breakdown = {}
        for dept in department_stats.index:
            dept_data = department_stats.loc[dept]
            
            # Clean all department statistics
            avg_stress = float(dept_data['mean']) if not pd.isna(dept_data['mean']) else 0.0
            stress_std = float(dept_data['std']) if not pd.isna(dept_data['std']) else 0.0
            min_stress = float(dept_data['min']) if not pd.isna(dept_data['min']) else 0.0
            max_stress = float(dept_data['max']) if not pd.isna(dept_data['max']) else 0.0
            
            department_breakdown[dept] = {
                'average_stress': avg_stress,
                'employee_count': int(dept_data['count']),
                'stress_std': stress_std,
                'min_stress': min_stress,
                'max_stress': max_stress,
                'risk_level': 'High' if avg_stress > 70 else 'Medium' if avg_stress > 40 else 'Low'
            }
        
        return department_breakdown
        
    except Exception as e:
        print(f"Error calculating department breakdown: {e}")
        return {}

def create_safe_json_response(data: Dict[str, Any]) -> str:
    """
    Create a JSON response with NaN values properly handled
    
    Args:
        data: Dictionary to serialize to JSON
        
    Returns:
        JSON string with NaN values cleaned
    """
    try:
        cleaned_data = clean_nan_values(data)
        return json.dumps(cleaned_data, indent=2)
    except Exception as e:
        print(f"Error creating safe JSON response: {e}")
        return json.dumps({'error': 'Failed to serialize response'})

def validate_numeric_data(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
    """
    Validate numeric data in DataFrame for potential issues
    
    Args:
        df: DataFrame to validate
        numeric_columns: List of numeric columns to check
        
    Returns:
        Validation report
    """
    report = {
        'valid': True,
        'issues': [],
        'warnings': []
    }
    
    for col in numeric_columns:
        if col not in df.columns:
            report['issues'].append(f"Missing column: {col}")
            report['valid'] = False
            continue
            
        # Check for constant values (will produce NaN correlations)
        if df[col].nunique() <= 1:
            report['warnings'].append(f"Column {col} has constant values (may cause NaN correlations)")
        
        # Check for all NaN values
        if df[col].isna().all():
            report['issues'].append(f"Column {col} contains only NaN values")
            report['valid'] = False
        
        # Check for infinite values
        if np.isinf(df[col]).any():
            report['warnings'].append(f"Column {col} contains infinite values")
    
    return report 