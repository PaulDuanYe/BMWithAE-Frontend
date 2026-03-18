"""
BMWithAE Flask Backend API
Connects frontend to code_v_0_1 modules
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import sys
import uuid
from werkzeug.utils import secure_filename
import threading
import time
from datetime import datetime

# Import backend configuration
import backend_config

# Import local backend modules (copied from code_v_0_1, no external dependencies)
from module_load import DataLoader
from eval import Evaluator
from module_BM import BiasMitigation as BM
from module_AE import AccuracyEnhancement as AE
from module_transform import Transform
from module_transform import calculate_x_train_y_train_nmi_dict
import core_config

app = Flask(__name__)
CORS(app)

# Create necessary directories
os.makedirs(backend_config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(backend_config.RESULTS_FOLDER, exist_ok=True)
os.makedirs(backend_config.LOGS_FOLDER, exist_ok=True)

# Global storage
datasets = {}  # dataset_id -> data dict
jobs = {}      # job_id -> job info
current_config = {}  # Current configuration
dataset_models = {}  # dataset_id -> trained model (cache for subgroup analysis)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in backend_config.ALLOWED_EXTENSIONS

def load_file_to_dataframe(filepath):
    """Load CSV or Excel file to DataFrame"""
    # Ensure absolute path
    filepath = os.path.abspath(filepath)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    ext = filepath.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        return pd.read_csv(filepath)
    elif ext in ['xlsx', 'xls']:
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def convert_to_serializable(obj):
    """Convert various data types to Python native types for JSON serialization."""
    if 'pandas' in str(type(obj)) and hasattr(obj, 'to_dict'):
        return convert_to_serializable(obj.to_dict())
    elif hasattr(obj, 'tolist'):
        return convert_to_serializable(obj.tolist())
    elif isinstance(obj, dict):
        # Convert all keys to strings to avoid mixed-type key issues in JSON
        return {
            str(k): convert_to_serializable(v) 
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.number):
        return float(obj) if np.issubdtype(obj.dtype, np.floating) else int(obj)
    else:
        return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj

def calculate_overall_fairness_score(metrics):
    """
    Calculate Overall Fairness Score from individual fairness metrics.
    
    ⚠️ 重要：Fairness Metrics 是差异值（越小越公平）
    但我们返回的 Overall Fairness Score 应该是：越高越好
    
    转换方式：Overall_Fairness_Score = 1 - average(all fairness metrics)
    - 当所有 metrics = 0（完美公平）→ Score = 1.0（满分）
    - 当 metrics 增大（不公平）→ Score 减小
    
    Replicates the logic from code_v_0_1/eval.py to ensure consistency.
    
    Args:
        metrics (dict): Dictionary containing fairness metrics (lower is better)
    
    Returns:
        float: Overall fairness score (higher is better, 1.0 is perfect fairness)
    """
    fairness_metric_names = [
        'BNC', 'BPC', 'CUAE', 'EOpp', 'EO', 
        'FDRP', 'FORP', 'FNRB', 'FPRB', 
        'NPVP', 'OAE', 'PPVP', 'SP'
    ]
    
    fairness_values = []
    
    for metric_name in fairness_metric_names:
        if metric_name in metrics:
            metric_value = metrics[metric_name]
            
            # Handle different metric formats
            if isinstance(metric_value, dict):
                # Nested dict (e.g., {'SEX': 0.001, 'MARRIAGE': 0.002})
                for v in metric_value.values():
                    if isinstance(v, (int, float, np.number)):
                        fairness_values.append(float(v))
            elif isinstance(metric_value, (int, float, np.number)):
                # Direct value
                fairness_values.append(float(metric_value))
    
    # Calculate mean of all fairness difference values
    if fairness_values:
        avg_difference = np.mean(fairness_values)
        
        # 转换为公平性得分：越高越好
        # 使用 1 - avg_difference，确保范围在 [0, 1]
        # 如果 avg_difference > 1，则 score 可能为负，取 max(0, ...)
        overall_score = max(0.0, 1.0 - avg_difference)
    else:
        overall_score = 1.0  # 没有metrics时，假设完美公平
    
    return float(overall_score)

def save_experiment_log(job_id, job, config_snapshot):
    """
    Save complete experiment log to JSON file for reproducibility.
    
    Args:
        job_id (str): Unique job identifier
        job (dict): Job data containing all experiment information
        config_snapshot (dict): Configuration parameters used in the experiment
    
    Returns:
        str: Path to the saved log file
    """
    try:
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log filename
        dataset_name = job.get('dataset_name', 'unknown')
        log_filename = f"experiment_{dataset_name}_{timestamp}_{job_id[:8]}.json"
        log_filepath = os.path.join(backend_config.LOGS_FOLDER, log_filename)
        
        # Prepare log data
        log_data = {
            "experiment_info": {
                "job_id": job_id,
                "timestamp": timestamp,
                "dataset_name": dataset_name,
                "dataset_shape": {
                    "rows": job.get('X').shape[0] if hasattr(job.get('X'), 'shape') else None,
                    "columns": job.get('X').shape[1] if hasattr(job.get('X'), 'shape') else None
                },
                "target_column": job.get('target'),
                "protected_columns": job.get('protected'),
                "duration_seconds": (time.time() - job.get('start_time', time.time())) if 'start_time' in job else None
            },
            "configuration": convert_to_serializable(config_snapshot),
            "initial_state": {
                "metrics": convert_to_serializable(job.get('init_metrics', {})),
                "epsilon": convert_to_serializable(job.get('init_epsilon', {})),
                "epsilon_threshold": job.get('epsilon_threshold'),
                "accuracy_threshold": job.get('acc_threshold')
            },
            "iterations": [],
            "final_state": {
                "terminated": job.get('terminated', False),
                "termination_reason": job.get('termination_reason', None),
                "total_iterations": job.get('current_iteration', 0),
                "final_metrics": convert_to_serializable(job.get('final_metrics', {}))
            }
        }
        
        # Add iteration history
        for iter_data in job.get('history', []):
            iteration_log = {
                "iteration": iter_data.get('iteration'),
                "metrics": convert_to_serializable(iter_data.get('metrics', {})),
                "epsilon": convert_to_serializable(iter_data.get('epsilon', {})),
                "selected_attribute": iter_data.get('selected_attribute'),
                "selected_label_O": iter_data.get('selected_label_O'),
                "current_max_epsilon": iter_data.get('current_max_epsilon'),
                "changed_dict": convert_to_serializable(iter_data.get('changed_dict', {}))
            }
            log_data["iterations"].append(iteration_log)
        
        # Write to JSON file
        with open(log_filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Experiment log saved to: {log_filepath}")
        return log_filepath
    
    except Exception as e:
        print(f"[ERROR] Failed to save experiment log: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================
# API: Data Management
# ============================================

@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    """Upload custom dataset"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': 'Invalid file format'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        dataset_id = str(uuid.uuid4())
        filepath = os.path.join(backend_config.UPLOAD_FOLDER, f"{dataset_id}_{filename}")
        file.save(filepath)
        
        # Load and analyze data
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        
        # Get parameters from request
        target_column = request.form.get('target_column')
        protected_columns = request.form.getlist('protected_columns[]')
        
        if not target_column or target_column not in df.columns:
            return jsonify({'status': 'error', 'message': 'Invalid target column'}), 400
        
        # Clear cached model for this dataset if it exists
        if dataset_id in dataset_models:
            del dataset_models[dataset_id]
        
        # Store dataset info
        datasets[dataset_id] = {
            'filepath': filepath,
            'filename': filename,
            'target_column': target_column,
            'protected_columns': protected_columns,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'preview': df.head(10).to_dict('records')
        }
        
        return jsonify({
            'status': 'success',
            'data': {
                'dataset_id': dataset_id,
                'filename': filename,
                'rows': df.shape[0],
                'columns': df.shape[1],
                'column_names': df.columns.tolist(),
                'preview': datasets[dataset_id]['preview']
            }
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/data/demo', methods=['POST'])
def load_demo():
    """Load demo dataset"""
    try:
        data = request.json
        dataset_name = data.get('dataset_name', 'credit')
        
        # Get demo dataset configuration
        if dataset_name not in backend_config.DEMO_DATASETS:
            return jsonify({'status': 'error', 'message': f'Demo dataset "{dataset_name}" not found'}), 404
        
        demo_config = backend_config.DEMO_DATASETS[dataset_name]
        filepath = demo_config['path']
        target_column = demo_config['target']
        protected_columns = demo_config['protected']
        
        # Load the dataframe to verify it works
        df = load_file_to_dataframe(filepath)
        
        dataset_id = f"demo_{dataset_name}"
        
        # Clear cached model for this dataset if it exists
        if dataset_id in dataset_models:
            del dataset_models[dataset_id]
        
        # Store dataset info with original relative path
        # (will be converted to absolute path when needed)
        datasets[dataset_id] = {
            'filepath': filepath,  # Store relative path from config
            'filename': f"{dataset_name}.xlsx",
            'target_column': target_column,
            'protected_columns': protected_columns,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'preview': df.head(10).to_dict('records')
        }
        
        return jsonify({
            'status': 'success',
            'data': {
                'dataset_id': dataset_id,
                'filename': f"{dataset_name}.xlsx",
                'rows': df.shape[0],
                'columns': df.shape[1],
                'column_names': df.columns.tolist(),
                'preview': datasets[dataset_id]['preview']
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

def calculate_bias_metrics(df, target_col, protected_col):
    """Calculate basic bias metrics for a given protected attribute"""
    try:
        # Ensure columns exist
        if target_col not in df.columns or protected_col not in df.columns:
            return None
        
        # Get unique groups in protected attribute
        groups = df[protected_col].unique()
        if len(groups) < 2:
            return None
        
        # Calculate Statistical Parity (Demographic Parity)
        # Difference in positive prediction rates between groups
        positive_rates = {}
        for group in groups:
            group_data = df[df[protected_col] == group]
            if len(group_data) > 0:
                positive_rate = (group_data[target_col] == 1).sum() / len(group_data)
                positive_rates[str(group)] = positive_rate
        
        if len(positive_rates) >= 2:
            rates = list(positive_rates.values())
            sp_diff = max(rates) - min(rates)
        else:
            sp_diff = 0
        
        # Calculate Disparate Impact
        # Ratio of positive rates (min/max)
        if len(positive_rates) >= 2:
            disparate_impact = min(rates) / max(rates) if max(rates) > 0 else 1
        else:
            disparate_impact = 1
        
        # Mock Equal Opportunity and Equalized Odds for now
        # (Would need predicted labels for real calculation)
        eo_diff = sp_diff * 0.8  # Approximate
        eodds_diff = sp_diff * 0.9  # Approximate
        
        return {
            'statistical_parity': float(sp_diff),
            'equal_opportunity': float(eo_diff),
            'equalized_odds': float(eodds_diff),
            'disparate_impact': float(disparate_impact),
            'group_positive_rates': positive_rates
        }
    except Exception as e:
        print(f"[ERROR] Failed to calculate bias metrics: {e}")
        return None

def calculate_combined_bias_metrics(df, target_col, protected_cols):
    """
    Calculate aggregated bias metrics for multiple protected attributes.
    Returns the average (or max) of metrics across all protected attributes.
    """
    try:
        if not protected_cols:
            return None
        
        all_metrics = []
        
        # Calculate metrics for each protected attribute
        for protected_col in protected_cols:
            metrics = calculate_bias_metrics(df, target_col, protected_col)
            if metrics:
                all_metrics.append(metrics)
        
        if not all_metrics:
            return None
        
        # Aggregate metrics - use maximum (worst case) for fairness metrics
        # This represents the worst bias across all protected attributes
        combined = {
            'statistical_parity': max(m['statistical_parity'] for m in all_metrics),
            'equal_opportunity': max(m['equal_opportunity'] for m in all_metrics),
            'equalized_odds': max(m['equalized_odds'] for m in all_metrics),
            'disparate_impact': min(m['disparate_impact'] for m in all_metrics),  # Min because closer to 0 is worse
            'protected_attributes': protected_cols,
            'individual_metrics': all_metrics
        }
        
        print(f"[DEBUG] Combined bias metrics for {protected_cols}: {combined}")
        
        return combined
    except Exception as e:
        print(f"[ERROR] Failed to calculate combined bias metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/api/data/<dataset_id>/info', methods=['GET'])
def get_dataset_info(dataset_id):
    """Get detailed dataset information including features and statistics"""
    try:
        if dataset_id not in datasets:
            return jsonify({'status': 'error', 'message': 'Dataset not found'}), 404
        
        dataset = datasets[dataset_id]
        filepath = dataset['filepath']
        
        # Load dataframe
        df = load_file_to_dataframe(filepath)
        
        # Get feature information
        features = []
        feature_types = {}
        
        for col in df.columns:
            # Determine feature type
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() <= 10:
                    ftype = 'categorical'
                else:
                    ftype = 'continuous'
            else:
                ftype = 'categorical'
            
            feature_types[col] = ftype
            features.append(col)
        
        # Calculate basic statistics for each feature
        feature_stats = {}
        for col in df.columns:
            stats = {
                'name': col,
                'type': feature_types[col],
                'unique_values': int(df[col].nunique()),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float(df[col].isnull().sum() / len(df) * 100)
            }
            
            if feature_types[col] == 'continuous':
                min_val = float(df[col].min()) if not pd.isna(df[col].min()) else 0
                max_val = float(df[col].max()) if not pd.isna(df[col].max()) else 1
                mean_val = float(df[col].mean()) if not pd.isna(df[col].mean()) else None
                std_val = float(df[col].std()) if not pd.isna(df[col].std()) else None
                
                # Calculate bin counts for Low/Medium/High
                range_val = max_val - min_val
                bin_edges = [
                    min_val,
                    min_val + range_val / 3,
                    min_val + 2 * range_val / 3,
                    max_val
                ]
                
                # Count samples in each bin
                low_count = int(((df[col] >= bin_edges[0]) & (df[col] < bin_edges[1])).sum())
                medium_count = int(((df[col] >= bin_edges[1]) & (df[col] < bin_edges[2])).sum())
                high_count = int(((df[col] >= bin_edges[2]) & (df[col] <= bin_edges[3])).sum())
                
                stats.update({
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'std': std_val,
                    'bin_counts': {
                        'Low': low_count,
                        'Medium': medium_count,
                        'High': high_count
                    },
                    'bin_ranges': {
                        'Low': f"({min_val:.1f}-{bin_edges[1]:.1f})",
                        'Medium': f"({bin_edges[1]:.1f}-{bin_edges[2]:.1f})",
                        'High': f"({bin_edges[2]:.1f}-{max_val:.1f})"
                    }
                })
            else:
                # Get value counts for categorical features
                value_counts = df[col].value_counts().head(10).to_dict()
                stats['value_counts'] = {str(k): int(v) for k, v in value_counts.items()}
            
            feature_stats[col] = stats
        
        # Get target column info
        target_col = dataset.get('target_column')
        target_distribution = None
        if target_col and target_col in df.columns:
            target_distribution = df[target_col].value_counts().to_dict()
            target_distribution = {str(k): int(v) for k, v in target_distribution.items()}
        
        # Get protected columns info
        protected_cols = dataset.get('protected_columns', [])
        
        return jsonify({
            'status': 'success',
            'data': {
                'dataset_id': dataset_id,
                'filename': dataset['filename'],
                'rows': df.shape[0],
                'columns': df.shape[1],
                'features': features,
                'feature_types': feature_types,
                'feature_stats': feature_stats,
                'target_column': target_col,
                'target_distribution': target_distribution,
                'protected_columns': protected_cols
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/data/<dataset_id>/bias-metrics', methods=['POST'])
def get_bias_metrics(dataset_id):
    """Calculate bias metrics for protected attribute(s)"""
    try:
        if dataset_id not in datasets:
            return jsonify({'status': 'error', 'message': 'Dataset not found'}), 404
        
        data = request.json
        protected_attrs = data.get('protected_attributes', [])
        
        # Backward compatibility: support single attribute
        if not protected_attrs:
            protected_attr = data.get('protected_attribute')
            if protected_attr:
                protected_attrs = [protected_attr]
        
        if not protected_attrs:
            return jsonify({'status': 'error', 'message': 'Protected attributes not specified'}), 400
        
        dataset = datasets[dataset_id]
        filepath = dataset['filepath']
        target_col = dataset.get('target_column')
        
        if not target_col:
            return jsonify({'status': 'error', 'message': 'Target column not found'}), 400
        
        # Load dataframe
        df = load_file_to_dataframe(filepath)
        
        # Calculate combined bias metrics for multiple attributes
        bias_metrics = calculate_combined_bias_metrics(df, target_col, protected_attrs)
        
        if bias_metrics is None:
            return jsonify({'status': 'error', 'message': 'Failed to calculate bias metrics'}), 500
        
        return jsonify({
            'status': 'success',
            'data': {
                'protected_attributes': protected_attrs,
                'metrics': bias_metrics
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/data/<dataset_id>/subgroup-metrics', methods=['POST'])
def get_subgroup_metrics(dataset_id):
    """Calculate performance metrics for a specific subgroup defined by conditions"""
    try:
        if dataset_id not in datasets:
            return jsonify({'status': 'error', 'message': 'Dataset not found'}), 404
        
        data = request.json
        conditions = data.get('conditions', {})  # e.g., {'AGE': 25, 'SEX': 'M'}
        
        if not conditions:
            return jsonify({'status': 'error', 'message': 'No conditions specified'}), 400
        
        dataset = datasets[dataset_id]
        filepath = dataset['filepath']
        target_col = dataset.get('target_column')
        
        if not target_col:
            return jsonify({'status': 'error', 'message': 'Target column not found'}), 400
        
        # Load dataframe
        df = load_file_to_dataframe(filepath)
        
        # Filter dataframe based on conditions
        filtered_df = df.copy()
        for feature, value in conditions.items():
            if feature not in df.columns:
                return jsonify({'status': 'error', 'message': f'Feature {feature} not found'}), 400
            
            # Check if value is a range string like "Low (-165580.0-211117.0)"
            import re
            range_match = re.match(r'^(Low|Medium|High)\s*\(([+-]?[\d.]+)-([+-]?[\d.]+)\)$', str(value))
            
            if range_match:
                # Range condition for continuous features
                label, min_val, max_val = range_match.groups()
                min_val = float(min_val)
                max_val = float(max_val)
                filtered_df = filtered_df[(filtered_df[feature] >= min_val) & (filtered_df[feature] < max_val)]
            else:
                # Exact match for categorical features
                # Try to convert value to appropriate type
                try:
                    # Try numeric conversion
                    if pd.api.types.is_numeric_dtype(df[feature]):
                        value = float(value) if '.' in str(value) else int(value)
                except (ValueError, TypeError):
                    pass  # Keep as string
                
                filtered_df = filtered_df[filtered_df[feature] == value]
        
        # Calculate subgroup size
        subgroup_size = len(filtered_df)
        overall_size = len(df)
        
        if subgroup_size == 0:
            return jsonify({'status': 'error', 'message': 'No samples match the specified conditions'}), 400
        
        # Calculate metrics directly from true labels (no model prediction needed)
        # Since we have labeled data, we can calculate actual statistics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        # Get overall dataset statistics
        y_overall = df[target_col]
        positive_rate_overall = float(y_overall.sum() / len(y_overall))
        
        overall_metrics = {
            'positive_rate': positive_rate_overall,
            'negative_rate': float(1 - positive_rate_overall),
            'total_samples': int(len(y_overall)),
            'positive_samples': int(y_overall.sum()),
            'negative_samples': int(len(y_overall) - y_overall.sum())
        }
        
        # Calculate subgroup statistics
        y_subgroup = filtered_df[target_col]
        
        # Log subgroup label distribution for debugging
        unique_labels = y_subgroup.unique()
        label_counts = y_subgroup.value_counts().to_dict()
        print(f"[DEBUG] Subgroup labels: {unique_labels}, counts: {label_counts}")
        
        # Calculate subgroup statistics
        positive_count = int(y_subgroup.sum())
        negative_count = int(len(y_subgroup) - positive_count)
        positive_rate = float(positive_count / len(y_subgroup)) if len(y_subgroup) > 0 else 0.0
        
        subgroup_metrics = {
            'positive_rate': positive_rate,
            'negative_rate': float(1 - positive_rate),
            'positive_samples': positive_count,
            'negative_samples': negative_count,
            'total_samples': int(len(y_subgroup))
        }
        
        # Calculate relative metrics (compared to overall)
        subgroup_metrics['positive_rate_ratio'] = float(positive_rate / positive_rate_overall) if positive_rate_overall > 0 else 0.0
        
        # Add class distribution for visualization
        subgroup_metrics['class_distribution'] = {
            '0': negative_count,
            '1': positive_count
        }
        
        # For compatibility with existing frontend, also provide these metrics
        # These represent the class balance, not model performance
        if positive_count > 0 and negative_count > 0:
            # Balanced subgroup
            subgroup_metrics['accuracy'] = float(max(positive_rate, 1 - positive_rate))  # Majority class baseline
            subgroup_metrics['precision'] = positive_rate  # Proportion of positive class
            subgroup_metrics['recall'] = positive_rate  # Same as precision for true labels
            subgroup_metrics['f1'] = positive_rate  # Harmonic mean
            subgroup_metrics['fpr'] = float(1 - positive_rate) if positive_count > 0 else 0.0
        elif positive_count == 0:
            # Only negative class
            subgroup_metrics['accuracy'] = 1.0  # All negative is "correct" for majority baseline
            subgroup_metrics['precision'] = 0.0
            subgroup_metrics['recall'] = 0.0
            subgroup_metrics['f1'] = 0.0
            subgroup_metrics['fpr'] = 0.0
            subgroup_metrics['note'] = 'Subgroup contains only negative class (label=0)'
        else:
            # Only positive class
            subgroup_metrics['accuracy'] = 1.0  # All positive is "correct"
            subgroup_metrics['precision'] = 1.0
            subgroup_metrics['recall'] = 1.0
            subgroup_metrics['f1'] = 1.0
            subgroup_metrics['fpr'] = 0.0
            subgroup_metrics['note'] = 'Subgroup contains only positive class (label=1)'
        
        return jsonify({
            'status': 'success',
            'data': {
                'subgroup_size': int(subgroup_size),
                'overall_size': int(overall_size),
                'subgroup_percentage': float(subgroup_size / overall_size * 100),
                'conditions': conditions,
                'subgroup_metrics': subgroup_metrics,
                'overall_metrics': overall_metrics
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================
# API: Configuration
# ============================================

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration from core_config"""
    try:
        config_params = {}
        for param_name in dir(core_config):
            if not param_name.startswith('__') and param_name.isupper():
                config_params[param_name] = getattr(core_config, param_name)
        
        return jsonify({
            'status': 'success',
            'config': config_params
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update core configuration"""
    try:
        data = request.json
        
        print(f"[DEBUG] Updating config with: {data}")
        
        # Update core_config dynamically
        for key, value in data.items():
            if hasattr(core_config, key):
                old_value = getattr(core_config, key)
                setattr(core_config, key, value)
                current_config[key] = value
                print(f"[DEBUG] Updated {key}: {old_value} → {value}")
        
        # Force reload of modules that use config (next import will use new values)
        # Note: Due to Python's import caching, modules already loaded won't see changes
        # But new Evaluator/BM/AE instances will read current config values
        
        return jsonify({
            'status': 'success',
            'message': 'Configuration updated successfully',
            'config': current_config
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================
# API: Debiasing Process
# ============================================

@app.route('/api/debias/init', methods=['POST'])
def init_debias():
    """Initialize debiasing job"""
    # Initialize variables for config restoration
    config_snapshot = {}
    affected_modules = []
    
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        custom_config = data.get('config', {})  # Optional config override
        protected_attributes = data.get('protected_attributes', [])  # User-selected protected attributes (array)
        
        # Backward compatibility: support single attribute
        if not protected_attributes:
            protected_attribute = data.get('protected_attribute')
            if protected_attribute:
                protected_attributes = [protected_attribute]
        
        print(f"[DEBUG] Initializing debiasing for dataset: {dataset_id}")
        if protected_attributes:
            print(f"[DEBUG] Using user-selected protected attributes: {protected_attributes}")
        if custom_config:
            print(f"[DEBUG] Using custom config: {custom_config}")
        
        if dataset_id not in datasets:
            print(f"[ERROR] Dataset not found: {dataset_id}")
            print(f"[DEBUG] Available datasets: {list(datasets.keys())}")
            return jsonify({'status': 'error', 'message': 'Dataset not found'}), 404
        else:
            print(f"[DEBUG] Dataset found: {dataset_id}")
            print(f"[DEBUG] Available datasets: {list(datasets.keys())}")
        
        # Apply custom config if provided (for this job only)
        
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(core_config, key):
                    config_snapshot[key] = getattr(core_config, key)
                    setattr(core_config, key, value)
                    print(f"[DEBUG] Applying config: {key} = {value}")
            
            # Also update the references in imported modules
            # This handles the "from config import XXX" issue
            import sys
            for module_name in ['eval', 'module_BM', 'module_AE', 'module_transform', 'module_load']:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    affected_modules.append(module_name)
                    for key, value in custom_config.items():
                        if hasattr(module, key):
                            setattr(module, key, value)
                            print(f"[DEBUG] Updated {module_name}.{key} = {value}")
        
        # Load data using DataLoader
        dataset_info = datasets[dataset_id]
        
        # Get absolute path for DataLoader
        filepath = dataset_info['filepath']
        # Normalize and get absolute path
        filepath = os.path.abspath(filepath)
        
        print(f"[DEBUG] Loading data from: {filepath}")
        print(f"[DEBUG] File exists: {os.path.exists(filepath)}")
        print(f"[DEBUG] Target column: {dataset_info['target_column']}")
        print(f"[DEBUG] Protected columns: {dataset_info['protected_columns']}")
        
        # Get project root directory
        project_root = backend_config.PROJECT_ROOT
        original_cwd = os.getcwd()
        
        # Save original config values (modify dict in-place so references work)
        original_dataset = core_config.DATASET.copy()
        
        try:
            # Change to project root so DataLoader can find files
            os.chdir(project_root)
            print(f"[DEBUG] Changed working directory to: {os.getcwd()}")
            
            # Modify DATASET dict in-place (don't replace it)
            core_config.DATASET.clear()
            core_config.DATASET['path'] = filepath
            core_config.DATASET['target'] = dataset_info['target_column']
            
            # Use user-selected protected attributes if provided, otherwise use default
            if protected_attributes:
                core_config.DATASET['protected'] = protected_attributes
                print(f"[DEBUG] Using user-selected protected attributes: {protected_attributes}")
            else:
                core_config.DATASET['protected'] = dataset_info['protected_columns']
                print(f"[DEBUG] Using default protected columns: {dataset_info['protected_columns']}")
            
            print(f"[DEBUG] core_config.DATASET updated: {core_config.DATASET}")
            
            loader = DataLoader()
            X, Y, O, categorical, numerical = loader.load_data()
            print(f"[DEBUG] Data loaded successfully: X={X.shape}, Y={Y.shape}, O={O.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Restore original working directory and config
            os.chdir(original_cwd)
            core_config.DATASET.clear()
            core_config.DATASET.update(original_dataset)
        
        # Initialize evaluator
        evaluator = Evaluator(
            label_O=O.columns.tolist(),
            label_Y=Y.name,
            cate_attrs=categorical,
            num_attrs=numerical
        )
        
        # Calculate initial epsilon
        init_epsilon = evaluator.calculate_epsilon(X, O, categorical, numerical)
        
        # Initial evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test, O_train, O_test = train_test_split(
            X, Y, O, test_size=core_config.PARAMS_TEST_SIZE, random_state=core_config.SEED
        )
        
        # 调试：检查训练和测试数据的标签分布
        import sys
        y_train_dist = Y_train.value_counts().to_dict()
        y_test_dist = Y_test.value_counts().to_dict()
        print(f"[DEBUG] Y_train distribution: {y_train_dist}", flush=True, file=sys.stderr)
        print(f"[DEBUG] Y_test distribution: {y_test_dist}", flush=True, file=sys.stderr)
        print(f"[DEBUG] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}", flush=True, file=sys.stderr)
        
        # Initialize transformer (will handle scaler if needed)
        transformer = Transform()
        
        # Apply transform (even with empty changed_dict, scaler will be applied if needed)
        # For initial evaluation, changed_dict is empty but scaler should be fitted
        initial_changed_dict = {}
        transformed_X_train = transformer.transform_data(
            X_train, initial_changed_dict, numerical, categorical)
        transformed_X_test = transformer.transform_data(
            X_test, initial_changed_dict, numerical, categorical)
        
        init_metrics = evaluator.evaluate(
            transformed_X_train, Y_train, O_train, transformed_X_test, Y_test, O_test
        )
        
        # 调试：检查初始metrics中的fairness指标
        import sys
        print(f"[DEBUG] Initial metrics keys: {list(init_metrics.keys())}", flush=True, file=sys.stderr)
        for metric_name in ['EO', 'SP', 'EOpp', 'BNC', 'BPC']:
            if metric_name in init_metrics:
                print(f"[DEBUG] {metric_name} = {init_metrics[metric_name]}", flush=True, file=sys.stderr)
        
        # 添加综合公平性分数
        init_metrics['Overall_Fairness'] = calculate_overall_fairness_score(init_metrics)
        
        # Calculate epsilon and accuracy thresholds (like code_v_0_1)
        # 计算初始平均 epsilon（所有 epsilon 值的平均）
        all_epsilon_values = []
        for o_label in init_epsilon.keys():
            for attr in init_epsilon[o_label].index:
                all_epsilon_values.append(init_epsilon[o_label][attr])
        init_avg_epsilon = np.mean(all_epsilon_values)
        
        # 阈值基于初始平均 epsilon（用户输入 0.9 表示降到初始平均值的 90%）
        epsilon_threshold = init_avg_epsilon * core_config.PARAMS_MAIN_THRESHOLD_EPSILON
        
        init_acc = init_metrics.get('ACC', 0)
        acc_threshold = init_acc * (1 + core_config.PARAMS_MAIN_THRESHOLD_ACCURACY)
        
        print(f"[DEBUG] Initial avg epsilon: {init_avg_epsilon}")
        print(f"[DEBUG] Epsilon threshold (init_avg × {core_config.PARAMS_MAIN_THRESHOLD_EPSILON}): {epsilon_threshold}")
        print(f"[DEBUG] Accuracy threshold: {acc_threshold}")
        
        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            'job_id': job_id,
            'dataset_id': dataset_id,
            'dataset_info': dataset_info,
            'dataset_name': dataset_info.get('filename', 'unknown'),
            'target': Y.name,
            'protected': list(O.columns) if hasattr(O, 'columns') else [],
            'state': 'initialized',
            'current_iteration': 0,
            'max_iteration': core_config.PARAMS_MAIN_MAX_ITERATION,
            'epsilon_threshold': epsilon_threshold,
            'epsilon_threshold_ratio': core_config.PARAMS_MAIN_THRESHOLD_EPSILON,  # 保存用户输入的比例
            'acc_threshold': acc_threshold,
            'init_avg_epsilon': init_avg_epsilon,  # 保存初始平均 epsilon，用于计算百分比
            'X': X,
            'Y': Y,
            'O': O,
            'categorical': categorical,
            'numerical': numerical,
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test,
            'O_train': O_train,
            'O_test': O_test,
            'evaluator': evaluator,
            'transformer': transformer,  # Use the already initialized transformer with fitted scaler
            'changed_dict': initial_changed_dict,  # May contain scaler params now
            'transformed_df': X.copy(),
            'history': [],
            'init_metrics': convert_to_serializable(init_metrics),
            'init_epsilon': convert_to_serializable(init_epsilon),
            'terminated': False,
            'termination_reason': None,
            'config_snapshot': config_snapshot.copy(),  # Save config for logging
            'start_time': time.time(),  # Record start time
            'nmi_org': calculate_x_train_y_train_nmi_dict(X_train, Y_train) # original NMI for reference
        }
        
        # Initialize BM and AE
        jobs[job_id]['bm'] = BM(
            evaluator=evaluator,
            transformer=jobs[job_id]['transformer'],
            label_O=O.columns.tolist(),
            cate_attrs=categorical,
            num_attrs=numerical
        )
        
        jobs[job_id]['ae'] = AE(
            evaluator=evaluator,
            transformer=jobs[job_id]['transformer'],
            label_Y=Y.name,
            cate_attrs=categorical,
            num_attrs=numerical
        )
        
        return jsonify({
            'status': 'success',
            'data': {
                'job_id': job_id,
                'init_metrics': jobs[job_id]['init_metrics'],
                'init_epsilon': jobs[job_id]['init_epsilon']
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
    finally:
        # Restore original config if it was modified
        if config_snapshot:
            print(f"[DEBUG] Restoring original config...")
            for key, value in config_snapshot.items():
                setattr(core_config, key, value)
            
            # Restore module references
            import sys
            for module_name in affected_modules:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    for key, value in config_snapshot.items():
                        if hasattr(module, key):
                            setattr(module, key, value)

@app.route("/debug/datasets")
def debug_datasets():
    return datasets

"""     fetch("http://localhost:5001/debug/datasets")
  .then(r => r.json())
  .then(console.log) """

@app.route('/api/debias/<job_id>/step', methods=['POST'])
def step_iteration(job_id):
    """Execute one complete iteration (BM + AE + evaluate)"""
    try:
        if job_id not in jobs:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404
        
        job = jobs[job_id]
        
        if job['terminated']:
            return jsonify({
                'status': 'success',
                'message': f'Job already terminated: {job["termination_reason"]}',
                'data': {'terminated': True, 'reason': job['termination_reason']}
            })
        
        if job['current_iteration'] >= job['max_iteration']:
            job['terminated'] = True
            job['termination_reason'] = 'Max iteration reached'
            return jsonify({
                'status': 'success',
                'message': 'Max iteration reached',
                'data': {'terminated': True, 'reason': 'Max iteration reached'}
            })
        
        job['current_iteration'] += 1
        iter_num = job['current_iteration']
        print(f"[DEBUG] Iteration {iter_num} started")
        
        iter_data = {'iteration': iter_num}
        
        # Step 1: Bias Mitigation (if enabled)
        if core_config.USE_BIAS_MITIGATION:
            print(f"[DEBUG] Performing bias mitigation...")
            current_epsilon = job['evaluator'].calculate_epsilon(
                job['transformed_df'], job['O'], job['categorical'], job['numerical']
            )
            
            selected_label_O, selected_attribute = job['bm']._find_max_epsilon_attribute(current_epsilon)
            print(f"[DEBUG] Selected: {selected_label_O}, {selected_attribute}")
            
            iter_data['selected_label_O'] = selected_label_O
            iter_data['selected_attribute'] = selected_attribute
            
            job['transformed_df'], job['changed_dict'] = job['bm'].mitigate(
                job['X'], job['Y'], job['O'], job['nmi_org'], job['changed_dict']
            )
            
            iter_data['epsilon'] = convert_to_serializable(current_epsilon)
        
        # Step 2: Accuracy Enhancement (if enabled)
        if core_config.USE_ACCURACY_ENHANCEMENT:
            print(f"[DEBUG] Performing accuracy enhancement...")
            old_changed_dict = job['changed_dict'].copy()
            # AE returns transformed data, don't discard it!
            _, job['changed_dict'] = job['ae'].enhance(
                job['X_train'], job['Y_train'], job['changed_dict']
            )
            print(f"[DEBUG] AE changed_dict before: {old_changed_dict}")
            print(f"[DEBUG] AE changed_dict after: {job['changed_dict']}")
            print(f"[DEBUG] AE made changes: {old_changed_dict != job['changed_dict']}")
            
            # After AE, we need to re-transform all data with the updated changed_dict
            print(f"[DEBUG] Re-transforming data with updated changed_dict from AE...")
        
        # Transform data (either from BM only, or BM+AE)
        # This ensures we always use the latest changed_dict
        job['transformed_df'] = job['transformer'].transform_data(
            job['X'], job['changed_dict'], job['numerical'], job['categorical']
        )
        
        transformed_X_train = job['transformer'].transform_data(
            job['X_train'], job['changed_dict'], job['numerical'], job['categorical']
        )
        transformed_X_test = job['transformer'].transform_data(
            job['X_test'], job['changed_dict'], job['numerical'], job['categorical']
        )
        
        # Evaluate
        metrics = job['evaluator'].evaluate(
            transformed_X_train, job['Y_train'], job['O_train'],
            transformed_X_test, job['Y_test'], job['O_test']
        )
        
        # 添加综合公平性分数
        metrics['Overall_Fairness'] = calculate_overall_fairness_score(metrics)
        
        # Calculate current epsilon
        current_epsilon = job['evaluator'].calculate_epsilon(
            job['transformed_df'], job['O'], job['categorical'], job['numerical']
        )
        selected_label_O, selected_attribute = job['bm']._find_max_epsilon_attribute(current_epsilon)
        current_max_epsilon = current_epsilon[selected_label_O][selected_attribute]
        
        # 计算当前平均 epsilon（用于终止条件判断）
        current_epsilon_values = []
        for o_label in current_epsilon.keys():
            for attr in current_epsilon[o_label].index:
                current_epsilon_values.append(current_epsilon[o_label][attr])
        current_avg_epsilon = np.mean(current_epsilon_values)
        
        current_acc = metrics.get('ACC', 0)
        
        print(f"[DEBUG] ========== Iteration {job['current_iteration']} Results ==========")
        print(f"[DEBUG] Current max epsilon: {current_max_epsilon:.10f}")
        print(f"[DEBUG] Current accuracy: {current_acc:.10f}")
        print(f"[DEBUG] Overall Fairness: {metrics.get('Overall_Fairness', 0):.10f}")
        print(f"[DEBUG] F1 Score: {metrics.get('F1', 0):.10f}")
        if job['history']:
            prev_acc = job['history'][-1]['metrics'].get('ACC', 0) if 'metrics' in job['history'][-1] else job['init_metrics'].get('ACC', 0)
            acc_change = current_acc - prev_acc
            print(f"[DEBUG] Accuracy change from previous iteration: {acc_change:+.10f}")
        print(f"[DEBUG] ==================================================")
        
        iter_data['metrics'] = convert_to_serializable(metrics)
        iter_data['changed_dict'] = convert_to_serializable(job['changed_dict'])
        iter_data['current_max_epsilon'] = current_max_epsilon
        iter_data['current_avg_epsilon'] = current_avg_epsilon  # 保存平均值
        
        job['history'].append(iter_data)
        
        # Check termination conditions
        terminated = False
        termination_reason = None
        
        print(f"[DEBUG] Checking termination conditions:")
        print(f"  - USE_BIAS_MITIGATION: {core_config.USE_BIAS_MITIGATION}")
        print(f"  - current_max_epsilon: {current_max_epsilon}")
        print(f"  - epsilon_threshold: {job['epsilon_threshold']}")
        print(f"  - Condition (current_max <= threshold): {current_max_epsilon <= job['epsilon_threshold']}")
        
        if core_config.USE_BIAS_MITIGATION and current_max_epsilon <= job['epsilon_threshold']:
            terminated = True
            termination_reason = f"Epsilon threshold reached: {current_max_epsilon:.6f} <= {job['epsilon_threshold']:.6f}"
        elif core_config.USE_ACCURACY_ENHANCEMENT and current_acc >= job['acc_threshold']:
            terminated = True
            termination_reason = f"Accuracy threshold reached: {current_acc:.6f} >= {job['acc_threshold']:.6f}"
        
        if terminated:
            job['terminated'] = True
            job['termination_reason'] = termination_reason
            print(f"[DEBUG] ✅ TERMINATED: {termination_reason}")
        else:
            print(f"[DEBUG] ❌ Not terminated, continuing...")
        
        return jsonify({
            'status': 'success',
            'data': {
                **iter_data,
                'terminated': terminated,
                'termination_reason': termination_reason
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

def _run_full_process_thread(job_id):
    """Background thread for running full debiasing process"""
    job = jobs[job_id]
    
    try:
        job['state'] = 'running'
        job['progress'] = 0
        
        #nmi_org = calculate_x_train_y_train_nmi_dict(job['X_train'], job['Y_train'])
        
        # Loop until termination condition or max iteration
        while job['current_iteration'] < job['max_iteration'] and not job['terminated']:
            try:
                job['current_iteration'] += 1
                iter_num = job['current_iteration']
                print(f"[DEBUG] Iteration {iter_num} started")
                
                iter_data = {'iteration': iter_num}
                
                # Step 1: Bias Mitigation (if enabled)
                if core_config.USE_BIAS_MITIGATION:
                    current_epsilon = job['evaluator'].calculate_epsilon(
                        job['transformed_df'], job['O'], job['categorical'], job['numerical']
                    )
                    
                    selected_label_O, selected_attribute = job['bm']._find_max_epsilon_attribute(current_epsilon)
                    
                    transformed_df, changed_dict = job['bm'].mitigate(
                        job['X'], job['Y'], job['O'], job['nmi_org'], job['changed_dict']
                    )
                    
                    job['transformed_df'] = transformed_df
                    job['changed_dict'] = changed_dict
                    
                    iter_data['selected_label_O'] = selected_label_O
                    iter_data['selected_attribute'] = selected_attribute
                    iter_data['epsilon'] = convert_to_serializable(current_epsilon)
                
                # Step 2: Accuracy Enhancement
                if core_config.USE_ACCURACY_ENHANCEMENT:
                    print(f"[DEBUG] AE enabled, executing enhancement...")
                    old_changed_dict = job['changed_dict'].copy()
                    _, changed_dict = job['ae'].enhance(
                        job['X_train'], job['Y_train'], job['changed_dict']
                    )
                    job['changed_dict'] = changed_dict
                    print(f"[DEBUG] AE changed_dict keys before: {list(old_changed_dict.keys())}")
                    print(f"[DEBUG] AE changed_dict keys after: {list(changed_dict.keys())}")
                    print(f"[DEBUG] AE made changes: {old_changed_dict != changed_dict}")
                    
                    # After AE, we need to re-transform all data with the updated changed_dict
                    print(f"[DEBUG] Re-transforming data with updated changed_dict from AE...")
                
                # Transform and evaluate (using the latest changed_dict from BM and/or AE)
                job['transformed_df'] = job['transformer'].transform_data(
                    job['X'], job['changed_dict'], job['numerical'], job['categorical']
                )
                
                transformed_X_train = job['transformer'].transform_data(
                    job['X_train'], job['changed_dict'], job['numerical'], job['categorical']
                )
                transformed_X_test = job['transformer'].transform_data(
                    job['X_test'], job['changed_dict'], job['numerical'], job['categorical']
                )
                
                metrics = job['evaluator'].evaluate(
                    transformed_X_train, job['Y_train'], job['O_train'],
                    transformed_X_test, job['Y_test'], job['O_test']
                )
                
                # 添加综合公平性分数
                metrics['Overall_Fairness'] = calculate_overall_fairness_score(metrics)
                
                # Calculate current epsilon and accuracy for termination check
                current_epsilon = job['evaluator'].calculate_epsilon(
                    job['transformed_df'], job['O'], job['categorical'], job['numerical']
                )
                selected_label_O, selected_attribute = job['bm']._find_max_epsilon_attribute(current_epsilon)
                current_max_epsilon = current_epsilon[selected_label_O][selected_attribute]
                
                # 计算当前平均 epsilon（用于终止条件判断）
                current_epsilon_values = []
                for o_label in current_epsilon.keys():
                    for attr in current_epsilon[o_label].index:
                        current_epsilon_values.append(current_epsilon[o_label][attr])
                current_avg_epsilon = np.mean(current_epsilon_values)
                
                current_acc = metrics.get('ACC', 0)
                
                print(f"[DEBUG] ========== Iteration {iter_num} Results ==========")
                print(f"[DEBUG] Current max epsilon: {current_max_epsilon:.10f}")
                print(f"[DEBUG] Current accuracy: {current_acc:.10f}")
                print(f"[DEBUG] Overall Fairness: {metrics.get('Overall_Fairness', 0):.10f}")
                print(f"[DEBUG] F1 Score: {metrics.get('F1', 0):.10f}")
                if job['history']:
                    prev_metrics = job['history'][-1].get('metrics', job['init_metrics'])
                    prev_acc = prev_metrics.get('ACC', 0)
                    acc_change = current_acc - prev_acc
                    print(f"[DEBUG] Accuracy change from previous iteration: {acc_change:+.10f}")
                else:
                    init_acc = job['init_metrics'].get('ACC', 0)
                    acc_change = current_acc - init_acc
                    print(f"[DEBUG] Accuracy change from initial: {acc_change:+.10f}")
                print(f"[DEBUG] ==================================================")
                
                iter_data['metrics'] = convert_to_serializable(metrics)
                iter_data['changed_dict'] = convert_to_serializable(job['changed_dict'])
                iter_data['current_max_epsilon'] = current_max_epsilon
                
                # 实时追加到job['history']，让前端可以轮询获取
                job['history'].append(iter_data)
                job['progress'] = (job['current_iteration'] / job['max_iteration']) * 100
                print(f"[DEBUG] Progress: {job['progress']:.1f}% ({job['current_iteration']}/{job['max_iteration']})")
                
                # Check termination conditions
                if core_config.USE_BIAS_MITIGATION and current_max_epsilon <= job['epsilon_threshold']:
                    job['terminated'] = True
                    job['termination_reason'] = f"Epsilon threshold reached: {current_max_epsilon:.6f} <= {job['epsilon_threshold']:.6f}"
                    print(f"[DEBUG] {job['termination_reason']}")
                    break
                elif core_config.USE_ACCURACY_ENHANCEMENT and current_acc >= job['acc_threshold']:
                    job['terminated'] = True
                    job['termination_reason'] = f"Accuracy threshold reached: {current_acc:.6f} >= {job['acc_threshold']:.6f}"
                    print(f"[DEBUG] {job['termination_reason']}")
                    break
                    
            except Exception as iter_error:
                print(f"[ERROR] Error in iteration {iter_num}: {iter_error}")
                import traceback
                traceback.print_exc()
                # Continue to next iteration or break based on error type
                raise  # Re-raise to be caught by outer exception handler
        
        job['state'] = 'completed'
        job['progress'] = 100
        
        # Final evaluation (use the last transformed data)
        if job['history']:
            # Use last iteration's transformed data
            transformed_X_train = job['transformer'].transform_data(
                job['X_train'], job['changed_dict'], job['numerical'], job['categorical']
            )
            transformed_X_test = job['transformer'].transform_data(
                job['X_test'], job['changed_dict'], job['numerical'], job['categorical']
            )
            final_metrics = job['evaluator'].evaluate(
                transformed_X_train, job['Y_train'], job['O_train'],
                transformed_X_test, job['Y_test'], job['O_test']
            )
            
            # 添加综合公平性分数
            final_metrics['Overall_Fairness'] = calculate_overall_fairness_score(final_metrics)
        else:
            # No iterations, use initial metrics
            final_metrics = job['init_metrics']
        
        job['final_metrics'] = convert_to_serializable(final_metrics)
        print(f"[DEBUG] Job {job_id} completed successfully")
        
        # Save experiment log
        log_path = save_experiment_log(job_id, job, job.get('config_snapshot', {}))
        if log_path:
            job['log_path'] = log_path
        
    except Exception as e:
        job['state'] = 'failed'
        job['error'] = str(e)
        print(f"[ERROR] Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save experiment log even if failed (for debugging)
        log_path = save_experiment_log(job_id, job, job.get('config_snapshot', {}))
        if log_path:
            job['log_path'] = log_path

@app.route('/api/debias/<job_id>/run-full', methods=['POST'])
def run_full_process(job_id):
    """Start full debiasing process in background thread"""
    try:
        if job_id not in jobs:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404
        
        job = jobs[job_id]
        
        if job['state'] == 'running':
            return jsonify({
                'status': 'error',
                'message': 'Job is already running'
            }), 400
        
        # 重置状态
        job['current_iteration'] = 0
        job['history'] = []
        job['terminated'] = False
        job['termination_reason'] = None
        job['progress'] = 0
        
        # 在后台线程启动执行
        thread = threading.Thread(target=_run_full_process_thread, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Job started in background',
            'data': {
                'job_id': job_id,
                'state': 'running'
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/debias/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    """Get job status"""
    try:
        if job_id not in jobs:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404
        
        job = jobs[job_id]
        
        response_data = {
            'job_id': job_id,
            'state': job['state'],
            'current_iteration': job['current_iteration'],
            'max_iteration': job['max_iteration'],
            'progress': job.get('progress', 0),
            'history': convert_to_serializable(job['history']),
            'init_metrics': convert_to_serializable(job['init_metrics']),
            'init_epsilon': convert_to_serializable(job['init_epsilon']),
            'terminated': job.get('terminated', False),
            'termination_reason': job.get('termination_reason', None),
            'epsilon_threshold': job.get('epsilon_threshold', 0),
            'acc_threshold': job.get('acc_threshold', 0),
            'init_avg_epsilon': job.get('init_avg_epsilon', 0),  # 初始平均 epsilon
            'max_epsilon_series': None,  # 最大 epsilon 序列（用于图表显示）
        }
        # 构建 max_epsilon_series: 包含各轮的 current_max_epsilon
        try:
            history = job.get('history', [])
            print(f"[DEBUG] Building max_epsilon_series for job {job_id}")
            print(f"[DEBUG] Processing {len(history)} history items")
            
            # 收集每轮的 current_max_epsilon
            max_eps_hist = []
            for i, h in enumerate(history):
                v = h.get('current_max_epsilon', None)
                print(f"[DEBUG] Iteration {i}: current_max_epsilon = {v}")
                if v is not None:
                    max_eps_hist.append(v)
            
            response_data['max_epsilon_series'] = max_eps_hist
            
            print(f"[DEBUG] Final max_epsilon_series (length={len(max_eps_hist)}): {max_eps_hist}")
            print(f"[DEBUG] epsilon_threshold: {job.get('epsilon_threshold', 0)}")
        except Exception as e:
            # 保底兜底
            print(f"[ERROR] Exception building max_epsilon_series: {e}")
            import traceback
            traceback.print_exc()
            response_data['max_epsilon_series'] = []
        
        # 如果已完成，添加最终结果和日志路径
        if job['state'] == 'completed':
            if 'final_metrics' in job:
                response_data['final_metrics'] = convert_to_serializable(job['final_metrics'])
            if 'log_path' in job:
                response_data['log_path'] = job['log_path']
        
        # 如果失败，添加错误信息和日志路径（如果有）
        if job['state'] == 'failed':
            if 'error' in job:
                response_data['error'] = job['error']
            if 'log_path' in job:
                response_data['log_path'] = job['log_path']
        
        return jsonify({
            'status': 'success',
            'data': response_data
        })
    except Exception as e:
        print(f"[ERROR] Error in get_job_status for job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

def explain_transformation(attr, transform, config):
    """
    Helper function to explain a transformation in human-readable format.
    
    Parameters:
        attr: Attribute name
        transform: Transformation parameters (dict or 'dropped')
        config: Configuration snapshot with PARAMS_TRANSFORM_STREAM_CONFIG
    
    Returns:
        List of explanation strings
    """
    explanation = []
    
    if transform == 'dropped':
        explanation.append(f"    - {attr}: DROPPED (removed from dataset)")
        return explanation
    
    if not isinstance(transform, dict):
        explanation.append(f"    - {attr}: {transform}")
        return explanation
    
    # Skip scaler internal params
    if attr == '__scaler__':
        explanation.append(f"    - StandardScaler: Applied to all numerical features")
        return explanation
    
    # Extract transformation parameters
    has_beta_o = 'beta_O' in transform
    has_beta_y = 'beta_Y' in transform
    
    if not has_beta_o and not has_beta_y:
        # Categorical attribute re-binning
        explanation.append(f"    - {attr}: Re-binning")
        for key, value in transform.items():
            if key not in ['mean_', 'scale_', 'var_', 'feature_names', 'n_features_in_']:
                explanation.append(f"        {key} → {value}")
        return explanation
    
    # Numerical attribute transformation
    explanation.append(f"    - {attr}:")
    
    # Get transformation config
    transform_method = config.get('PARAMS_TRANSFORM', 'poly')
    transform_multi = config.get('PARAMS_TRANSFORM_MULTI', 't1')
    alpha_o = config.get('PARAMS_MAIN_ALPHA_O', 0.8)
    
    # Explain transformation type
    transform_type_map = {
        'poly': 'Polynomial',
        'log': 'Logarithmic',
        'arcsin': 'Arcsine'
    }
    explanation.append(f"        Type: {transform_type_map.get(transform_method, transform_method)}")
    
    if has_beta_o:
        beta_o_idx = transform['beta_O']
        explanation.append(f"        beta_O: index={beta_o_idx} (bias mitigation)")
        
    if has_beta_y:
        beta_y_idx = transform['beta_Y']
        explanation.append(f"        beta_Y: index={beta_y_idx} (accuracy enhancement)")
    
    if has_beta_o and has_beta_y:
        explanation.append(f"        Combined: {alpha_o:.2f} * f(beta_O) + {1-alpha_o:.2f} * f(beta_Y)")
    
    # Explain multi-beta mode
    multi_mode_map = {
        't1': 'Single value transform',
        't2': 'Cumulative product transform',
        't3': 'Cumulative sum transform'
    }
    explanation.append(f"        Mode: {multi_mode_map.get(transform_multi, transform_multi)}")
    
    return explanation

@app.route('/api/debias/<job_id>/download_transforms', methods=['GET'])
def download_transforms(job_id):
    """Download transformation rules as text file"""
    try:
        if job_id not in jobs:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404
        
        job = jobs[job_id]
        
        # Generate transformation rules text
        text_lines = []
        text_lines.append("=" * 80)
        text_lines.append("TRANSFORMATION RULES REPORT")
        text_lines.append("=" * 80)
        text_lines.append("")
        
        # Experiment info
        text_lines.append(f"Job ID: {job_id}")
        text_lines.append(f"Dataset: {job.get('dataset_name', 'N/A')}")
        text_lines.append(f"Target Column: {job.get('target', 'N/A')}")
        text_lines.append(f"Protected Attributes: {', '.join(job.get('protected', []))}")
        config = job.get('config_snapshot', {})
        text_lines.append(f"Classifier: {config.get('PARAMS_MAIN_CLASSIFIER', 'N/A')}")
        text_lines.append(f"Use Bias Mitigation: {config.get('USE_BIAS_MITIGATION', 'N/A')}")
        text_lines.append(f"Use Accuracy Enhancement: {config.get('USE_ACCURACY_ENHANCEMENT', 'N/A')}")
        text_lines.append(f"Total Iterations: {len(job.get('history', []))}")
        text_lines.append("")
        text_lines.append("=" * 80)
        text_lines.append("")
        
        # Initial state
        text_lines.append("INITIAL STATE (Iteration 0)")
        text_lines.append("-" * 80)
        init_metrics = job.get('init_metrics', {})
        if init_metrics:
            text_lines.append(f"  Initial Accuracy: {init_metrics.get('ACC', 0):.6f}")
            text_lines.append(f"  Initial F1 Score: {init_metrics.get('F1', 0):.6f}")
            text_lines.append(f"  Initial Recall: {init_metrics.get('Recall', 0):.6f}")
            text_lines.append(f"  Initial Precision: {init_metrics.get('Precision', 0):.6f}")
        text_lines.append(f"  Transformations: None (baseline)")
        text_lines.append("")
        
        # Iterate through history
        history = job.get('history', [])
        prev_changed_dict = {}
        
        for idx, iter_data in enumerate(history, start=1):
            text_lines.append(f"ITERATION {idx}")
            text_lines.append("-" * 80)
            
            # BM selection info
            if 'selected_attribute' in iter_data:
                text_lines.append(f"  Bias Mitigation Target:")
                text_lines.append(f"    - Protected Group: {iter_data.get('selected_label_O', 'N/A')}")
                text_lines.append(f"    - Selected Attribute: {iter_data['selected_attribute']}")
                text_lines.append("")
            
            # Changed dict (transformations)
            changed_dict = iter_data.get('changed_dict', {})
            
            # Find new transformations in this iteration
            new_transforms = {}
            for attr, transform in changed_dict.items():
                if attr not in prev_changed_dict or prev_changed_dict.get(attr) != transform:
                    new_transforms[attr] = transform
            
            if new_transforms:
                text_lines.append(f"  New Transformations Applied in This Iteration:")
                for attr, transform in new_transforms.items():
                    explanation_lines = explain_transformation(attr, transform, config)
                    text_lines.extend(explanation_lines)
                text_lines.append("")
            else:
                text_lines.append(f"  New Transformations: None (only re-evaluation)")
                text_lines.append("")
            
            # Show all accumulated transformations
            text_lines.append(f"  Accumulated Transformations (Total: {len([k for k in changed_dict.keys() if k != '__scaler__'])} attributes):")
            if changed_dict:
                for attr, transform in changed_dict.items():
                    if attr == '__scaler__':
                        text_lines.append(f"    - StandardScaler: Enabled")
                    else:
                        explanation_lines = explain_transformation(attr, transform, config)
                        text_lines.extend(explanation_lines)
            else:
                text_lines.append(f"    None")
            text_lines.append("")
            
            # Metrics
            metrics = iter_data.get('metrics', {})
            if metrics:
                text_lines.append(f"  Performance Metrics After This Iteration:")
                text_lines.append(f"    - Accuracy: {metrics.get('ACC', 0):.6f}")
                text_lines.append(f"    - F1 Score: {metrics.get('F1', 0):.6f}")
                text_lines.append(f"    - Recall: {metrics.get('Recall', 0):.6f}")
                text_lines.append(f"    - Precision: {metrics.get('Precision', 0):.6f}")
                text_lines.append("")
            
            # Fairness metrics summary
            fairness_metrics = ['EO', 'SP', 'EOpp']
            has_fairness = any(m in metrics for m in fairness_metrics)
            if has_fairness:
                text_lines.append(f"  Fairness Metrics:")
                for metric_name in fairness_metrics:
                    if metric_name in metrics:
                        metric_val = metrics[metric_name]
                        if isinstance(metric_val, dict):
                            for attr, val in metric_val.items():
                                text_lines.append(f"    - {metric_name}[{attr}]: {val:.6f}")
                        else:
                            text_lines.append(f"    - {metric_name}: {metric_val:.6f}")
                text_lines.append("")
            
            # Epsilon
            if 'current_max_epsilon' in iter_data:
                text_lines.append(f"  Bias Concentration (Max Epsilon): {iter_data['current_max_epsilon']:.10f}")
            if 'current_avg_epsilon' in iter_data:
                text_lines.append(f"  Bias Concentration (Avg Epsilon): {iter_data['current_avg_epsilon']:.10f}")
            
            text_lines.append("")
            prev_changed_dict = changed_dict.copy()
        
        # Final state
        text_lines.append("=" * 80)
        text_lines.append("FINAL STATE")
        text_lines.append("-" * 80)
        text_lines.append(f"  Status: {job.get('state', 'N/A')}")
        if job.get('terminated'):
            text_lines.append(f"  Terminated: Yes")
        if job.get('termination_reason'):
            text_lines.append(f"  Termination Reason: {job['termination_reason']}")
        
        final_metrics = job.get('final_metrics', {})
        if final_metrics:
            text_lines.append(f"  Final Accuracy: {final_metrics.get('ACC', 0):.6f}")
            text_lines.append(f"  Final F1 Score: {final_metrics.get('F1', 0):.6f}")
            text_lines.append(f"  Final Recall: {final_metrics.get('Recall', 0):.6f}")
            text_lines.append(f"  Final Precision: {final_metrics.get('Precision', 0):.6f}")
        
        text_lines.append("")
        text_lines.append("=" * 80)
        text_lines.append("COMPLETE TRANSFORMATION DICTIONARY")
        text_lines.append("=" * 80)
        text_lines.append("")
        text_lines.append("This section shows the raw transformation parameters as stored in changed_dict.")
        text_lines.append("These parameters are used by the Transform module to apply mathematical transformations.")
        text_lines.append("")
        
        # Get final changed_dict
        final_changed_dict = job.get('changed_dict', {})
        if not final_changed_dict:
            # Try to get from last history item
            if history:
                final_changed_dict = history[-1].get('changed_dict', {})
        
        if final_changed_dict:
            import json
            text_lines.append("Final changed_dict (JSON format):")
            text_lines.append("-" * 80)
            # Pretty print the changed_dict
            try:
                json_str = json.dumps(final_changed_dict, indent=2, ensure_ascii=False)
                text_lines.append(json_str)
            except:
                # Fallback to simple string representation
                for attr, transform in final_changed_dict.items():
                    text_lines.append(f"{attr}: {transform}")
            
            text_lines.append("")
            text_lines.append("-" * 80)
            text_lines.append("Parameter Explanation:")
            text_lines.append("  - 'dropped': Attribute was removed from the dataset")
            text_lines.append("  - 'beta_O': Index for bias mitigation transformation")
            text_lines.append("  - 'beta_Y': Index for accuracy enhancement transformation")
            text_lines.append("  - '__scaler__': StandardScaler parameters for numerical features")
            text_lines.append("")
            text_lines.append(f"Transformation Method: {config.get('PARAMS_TRANSFORM', 'N/A')}")
            text_lines.append(f"Multi-Beta Mode: {config.get('PARAMS_TRANSFORM_MULTI', 'N/A')}")
            text_lines.append(f"Alpha_O (BM weight): {config.get('PARAMS_MAIN_ALPHA_O', 'N/A')}")
            
            # Show stream_data config
            stream_config = config.get('PARAMS_TRANSFORM_STREAM_CONFIG', {})
            if stream_config:
                text_lines.append("")
                text_lines.append("Stream Configuration (for beta transformations):")
                for key, value in stream_config.items():
                    text_lines.append(f"  - {key}: {value}")
        else:
            text_lines.append("No transformations were applied.")
        
        text_lines.append("")
        text_lines.append("=" * 80)
        text_lines.append("END OF REPORT")
        text_lines.append("=" * 80)
        
        # Join lines and create response
        text_content = "\n".join(text_lines)
        
        # Create response with text file
        from flask import make_response
        response = make_response(text_content)
        response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename="transforms_{job_id[:8]}.txt"'
        
        return response
        
    except Exception as e:
        print(f"[ERROR] Error in download_transforms for job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================
# Start Server
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("BMWithAE Backend Server")
    print("=" * 60)
    print(f"Server running at: http://{backend_config.HOST}:{backend_config.PORT}")
    print(f"API endpoints available at: http://{backend_config.HOST}:{backend_config.PORT}/api/")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    print("\n⚠️  WARNING: Using Flask development server")
    print("For production deployment, use a WSGI server like Gunicorn:")
    print(f"  gunicorn -w 4 -b {backend_config.HOST}:{backend_config.PORT} wsgi:app")
    print("=" * 60)
    app.run(
        host=backend_config.HOST, 
        port=backend_config.PORT, 
        debug=backend_config.DEBUG, 
        threaded=True,
        use_reloader=False  # 禁用自动重载，避免运行过程中重启
    )

