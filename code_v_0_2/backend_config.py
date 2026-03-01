"""
Backend Configuration
Supports environment-based configuration for development and production
"""
import os

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Environment Configuration
ENV = os.getenv('FLASK_ENV', 'development')  # 'development' or 'production'

# Server Configuration
HOST = os.getenv('FLASK_HOST', '0.0.0.0')
PORT = int(os.getenv('FLASK_PORT', '5001'))
DEBUG = ENV == 'development'  # Automatically disable debug in production

# File Upload Configuration
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'backend', 'uploads')
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'backend', 'results')
LOGS_FOLDER = os.path.join(PROJECT_ROOT, 'backend', 'logs')
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB

# Demo Datasets Configuration
DEMO_DATASETS = {
    'credit': {
        'path': os.path.join(PROJECT_ROOT, 'data', 'credit.xlsx'),
        'target': 'default payment next month',
        'protected': ['SEX', 'MARRIAGE'],  # Default: only SEX and MARRIAGE, not AGE
        'description': 'Credit Card Default Dataset'
    },
    'compas': {
        'path': os.path.join(PROJECT_ROOT, 'data', 'data_compas.csv'),
        'target': 'two_year_recid',
        'protected': ['sex', 'race'],
        'description': 'COMPAS Recidivism Dataset'
    }
}
