#!/bin/bash
# Production deployment script for BMWithAE backend
# For Linux/Mac systems

# Set environment variables
export FLASK_ENV=production
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5001

# Number of worker processes
# NOTE: Using 1 worker because datasets dict is stored in memory
# For multi-worker support, need to use Redis or database for session storage
WORKERS=1

# Timeout for worker processes (seconds)
TIMEOUT=120

# Access log file
ACCESS_LOG="logs/access.log"
ERROR_LOG="logs/error.log"
COMBINED_LOG="logs/combined.log"

# Create log directory if it doesn't exist
mkdir -p logs

# Clear all log files on startup
> "$ACCESS_LOG"
> "$ERROR_LOG"
> "$COMBINED_LOG"

# Add startup marker to logs
echo "========================================" >> "$COMBINED_LOG"
echo "Backend started at $(date)" >> "$COMBINED_LOG"
echo "========================================" >> "$COMBINED_LOG"

echo "========================================" >> "$ERROR_LOG"
echo "Backend started at $(date)" >> "$ERROR_LOG"
echo "========================================" >> "$ERROR_LOG"

echo "========================================"
echo "Starting BMWithAE Backend (Production)"
echo "========================================"
echo "Workers: $WORKERS"
echo "Timeout: ${TIMEOUT}s"
echo "Host: $FLASK_HOST:$FLASK_PORT"
echo "========================================"

# Start Gunicorn with output redirection
# All stdout and stderr will be captured in combined.log
gunicorn \
    --workers $WORKERS \
    --worker-class sync \
    --timeout $TIMEOUT \
    --bind $FLASK_HOST:$FLASK_PORT \
    --access-logfile $ACCESS_LOG \
    --error-logfile $ERROR_LOG \
    --log-level info \
    --capture-output \
    --enable-stdio-inheritance \
    --preload \
    wsgi:app 2>&1 | tee -a "$COMBINED_LOG"

# Alternative: Using Waitress (cross-platform)
# waitress-serve --host=$FLASK_HOST --port=$FLASK_PORT --threads=8 wsgi:app

