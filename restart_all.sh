#!/bin/bash
# BMWithAE Services Management Script
# This script can be used for both initial startup and restart

echo "=========================================="
echo "BMWithAE Services Manager"
echo "=========================================="

# Stop all services (safe even if not running)
echo "Stopping old services (if any)..."
screen -S bmwithae-backend -X quit 2>/dev/null || true
screen -S bmwithae-frontend -X quit 2>/dev/null || true
pkill -f "gunicorn.*wsgi:app" 2>/dev/null || true
pkill -f "http.server 8000" 2>/dev/null || true
sleep 3

# Start Backend
echo "Starting Backend..."
screen -dmS bmwithae-backend bash -c 'cd /public/products/BMWithAE/backend && source /root/miniconda3/bin/activate bmwithae_py312 && ./start_production.sh'
sleep 3

# Start Frontend
echo "Starting Frontend..."
screen -dmS bmwithae-frontend bash -c 'cd /public/products/BMWithAE/frontend && python3 -m http.server 8000'
sleep 2

echo ""
echo "=========================================="
echo "Services Started Successfully!"
echo "=========================================="
echo ""
echo "View all sessions:"
echo "   screen -ls"
echo ""
echo "Attach to backend:"
echo "   screen -r bmwithae-backend"
echo ""
echo "Attach to frontend:"
echo "   screen -r bmwithae-frontend"
echo ""
echo "Detach session:"
echo "   Ctrl+A then D"
echo ""
echo "Access URLs:"
echo "   Frontend: http://8.148.159.241:8000"
echo "   Backend:  http://8.148.159.241:5001/api/config"
echo ""
echo "Remember to open ports 5001 and 8000 in Aliyun Security Group!"
echo ""

