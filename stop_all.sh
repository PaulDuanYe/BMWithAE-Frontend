#!/bin/bash
# BMWithAE Stop Services Script

echo "=========================================="
echo "Stopping BMWithAE Services"
echo "=========================================="

# Stop backend
echo "Stopping backend..."
screen -S bmwithae-backend -X quit 2>/dev/null && echo "✅ Backend stopped" || echo "⚠️  Backend session not found"

# Stop frontend
echo "Stopping frontend..."
screen -S bmwithae-frontend -X quit 2>/dev/null && echo "✅ Frontend stopped" || echo "⚠️  Frontend session not found"

echo ""
echo "=========================================="
echo "Services Stopped"
echo "=========================================="
echo ""
echo "📋 Check remaining sessions:"
echo "   screen -ls"
echo ""

