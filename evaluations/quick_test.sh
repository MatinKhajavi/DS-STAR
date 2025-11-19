#!/bin/bash
# Quick test script for DS-STAR on DABStep dev split

echo "🚀 DS-STAR Quick Test on DABStep"
echo "================================"
echo ""
echo "Running on first 10 tasks from dev split..."
echo ""

if poetry run python evaluations/run_dabstep.py \
    --split dev \
    --max-tasks 10 \
    --model gemini-2.5-pro \
    --provider gemini \
    --max-rounds 20; then
    echo ""
    echo "✅ Test complete! Check the 'runs/' directory for results."
else
    echo ""
    echo "❌ Test failed with exit code $?"
    exit 1
fi

