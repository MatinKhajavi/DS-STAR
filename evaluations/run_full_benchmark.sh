#!/bin/bash
# Run DS-STAR on full DABStep benchmark (default split)
# This will generate a submission-ready file for the leaderboard

set -e  # Exit on error

echo "🚀 DS-STAR Full Benchmark Run"
echo "=============================="
echo ""
echo "⚠️  This will run on the FULL default split (all tasks)"
echo "   This may take several hours depending on your setup."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted by user"
    exit 0
fi

echo ""
echo "Starting benchmark run..."
echo "Timestamp: $(date)"
echo ""

# Run the benchmark
poetry run python evaluations/run_dabstep.py \
    --split default \
    --model gemini-2.5-pro \
    --provider gemini \
    --max-rounds 20

EXIT_CODE=$?

echo ""
echo "=============================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Benchmark complete!"
    echo ""
    echo "📁 Results saved to: ./runs/"
    echo ""
    echo "To submit to leaderboard:"
    echo "  1. Find the file: runs/ds-star_default_*_submission.jsonl"
    echo "  2. Submit at: https://huggingface.co/spaces/adyen/DABstep"
else
    echo "❌ Benchmark failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

