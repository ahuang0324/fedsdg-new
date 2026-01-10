#!/bin/bash
# FedSDG Gate Visualization Script
# Visualize gate coefficients (lambda_k) from trained FedSDG models

# ==================== Configuration ====================
EXPERIMENT=""           # Experiment name (supports fuzzy matching)
CHECKPOINT=""           # Or: direct checkpoint path
PREFIX="analysis"       # Output file prefix
USE_LATEST=false        # Use latest experiment
LIST_ONLY=false         # List available experiments
# =======================================================

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment|-e)
            EXPERIMENT="$2"
            shift 2
            ;;
        --checkpoint|-c)
            CHECKPOINT="$2"
            shift 2
            ;;
        --prefix|-p)
            PREFIX="$2"
            shift 2
            ;;
        --latest)
            USE_LATEST=true
            shift
            ;;
        --list|-l)
            LIST_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: bash run_gate_visualization.sh [options]"
            echo ""
            echo "Options:"
            echo "  --experiment, -e NAME   Experiment name (supports fuzzy matching)"
            echo "  --checkpoint, -c PATH   Checkpoint directory path"
            echo "  --prefix, -p PREFIX     Output file prefix (default: analysis)"
            echo "  --latest                Use the latest experiment"
            echo "  --list, -l              List all available experiments"
            echo ""
            echo "Examples:"
            echo "  bash run_gate_visualization.sh --list"
            echo "  bash run_gate_visualization.sh --latest"
            echo "  bash run_gate_visualization.sh --experiment cifar100_vit_pretrained"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "FedSDG Gate Visualization"
echo "=========================================="
echo ""

# Change to project root directory
cd "$(dirname "$0")/../.."

# Build command - 直接调用 fl.utils.visualization 模块
CMD="python -m fl.utils.visualization"

if [ "$LIST_ONLY" = true ]; then
    CMD="${CMD} --list"
elif [ "$USE_LATEST" = true ]; then
    CMD="${CMD} --latest --prefix ${PREFIX}"
elif [ -n "$CHECKPOINT" ]; then
    CMD="${CMD} --checkpoint ${CHECKPOINT} --prefix ${PREFIX}"
elif [ -n "$EXPERIMENT" ]; then
    CMD="${CMD} --experiment ${EXPERIMENT} --prefix ${PREFIX}"
else
    echo "No action specified. Use --help for usage."
    exit 0
fi

# Run visualization
echo "Running: ${CMD}"
echo ""
${CMD}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Done!"
else
    echo "Failed!"
fi
echo "=========================================="

exit $EXIT_CODE
