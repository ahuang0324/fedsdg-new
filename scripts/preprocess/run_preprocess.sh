#!/bin/bash
# Data Preprocessing Script
# Preprocess datasets for offline loading (OfflineCIFAR10/OfflineCIFAR100)

# ==================== Configuration ====================
DATASET="cifar100"      # cifar10 or cifar100
IMAGE_SIZE=224          # Target image size (224 for ViT pretrained)
FORCE=false             # Set to true to overwrite existing files
# =======================================================

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --image_size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help)
            echo "Usage: bash run_preprocess.sh [options]"
            echo ""
            echo "Options:"
            echo "  --dataset     Dataset name (cifar10, cifar100). Default: cifar100"
            echo "  --image_size  Target image size. Default: 224"
            echo "  --force       Force overwrite existing files"
            echo ""
            echo "Examples:"
            echo "  bash run_preprocess.sh --dataset cifar100 --image_size 224"
            echo "  bash run_preprocess.sh --dataset cifar10 --image_size 128 --force"
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
echo "Data Preprocessing"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Dataset: ${DATASET}"
echo "  - Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  - Force overwrite: ${FORCE}"
echo ""
echo "=========================================="
echo ""

# Change to project root directory
cd "$(dirname "$0")/../.."

# Build command
CMD="python -m fl.data.preprocess --dataset ${DATASET} --image_size ${IMAGE_SIZE}"
if [ "$FORCE" = true ]; then
    CMD="${CMD} --force"
fi

# Run preprocessing
echo "Running: ${CMD}"
echo ""
${CMD}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Preprocessing Complete!"
    echo "=========================================="
    echo ""
    echo "Output location:"
    echo "  ./datasets/preprocessed/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/"
    echo ""
    echo "You can now use offline data in training:"
    echo "  --use_offline_data --offline_data_root ./datasets/preprocessed"
else
    echo "Preprocessing Failed!"
    echo "=========================================="
fi

exit $EXIT_CODE
