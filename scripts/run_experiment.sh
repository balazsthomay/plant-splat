#!/bin/bash
# Phase 5: Synthetic→Real validation experiment
#
# Before running, upload synthetic data:
#   tar -czf synthetic_data.tar.gz data/synthetic data/synthetic_diseased
#   scp -P <port> synthetic_data.tar.gz root@<ip>:/workspace/plant-splat/
#   # On VM: tar -xzf synthetic_data.tar.gz
#
# Usage: bash scripts/run_experiment.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$PROJECT_DIR"

# 1. Check synthetic data
echo "[1/3] Checking data..."
if [ ! -d "data/synthetic/images" ] || [ ! -d "data/synthetic_diseased/images" ]; then
    echo "ERROR: Synthetic data not found. Upload first:"
    echo "  tar -xzf synthetic_data.tar.gz"
    exit 1
fi
echo "  Synthetic: $(ls data/synthetic/images/*.png | wc -l) + $(ls data/synthetic_diseased/images/*.png | wc -l) images"

# 2. Download PlantSegV2 if needed
echo "[2/3] Checking PlantSegV2..."
if [ ! -f "data/plantsegv2/Metadatav2.csv" ]; then
    if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_API_TOKEN" ]; then
        echo "ERROR: Set KAGGLE_USERNAME and KAGGLE_API_TOKEN"
        exit 1
    fi
    pip install -q kaggle
    kaggle datasets download -d weitianqi/plantseg -p data/
    unzip -q data/plantseg.zip -d data/plantsegv2
    rm data/plantseg.zip
    # Fix nested extraction (zip contains plantsegv2/ folder)
    if [ -d "data/plantsegv2/plantsegv2" ]; then
        mv data/plantsegv2/plantsegv2/* data/plantsegv2/
        rmdir data/plantsegv2/plantsegv2
    fi
fi
# Flatten images if in subdirs (train/val/test)
if [ -d "data/plantsegv2/images/train" ]; then
    mv data/plantsegv2/images/*/* data/plantsegv2/images/
    rmdir data/plantsegv2/images/train data/plantsegv2/images/val data/plantsegv2/images/test 2>/dev/null || true
fi
echo "  PlantSegV2: $(ls data/plantsegv2/images/*.jpg 2>/dev/null | wc -l) images"

# 3. Run experiment
echo "[3/3] Running experiment..."
uv run src/classify.py experiment --epochs 30 --output results/

echo ""
echo "Done! Results: results/experiment_results.json"
