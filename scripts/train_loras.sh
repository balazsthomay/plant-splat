#!/bin/bash
# Train missing disease LoRAs
# Run on GPU VM (RTX 4090 recommended, ~20 min per disease)
#
# Prerequisites:
# - Kaggle API credentials (~/.kaggle/kaggle.json)
#   Get from: https://www.kaggle.com/settings → Create New Token
#
# Usage: bash scripts/train_loras.sh

set -e

echo "=============================================="
echo "Disease LoRA Training Setup"
echo "=============================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# 1. Check Kaggle credentials
echo ""
echo "[1/6] Checking Kaggle credentials..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: Kaggle credentials not found."
    echo ""
    echo "Setup instructions:"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Click 'Create New Token' (downloads kaggle.json)"
    echo "  3. Upload to VM: scp kaggle.json vast:~/.kaggle/"
    echo "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo "  5. Re-run this script"
    exit 1
fi
chmod 600 ~/.kaggle/kaggle.json
echo "Kaggle credentials: OK"

# 2. Install kohya-ss
echo ""
echo "[2/6] Installing kohya-ss..."
KOHYA_DIR="$PROJECT_DIR/tools/kohya"
if [ -f "$KOHYA_DIR/sdxl_train_network.py" ]; then
    echo "kohya-ss already installed, skipping..."
else
    git clone https://github.com/kohya-ss/sd-scripts "$KOHYA_DIR"
    cd "$KOHYA_DIR"
    pip install -r requirements.txt

    # Configure accelerate (non-interactive)
    mkdir -p ~/.cache/huggingface/accelerate
    cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    echo "accelerate configured"
    cd "$PROJECT_DIR"
fi
echo "kohya-ss: OK"

# 3. Download PlantSeg dataset
echo ""
echo "[3/6] Downloading PlantSeg dataset..."
if [ -d "data/plantsegv2" ]; then
    echo "PlantSeg already downloaded, skipping..."
else
    pip install -q kaggle
    mkdir -p data
    kaggle datasets download -d weitianqi/plantseg -p data/
    unzip -q data/plantseg.zip -d data/plantsegv2
    rm data/plantseg.zip
fi
echo "PlantSeg: OK"

# 4. Prepare training data
echo ""
echo "[4/6] Preparing training data..."
if [ -d "data/disease_training/rust" ]; then
    echo "Training data already prepared, skipping..."
else
    uv run src/prepare_disease_data.py --plantseg data/plantsegv2 --output data/disease_training
fi
echo "Training data: OK"

# 5. Generate configs
echo ""
echo "[5/6] Generating training configs..."
uv run src/train_disease_lora.py --all --config-only

# 6. Train all LoRAs
echo ""
echo "[6/6] Training LoRAs (this takes ~1.5 hours for 5 diseases)..."
cd "$KOHYA_DIR"

for cfg in "$PROJECT_DIR"/models/lora/*_train_config.toml; do
    disease=$(basename "$cfg" _train_config.toml)

    # Skip if LoRA already exists
    if [ -f "$PROJECT_DIR/models/lora/${disease}.safetensors" ]; then
        echo "[$disease] Already trained, skipping..."
        continue
    fi

    echo ""
    echo "[$disease] Training..."
    accelerate launch sdxl_train_network.py --config_file="$cfg"
done

cd "$PROJECT_DIR"

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "LoRAs saved to: models/lora/"
ls -la models/lora/*.safetensors 2>/dev/null || echo "(no .safetensors files found)"
echo ""
echo "Next: Run disease synthesis"
echo "  uv run src/synthesize_disease.py data/synthetic/ --lora-dir models/lora/ -o data/synthetic_diseased/"
