#!/bin/bash
# VM Setup Script for plant-splat
# Run once on a fresh GPU instance (tested on Ubuntu 22.04 + RTX 3070/3090/4090)
#
# Usage: bash scripts/setup_vm.sh
#
# Prerequisites:
# - CUDA toolkit installed (nvcc in PATH)
# - Python 3.12+
# - uv installed

set -e  # Exit on error

echo "=============================================="
echo "plant-splat VM Setup"
echo "=============================================="

# Detect GPU architecture
GPU_ARCH=${GPU_ARCH:-86}  # Default to RTX 3070 (Ampere)
echo "GPU architecture: sm_$GPU_ARCH (set GPU_ARCH env var to override)"

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA not found. Install CUDA toolkit first."
    exit 1
fi
echo "CUDA: $(nvcc --version | grep release)"

# 1. System dependencies
echo ""
echo "[1/5] Installing system dependencies..."
apt update
apt install -y \
    xvfb \
    libopencv-dev \
    git cmake ninja-build \
    libboost-all-dev libeigen3-dev libflann-dev libfreeimage-dev \
    libmetis-dev libgoogle-glog-dev libgtest-dev libsqlite3-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev

# 2. Build COLMAP with CUDA
echo ""
echo "[2/5] Building COLMAP with CUDA..."
COLMAP_DIR="/opt/colmap"
if [ -f "/usr/local/bin/colmap" ] && /usr/local/bin/colmap -h 2>&1 | grep -q "with CUDA"; then
    echo "COLMAP with CUDA already installed, skipping..."
else
    rm -rf "$COLMAP_DIR"
    git clone https://github.com/colmap/colmap.git "$COLMAP_DIR"
    cd "$COLMAP_DIR"
    git checkout 3.9.1

    # Fix missing include (known issue in 3.9.1)
    grep -rl "std::unique_ptr\|std::shared_ptr" --include="*.cc" --include="*.h" src | while read f; do
        if ! grep -q "#include <memory>" "$f"; then
            sed -i '1i #include <memory>' "$f"
        fi
    done

    mkdir -p build && cd build
    cmake .. -GNinja \
        -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH \
        -DCMAKE_BUILD_TYPE=Release
    ninja -j$(nproc)
    ninja install
    cd /
fi
echo "COLMAP: $(/usr/local/bin/colmap -h 2>&1 | head -1)"

# 3. Build OpenSplat
echo ""
echo "[3/5] Building OpenSplat..."
# Resolve project dir robustly (works whether script is run via absolute or relative path)
SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ ! "$SCRIPT_PATH" = /* ]]; then
    SCRIPT_PATH="$(pwd)/$SCRIPT_PATH"
fi
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OPENSPLAT_DIR="$PROJECT_DIR/tools/OpenSplat"

if [ -f "$OPENSPLAT_DIR/build/opensplat" ]; then
    echo "OpenSplat already built, skipping..."
else
    mkdir -p "$PROJECT_DIR/tools"
    rm -rf "$OPENSPLAT_DIR"
    git clone https://github.com/pierotofy/OpenSplat.git "$OPENSPLAT_DIR"
    cd "$OPENSPLAT_DIR"
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH \
        -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    make -j$(nproc)
fi
echo "OpenSplat: $OPENSPLAT_DIR/build/opensplat"

# 4. Python dependencies (SAM 3)
echo ""
echo "[4/5] Installing Python dependencies..."
cd "$PROJECT_DIR"
uv sync
uv add "sam3 @ git+https://github.com/facebookresearch/sam3.git" || true
uv add decord pycocotools || true

# 5. HuggingFace auth check
echo ""
echo "[5/5] Checking HuggingFace authentication..."
if uv run python3 -c "from huggingface_hub import HfFolder; assert HfFolder.get_token()" 2>/dev/null; then
    echo "HuggingFace: authenticated"
else
    echo "WARNING: HuggingFace not authenticated. Run: hf auth login"
fi

# Environment setup
echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Environment variables (add to ~/.bashrc):"
echo "  export OMP_NUM_THREADS=8"
echo "  export QT_QPA_PLATFORM=offscreen"
echo ""
echo "Run pipeline:"
echo "  xvfb-run -a uv run src/reconstruct.py data/raw/plant.MOV --isolate"
echo ""
