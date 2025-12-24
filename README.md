# plant-splat

Synthetic data pipeline for plant disease detection using 3D Gaussian splatting.

## Setup

```bash
# Python environment
uv venv --python 3.12
source .venv/bin/activate
uv sync

# System dependencies (macOS)
brew install colmap opencv cmake ffmpeg

# Build OpenSplat (with MPS for Apple Silicon)
# Requires Xcode with Metal toolchain: xcodebuild -downloadComponent MetalToolchain
git clone --depth 1 https://github.com/pierotofy/OpenSplat tools/OpenSplat
cd tools/OpenSplat && mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)');$(brew --prefix opencv)/lib/cmake/opencv4" -DGPU_RUNTIME=MPS ..
make -j8
cd ../../..
```

## Reconstruction Pipeline

Convert a video of a plant into a Gaussian splat:

```bash
# Full pipeline: video → frames → COLMAP → OpenSplat
uv run python -m src.reconstruct data/raw/mint.MOV -o data --name mint

# With custom parameters
uv run python -m src.reconstruct data/raw/mint.MOV \
    --frame-skip 10 \   # Extract every Nth frame (default: 10)
    --iters 3000 \      # Training iterations (default: 3000)
    --downscale 1       # Image scale factor (default: 1, use 2 for faster/lower quality)
```

Output: `data/splats/<name>.ply`

## Viewing Splats

Load `.ply` files in [SuperSplat](https://superspl.at/editor) or any Gaussian splat viewer.
