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

## Usage

```bash
# Full scene (includes background)
uv run src/reconstruct.py data/raw/plant.MOV

# Isolated plant (background removed)
uv run src/reconstruct.py data/raw/plant.MOV --isolate
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--isolate` | off | Remove background (SAM 2 + filtering + post-process) |
| `--name` | video filename | Project name |
| `--frame-skip` | 20 | Extract every Nth frame |
| `--iters` | 3000 | Training iterations |
| `--downscale` | 1 | Image scale factor |

### Output

- Full scene: `data/splats/<name>.ply`
- Isolated: `data/splats/<name>_clean.ply`

## How Isolation Works

The `--isolate` pipeline:
1. **Extract frames** from video
2. **COLMAP** reconstructs sparse 3D point cloud
3. **SAM 2** segments the subject in each frame
4. **Filter points** by mask projection (keep foreground only)
5. **OpenSplat** trains on filtered points
6. **Post-process** removes residual background Gaussians

### Subject Detection

SAM 2 uses a center point prompt on frame 0—it clicks the exact center of the image and segments whatever object is there, then propagates that mask through all frames.

**Requirement:** Keep the plant centered when filming. The pot can be included—it provides realistic context for synthetic training data.

**Limitation:** SAM 2 has no semantic understanding. It segments whatever the center click lands on, so off-center plants will fail.

**Ideal solution:** SAM 3 supports natural language prompts ("segment the plant") but requires CUDA (Triton dependency). With CUDA, text-prompted segmentation would remove the center-framing requirement.

## Viewing Splats

Load `.ply` files in [SuperSplat](https://superspl.at/editor) or any Gaussian splat viewer.
