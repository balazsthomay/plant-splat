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

## Rendering

### Production (CUDA)

Use [gsplat](https://github.com/nerfstudio-project/gsplat) for production rendering. It's 100-1000x faster than pure PyTorch thanks to custom CUDA kernels and tile-based rasterization.

### Development (Mac/MPS)

gsplat requires CUDA. On Apple Silicon, we use a pure PyTorch renderer (`src/render.py`). It's slow (1-30 sec/frame) but functional for dataset generation.

| Backend | Speed | Use Case |
|---------|-------|----------|
| gsplat (CUDA) | ~60 fps | Production, real-time |
| Pure PyTorch (MPS/CPU) | 1-30 sec/frame | Development, batch rendering |

**Note:** gsplat-mps exists but is AGPLv3-licensed and stuck at v0.1.3.

## Dataset Generation

Generate synthetic training images with varied viewpoints, lighting, and backgrounds:

```bash
# Default: varied backgrounds + lighting augmentation
uv run src/generate_dataset.py data/splats/mint3_clean.ply -n 1000 -o data/synthetic/

# With custom background images
uv run src/generate_dataset.py data/splats/mint3_clean.ply -n 1000 --bg-dir data/backgrounds/

# White backgrounds, no lighting variation
uv run src/generate_dataset.py data/splats/mint3_clean.ply -n 1000 --bg-mode white --no-lighting
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o` | `data/synthetic/` | Output directory |
| `-n` | 100 | Number of views (azimuth × elevation grid) |
| `-d` | 1 | Downscale factor (1 = full res, 2 = half) |
| `--bg-mode` | varied | Background: `varied` (greenhouse colors), `white`, `black`, `random` |
| `--bg-dir` | none | Directory with background images (random crops) |
| `--no-lighting` | off | Disable lighting augmentation |

### Augmentations

**Lighting** (enabled by default):
- Color temperature: 2700K (warm) to 8000K (cool)
- Intensity: 0.6× to 1.3×
- Contrast: 0.9× to 1.1×

**Backgrounds**:
- `varied`: Greenhouse-like palette (whites, grays, greens, browns)
- `--bg-dir`: Random crops from your images, resized to match

### Output

```
data/synthetic/
├── images/          # RGB renders with augmentations
├── masks/           # Alpha masks (= segmentation)
└── annotations.json # Bounding boxes, camera params, lighting/bg metadata
```

### GPU Rental (for large datasets)

For 1000+ images, rent a GPU. RTX 3060/3070 is plenty for 31k Gaussians. Budget ~20GB storage (splat + outputs + deps).

## Disease Synthesis

Apply diseases to healthy renders using SD 1.5 inpainting:

```bash
# Random diseases, severity 0.3-0.7
uv run src/synthesize_disease.py data/synthetic/ -o data/synthetic_diseased/

# Specific disease
uv run src/synthesize_disease.py data/synthetic/ --disease powdery_mildew
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o` | `data/synthetic_diseased/` | Output directory |
| `-n` | all | Number of images |
| `--disease` | random | `powdery_mildew`, `leaf_spot`, `rust`, `chlorosis`, `blight` |
| `--severity-min` | 0.3 | Min severity (0-1) |
| `--severity-max` | 0.7 | Max severity (0-1) |
| `--steps` | 30 | Diffusion steps (lower = faster) |

### Output

```
data/synthetic_diseased/
├── images/          # Diseased images
├── masks/           # Plant masks
├── disease_masks/   # Affected regions
└── annotations.json
```