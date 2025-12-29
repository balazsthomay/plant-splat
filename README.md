# plant-splat

Synthetic data pipeline for plant disease detection using 3D Gaussian splatting.

## Setup (GPU VM)

Requires CUDA. Tested on RTX 4090.

```bash
# Clone and setup
git clone <repo> plant-splat && cd plant-splat
uv venv --python 3.12
source .venv/bin/activate
uv sync

# SAM 3 + dependencies (CUDA only, not in pyproject.toml)
uv add "sam3 @ git+https://github.com/facebookresearch/sam3.git"
uv add decord pycocotools

# Authenticate for gated model weights
# First request access at: https://huggingface.co/facebook/sam3
hf auth login
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
3. **SAM 3** segments the subject using text prompt (e.g., "potted plant without pot")
4. **Filter points** by mask projection (keep foreground only)
5. **OpenSplat** trains on filtered points
6. **Post-process** filters Gaussians by mask projection (removes any that leaked through)

### Segmentation

SAM 3 uses text prompts for semantic segmentation. The default prompt excludes the pot:

```bash
uv run src/segment.py data/colmap/mint/images/
uv run src/segment.py data/colmap/mint/images/ --prompt "plant leaves"
```

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

Apply diseases to healthy renders using SDXL img2img + per-disease LoRAs trained on PlantSeg.

**Separate LoRA per disease** (`powdery_mildew`, `rust`, `leaf_spot`, `blight`, `chlorosis`) for better quality. ~25MB each, ~20 min training per disease on RTX 4090.

### Step 1: Prepare Training Data (local Mac)

```bash
# Download PlantSeg from Kaggle: https://www.kaggle.com/datasets/weitianqi/plantseg
# Extract to data/plantsegv2/

# Prepare per-disease folders
uv run src/prepare_disease_data.py --plantseg data/plantsegv2 --output data/disease_training
```

### Step 2: Clone & Upload to Vast.ai

```bash
# On VM: clone repo
cd /workspace && git clone <your-repo> plant-splat && cd plant-splat

# From Mac: upload training data
scp -r data/disease_training/ vast:/workspace/plant-splat/data/
```

### Step 3: Train on Vast.ai (RTX 4090)

```bash
# Setup kohya
git clone https://github.com/kohya-ss/sd-scripts tools/kohya
cd tools/kohya && pip install -r requirements.txt && accelerate config

# Generate configs for all 5 diseases
cd /workspace/plant-splat
uv run src/train_disease_lora.py --all --config-only

# Train all (~1.5 hours total)
cd tools/kohya
for cfg in /workspace/plant-splat/models/lora/*_train_config.toml; do
    accelerate launch sdxl_train_network.py --config_file="$cfg"
done
```

### Step 4: Download LoRAs

```bash
scp vast:/workspace/plant-splat/models/lora/*.safetensors models/lora/
```

### Step 5: Generate Diseased Images

```bash
# All diseases (auto-loads per-disease LoRA)
uv run src/synthesize_disease.py data/synthetic/ --lora-dir models/lora/

# Specific disease
uv run src/synthesize_disease.py data/synthetic/ --lora-dir models/lora/ --disease rust -n 200
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o` | `data/synthetic_diseased/` | Output directory |
| `-n` | all | Number of images |
| `--lora-dir` | none | Directory with per-disease LoRAs |
| `--lora` | none | Path to single LoRA (legacy) |
| `--lora-scale` | 1.0 | LoRA influence (0-1) |
| `--disease` | random | Disease type |
| `--severity-min` | 0.3 | Min severity (0-1) |
| `--severity-max` | 0.7 | Max severity (0-1) |

### Output

```
data/synthetic_diseased/
├── images/           # Diseased images
├── masks/            # Plant masks
├── disease_masks/    # Affected regions
└── annotations.json  # Includes disease type, severity per image
```

## References

### 3D Plant Reconstruction
- [PlantGaussian: 3D Gaussian Splatting for Plant Modeling](https://www.sciencedirect.com/science/article/pii/S2214514125000261) - Crop Journal 2025
- [Splanting: 3DGS Plant Dataset](https://dl.acm.org/doi/10.1145/3681758.3698009) - SIGGRAPH Asia 2024
- [3DGS vs NeRF for Wheat](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf022/8096368) - GigaScience 2025

### Synthetic Data for Plant Disease
- [PhytoSynth: Generative Models for Crop Disease](https://arxiv.org/abs/2505.01823) - arXiv 2025
- [Synthetic Data at Scale for Plant Disease](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1360113/full) - Frontiers 2024
- [DiffusionPix2Pix for Graded Disease Severity](https://www.sciencedirect.com/science/article/pii/S0168169924010810) - Computers in Agriculture 2024
- [Diffusion for Plant Disease Augmentation](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1280496/full) - Frontiers 2023

### ControlNet & Domain Adaptation
- [Weed Augmentation with ControlNet](https://www.sciencedirect.com/science/article/abs/pii/S0168169925002297) - Computers in Agriculture 2025
- [Domain-Targeted Plant Style Transfer (LoRA + ControlNet)](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/papers/Hartley_Domain_Targeted_Synthetic_Plant_Style_Transfer_using_Stable_Diffusion_LoRA_CVPRW_2024_paper.pdf) - CVPR 2024