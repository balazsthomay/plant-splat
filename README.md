# plant-splat

Synthetic data pipeline for plant disease detection using 3D Gaussian splatting.

**Result:** Classifier trained on synthetic renders achieves **99.5% accuracy** on real photographs.

## Quick Start

```bash
# Setup (GPU VM, 16GB+ VRAM: RTX 4070+, A10, A100)
git clone https://github.com/balazsthomay/plant-splat.git && cd plant-splat
bash scripts/setup_vm.sh  # Auto-detects GPU
hf auth login

# Full pipeline
xvfb-run -a uv run src/reconstruct.py data/raw/plant.MOV --isolate
uv run src/generate_dataset.py data/splats/plant_clean.ply -n 1000
uv run src/synthesize_disease.py data/synthetic/ --lora-dir models/lora/
bash scripts/run_experiment.sh
```

## Setup

Requires CUDA with **16GB+ VRAM**. SAM 3 text prompts need full video in memory for temporal consistency.

`scripts/setup_vm.sh` auto-detects GPU architecture and installs COLMAP, OpenSplat, and SAM 3.

<details>
<summary>Manual setup (if script fails)</summary>

```bash
# Python deps
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv add "sam3 @ git+https://github.com/facebookresearch/sam3.git"
uv add decord pycocotools

# System deps
apt install -y xvfb libopencv-dev

# COLMAP with CUDA
git clone https://github.com/colmap/colmap.git /opt/colmap
cd /opt/colmap && mkdir build && cd build
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
ninja && ninja install

# OpenSplat
git clone https://github.com/pierotofy/OpenSplat.git tools/OpenSplat
cd tools/OpenSplat && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
make -j$(nproc)
```
</details>

## Usage

```bash
# Full scene (includes background)
xvfb-run -a uv run src/reconstruct.py data/raw/plant.MOV

# Isolated plant (background removed via SAM 3)
xvfb-run -a uv run src/reconstruct.py data/raw/plant.MOV --isolate
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--isolate` | off | Remove background (SAM 3 + filtering + post-process) |
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

Use [gsplat](https://github.com/nerfstudio-project/gsplat) for production rendering. It's 100-1000x faster than pure PyTorch thanks to custom CUDA kernels and tile-based rasterization.


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

## Disease Synthesis

Apply diseases to healthy renders using SDXL img2img + per-disease LoRAs trained on PlantSeg.

Diseases: `powdery_mildew`, `rust`, `leaf_spot`, `blight`, `chlorosis`

### Train LoRAs (on GPU VM)

Prerequisites: [Kaggle API credentials](https://www.kaggle.com/settings) → Create New Token

```bash
export KAGGLE_USERNAME='your_username'
export KAGGLE_API_TOKEN='your_api_key'

# Automated: downloads PlantSeg, installs kohya, trains all 5 LoRAs (~1.5 hours)
bash scripts/train_loras.sh
```

### Generate Diseased Images

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

## Validation

Train a classifier on synthetic data, test on real images:

```bash
bash scripts/run_experiment.sh
```

### Results (ResNet50, PlantSegV2 test set)

| Task | Accuracy |
|------|----------|
| Binary (healthy vs diseased) | **99.5%** |
| 5-way (disease classification) | 16.2% |

Binary classification proves the thesis: synthetic renders from Gaussian splats transfer to real photographs. 5-way classification needs more data or better disease synthesis.

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