# Subtract Pot Masks and Rebuild Splat

Run these commands on the VM in order.

# 1. Segment the pot only                                                            
uv run src/segment.py data/colmap/mint/images/ --prompt "terracotta pot"

# Move pot masks aside                                                               
mkdir -p data/colmap/mint/pot_masks                                                  
mv data/colmap/mint/images/*_mask.png data/colmap/mint/pot_masks/                    
                                                                                       
# 2. Segment whole plant                                                             
uv run src/segment.py data/colmap/mint/images/ --prompt "potted plant"               
                                                                                       
# 3. Subtract pot from plant

## 1. Subtract pot from plant masks

```bash
uv run python -c "
from pathlib import Path
from PIL import Image
import numpy as np

images_dir = Path('data/colmap/mint/images')
pot_dir = Path('data/colmap/mint/pot_masks')

plant_masks = sorted(images_dir.glob('*_mask.png'))
print(f'Found {len(plant_masks)} plant masks')

for plant_mask_path in plant_masks:
    pot_mask_path = pot_dir / plant_mask_path.name

    plant = np.array(Image.open(plant_mask_path))
    pot = np.array(Image.open(pot_mask_path)) if pot_mask_path.exists() else np.zeros_like(plant)

    # Plant minus pot: keep plant pixels that aren't pot
    result = np.where(pot > 127, 0, plant).astype(np.uint8)

    Image.fromarray(result).save(plant_mask_path)

print('Done - masks updated')
"
```

## 2. Set environment variable

```bash
export OMP_NUM_THREADS=8
```

## 3. Re-filter points with new masks

```bash
uv run python -c "
from pathlib import Path
from src.filter_points import filter_points
filter_points(
    Path('data/colmap/mint/sparse/0'),
    Path('data/colmap/mint/images'),
    Path('data/colmap/mint/sparse_filtered/0'),
    min_visible_ratio=0.5
)
"
```

## 4. Re-run OpenSplat

```bash
/workspace/plant-splat/tools/OpenSplat/build/opensplat data/colmap/mint_plantonly -o data/splats/mint_raw.ply --num-iters 3000 -d 1
```

## 5. Post-process (geometric filtering)

```bash
uv run python -c "
from pathlib import Path
from src.filter_splat import filter_splat
filter_splat(Path('data/splats/mint_raw.ply'), Path('data/splats/mint_clean.ply'), 0.15, None, None, 92)
"
```

## 6. Download and verify

```bash
# From your Mac:
scp vast:/workspace/plant-splat/data/splats/mint_clean.ply ~/Downloads/
```

Open in [SuperSplat](https://superspl.at/editor) and check:
- Plant visible
- Pot removed
- No floating artifacts
