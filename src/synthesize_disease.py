"""
Synthesize diseased plant images from healthy renders.

Reads healthy images from data/synthetic/, applies disease using SD 1.5 inpainting,
outputs to data/synthetic_diseased/.

Usage:
    # Random diseases, random severity
    uv run src/synthesize_disease.py data/synthetic/ -o data/synthetic_diseased/

    # Specific disease type
    uv run src/synthesize_disease.py data/synthetic/ -o data/synthetic_diseased/ \
        --disease powdery_mildew --severity-min 0.4 --severity-max 0.8

    # Fast mode (fewer steps, for testing)
    uv run src/synthesize_disease.py data/synthetic/ -o data/synthetic_diseased/ \
        --steps 15 -n 10
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from disease import DISEASE_CONFIGS, DiseaseAugmentor, DiseaseType, get_device


def load_annotations(input_dir: Path) -> dict:
    """Load annotations.json from input directory."""
    ann_path = input_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"No annotations.json found in {input_dir}")
    with open(ann_path) as f:
        return json.load(f)


def synthesize_diseases(
    input_dir: Path,
    output_dir: Path,
    n_images: int | None = None,
    disease_type: str | None = None,
    severity_min: float = 0.3,
    severity_max: float = 0.7,
    pattern: str = "patchy",
    num_inference_steps: int = 30,
    seed: int | None = None,
    device: str | None = None,
) -> None:
    """Synthesize diseased images from healthy dataset.

    Args:
        input_dir: Directory with healthy images (data/synthetic/)
        output_dir: Output directory for diseased images
        n_images: Number of images to process (None = all)
        disease_type: Specific disease or None for random
        severity_min: Minimum severity (0-1)
        severity_max: Maximum severity (0-1)
        pattern: Change map pattern ("patchy", "edge", "uniform")
        num_inference_steps: Diffusion steps
        seed: Random seed for reproducibility
        device: PyTorch device (auto-detect if None)
    """
    # Load source annotations
    source_ann = load_annotations(input_dir)
    images_data = source_ann.get("images", [])

    if not images_data:
        raise ValueError(f"No images found in {input_dir}/annotations.json")

    # Limit images if requested
    if n_images is not None and n_images < len(images_data):
        if seed is not None:
            random.seed(seed)
        images_data = random.sample(images_data, n_images)

    # Setup output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "masks").mkdir(exist_ok=True)
    (output_dir / "disease_masks").mkdir(exist_ok=True)

    # Determine diseases to use
    if disease_type:
        diseases = [DiseaseType(disease_type)]
    else:
        diseases = list(DiseaseType)

    # Initialize augmentor
    device = device or get_device()
    print(f"[synthesize] Device: {device}")
    print(f"[synthesize] Diseases: {[d.value for d in diseases]}")
    print(f"[synthesize] Severity: {severity_min:.2f} - {severity_max:.2f}")
    print(f"[synthesize] Pattern: {pattern}")
    print(f"[synthesize] Steps: {num_inference_steps}")
    print(f"[synthesize] Processing {len(images_data)} images...")

    augmentor = DiseaseAugmentor(
        device=device,
        num_inference_steps=num_inference_steps,
    )

    # Process images
    output_annotations = []
    rng = random.Random(seed)

    for i, img_data in enumerate(images_data):
        # Load image and mask
        img_path = input_dir / img_data["image"]
        mask_path = input_dir / img_data["mask"]

        if not img_path.exists() or not mask_path.exists():
            print(f"[synthesize] Skipping {img_data['image']} (file not found)")
            continue

        rgb = np.array(Image.open(img_path).convert("RGB"))
        alpha = np.array(Image.open(mask_path).convert("L"))

        # Random disease and severity
        disease = rng.choice(diseases)
        severity = rng.uniform(severity_min, severity_max)
        img_seed = rng.randint(0, 2**31) if seed is not None else None

        print(
            f"[synthesize] [{i+1}/{len(images_data)}] "
            f"{disease.value} severity={severity:.2f}",
            end="",
            flush=True,
        )

        # Apply disease
        diseased_rgb, change_map, disease_meta = augmentor.apply(
            rgb=rgb,
            alpha=alpha,
            disease_type=disease,
            severity=severity,
            seed=img_seed,
            pattern=pattern,
        )

        print(" ✓")

        # Save outputs
        out_id = f"{i:04d}"
        out_img_path = output_dir / "images" / f"{out_id}.png"
        out_mask_path = output_dir / "masks" / f"{out_id}_mask.png"
        out_disease_mask_path = output_dir / "disease_masks" / f"{out_id}_disease.png"

        Image.fromarray(diseased_rgb).save(out_img_path)
        Image.fromarray(alpha).save(out_mask_path)
        Image.fromarray(change_map).save(out_disease_mask_path)

        # Build annotation (extend source with disease info)
        ann = {
            "id": i,
            "image": f"images/{out_id}.png",
            "mask": f"masks/{out_id}_mask.png",
            "disease_mask": f"disease_masks/{out_id}_disease.png",
            "width": img_data.get("width", rgb.shape[1]),
            "height": img_data.get("height", rgb.shape[0]),
            "healthy_source": str(img_path),
            "disease": disease_meta,
        }

        # Carry over camera, lighting, background, bbox from source
        for key in ["camera", "lighting", "background", "bbox"]:
            if key in img_data:
                ann[key] = img_data[key]

        output_annotations.append(ann)

    # Save annotations
    output_ann = {
        "source_dir": str(input_dir),
        "n_images": len(output_annotations),
        "diseases": [d.value for d in diseases],
        "severity_range": [severity_min, severity_max],
        "pattern": pattern,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "images": output_annotations,
    }

    ann_path = output_dir / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(output_ann, f, indent=2)

    print(f"\n[synthesize] Done! Generated {len(output_annotations)} diseased images")
    print(f"[synthesize] Output: {output_dir}")
    print(f"[synthesize] Annotations: {ann_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize diseased plant images from healthy renders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random diseases, default severity
  uv run src/synthesize_disease.py data/synthetic/ -o data/synthetic_diseased/

  # Specific disease with custom severity
  uv run src/synthesize_disease.py data/synthetic/ -o data/synthetic_diseased/ \\
      --disease powdery_mildew --severity-min 0.4 --severity-max 0.8

  # Fast test run (10 images, 15 steps)
  uv run src/synthesize_disease.py data/synthetic/ -o data/synthetic_diseased/ \\
      -n 10 --steps 15
        """,
    )
    parser.add_argument("input", type=Path, help="Input directory with healthy images")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("data/synthetic_diseased"),
        help="Output directory (default: data/synthetic_diseased)"
    )
    parser.add_argument(
        "-n", "--n-images", type=int, default=None,
        help="Number of images to process (default: all)"
    )
    parser.add_argument(
        "--disease", type=str, default=None,
        choices=[d.value for d in DiseaseType],
        help="Specific disease type (default: random)"
    )
    parser.add_argument(
        "--severity-min", type=float, default=0.3,
        help="Minimum severity (default: 0.3)"
    )
    parser.add_argument(
        "--severity-max", type=float, default=0.7,
        help="Maximum severity (default: 0.7)"
    )
    parser.add_argument(
        "--pattern", type=str, default="patchy",
        choices=["patchy", "edge", "uniform"],
        help="Disease distribution pattern (default: patchy)"
    )
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Diffusion inference steps (default: 30, use 15-20 for faster)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda, mps, cpu (default: auto-detect)"
    )

    args = parser.parse_args()

    # Show device info
    detected = get_device()
    print(f"[synthesize] Auto-detected device: {detected}")

    synthesize_diseases(
        input_dir=args.input,
        output_dir=args.output,
        n_images=args.n_images,
        disease_type=args.disease,
        severity_min=args.severity_min,
        severity_max=args.severity_max,
        pattern=args.pattern,
        num_inference_steps=args.steps,
        seed=args.seed,
        device=args.device,
    )
