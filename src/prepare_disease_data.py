"""
Prepare PlantSeg dataset for LoRA training.

Extracts images by disease category, creates caption files with trigger words,
and structures data for kohya-ss training.

Usage:
    # Single merged LoRA (recommended)
    uv run src/prepare_disease_data.py --plantseg data/plantsegv2 --output data/disease_training --merged

    # Separate folders per disease (5 LoRAs)
    uv run src/prepare_disease_data.py --plantseg data/plantsegv2 --output data/disease_training
"""

import argparse
import csv
import shutil
from pathlib import Path


# Map our disease types to PlantSeg disease names (partial matches)
DISEASE_MAPPING = {
    "powdery_mildew": [
        "powdery mildew",
    ],
    "rust": [
        "rust",  # matches: wheat stripe rust, bean rust, corn rust, apple rust, coffee leaf rust, etc.
    ],
    "leaf_spot": [
        "leaf spot",
        "angular leaf spot",
        "frog eye leaf spot",
    ],
    "blight": [
        "blight",  # matches: early blight, late blight, leaf blight
        "scab",    # similar necrotic appearance
    ],
    "chlorosis": [
        "yellowing",
        "mosaic",      # causes yellowing patterns
        "greening",    # citrus greening causes yellow
        "leaf curl",   # often causes yellowing
    ],
}

# Trigger words for DreamBooth training
TRIGGER_WORDS = {
    "powdery_mildew": "sks_mildew",
    "rust": "sks_rust",
    "leaf_spot": "sks_spot",
    "blight": "sks_blight",
    "chlorosis": "sks_chlorosis",
}

# Caption templates per disease
CAPTION_TEMPLATES = {
    "powdery_mildew": "a {trigger} plant disease, white powdery fungal coating on leaf surface",
    "rust": "a {trigger} plant disease, orange-brown rust pustules and spores on leaf",
    "leaf_spot": "a {trigger} plant disease, brown circular necrotic lesions with yellow halos",
    "blight": "a {trigger} plant disease, dark brown-black necrotic tissue decay",
    "chlorosis": "a {trigger} plant disease, yellow discoloration and chlorotic leaves",
}


def load_metadata(plantseg_dir: Path) -> list[dict]:
    """Load PlantSeg metadata CSV."""
    csv_path = plantseg_dir / "Metadatav2.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata not found: {csv_path}")

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def match_disease(plantseg_disease: str, target_disease: str) -> bool:
    """Check if a PlantSeg disease name matches our target disease category."""
    plantseg_lower = plantseg_disease.lower()
    patterns = DISEASE_MAPPING.get(target_disease, [])
    return any(pattern in plantseg_lower for pattern in patterns)


def prepare_disease_data(
    plantseg_dir: Path,
    output_dir: Path,
    diseases: list[str] | None = None,
    max_per_disease: int = 100,
    include_masks: bool = True,
    merged: bool = False,
) -> dict[str, int]:
    """
    Prepare disease images for LoRA training.

    Args:
        plantseg_dir: Path to PlantSeg dataset
        output_dir: Output directory for training data
        diseases: List of disease types to prepare (None = all)
        max_per_disease: Maximum images per disease type
        include_masks: Whether to copy mask files
        merged: If True, output all diseases to single folder (1 LoRA)

    Returns:
        Dict mapping disease type to number of images prepared
    """
    if diseases is None:
        diseases = list(DISEASE_MAPPING.keys())

    metadata = load_metadata(plantseg_dir)
    print(f"[prepare] Loaded {len(metadata)} entries from metadata")
    print(f"[prepare] Mode: {'merged (1 LoRA)' if merged else 'separate (5 LoRAs)'}")

    # Group images by our disease categories
    disease_images: dict[str, list[dict]] = {d: [] for d in diseases}

    for entry in metadata:
        plantseg_disease = entry.get("Disease", "")
        for disease in diseases:
            if match_disease(plantseg_disease, disease):
                disease_images[disease].append(entry)
                break  # Only assign to first matching category

    # Report findings
    print("\n[prepare] Disease matches found:")
    for disease, images in disease_images.items():
        print(f"  {disease}: {len(images)} images")

    # Prepare output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}

    # For merged mode, use single output directory
    if merged:
        merged_dir = output_dir / "all_diseases"
        merged_dir.mkdir(exist_ok=True)
        global_idx = 0

    for disease in diseases:
        images = disease_images[disease][:max_per_disease]
        if not images:
            print(f"\n[prepare] Skipping {disease} - no images found")
            continue

        # Output directory: merged or per-disease
        if merged:
            out_dir = merged_dir
        else:
            out_dir = output_dir / disease
            out_dir.mkdir(exist_ok=True)

        trigger = TRIGGER_WORDS[disease]
        caption = CAPTION_TEMPLATES[disease].format(trigger=trigger)

        print(f"\n[prepare] Processing {disease}: {len(images)} images")

        local_count = 0
        for i, entry in enumerate(images):
            img_name = entry["Name"]
            split = entry["Split"].lower()

            # Determine source paths based on split
            if split == "training":
                split_folder = "train"
            elif split == "validation":
                split_folder = "val"
            else:
                split_folder = "test"

            src_img = plantseg_dir / "images" / split_folder / img_name
            src_mask = plantseg_dir / "annotations" / split_folder / entry["Label file"]

            if not src_img.exists():
                print(f"  [skip] Image not found: {src_img}")
                continue

            # Output paths
            if merged:
                out_base = f"{global_idx:04d}_{disease}"
                global_idx += 1
            else:
                out_base = f"{disease}_{i:04d}"

            out_img = out_dir / f"{out_base}.jpg"
            out_txt = out_dir / f"{out_base}.txt"
            out_mask = out_dir / f"{out_base}_mask.png"

            # Copy image
            shutil.copy2(src_img, out_img)

            # Write caption
            out_txt.write_text(caption)

            # Copy mask if requested and exists
            if include_masks and src_mask.exists():
                shutil.copy2(src_mask, out_mask)

            local_count += 1

        counts[disease] = local_count
        if merged:
            print(f"  → Added {local_count} images to {merged_dir}")
        else:
            print(f"  → Saved {local_count} images to {out_dir}")

    if merged:
        total = sum(counts.values())
        print(f"\n[prepare] Merged dataset: {total} images in {merged_dir}")

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PlantSeg data for LoRA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single merged LoRA (recommended)
  uv run src/prepare_disease_data.py --plantseg data/plantsegv2 --output data/disease_training --merged

  # Separate folders for 5 LoRAs
  uv run src/prepare_disease_data.py --plantseg data/plantsegv2 --output data/disease_training

  # More images per disease
  uv run src/prepare_disease_data.py --plantseg data/plantsegv2 --output data/disease_training --merged --max 200
        """,
    )
    parser.add_argument(
        "--plantseg", type=Path, required=True,
        help="Path to PlantSeg dataset directory"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("data/disease_training"),
        help="Output directory for training data"
    )
    parser.add_argument(
        "--merged", action="store_true",
        help="Merge all diseases into single folder (1 LoRA, recommended)"
    )
    parser.add_argument(
        "--disease", type=str, default=None,
        choices=list(DISEASE_MAPPING.keys()),
        help="Prepare only this disease type (default: all)"
    )
    parser.add_argument(
        "--max", type=int, default=100,
        help="Maximum images per disease type (default: 100)"
    )
    parser.add_argument(
        "--no-masks", action="store_true",
        help="Don't copy mask files"
    )

    args = parser.parse_args()

    diseases = [args.disease] if args.disease else None

    counts = prepare_disease_data(
        plantseg_dir=args.plantseg,
        output_dir=args.output,
        diseases=diseases,
        max_per_disease=args.max,
        include_masks=not args.no_masks,
        merged=args.merged,
    )

    print("\n[prepare] Done!")
    print(f"[prepare] Output: {args.output}")
    total = sum(counts.values())
    print(f"[prepare] Total: {total} images across {len(counts)} disease types")


if __name__ == "__main__":
    main()
