"""
SAM 3 video segmentation for plant isolation.

Uses video predictor with text prompt for semantic segmentation.
Text prompts allow precise control (e.g., "potted plant without pot").

Requires CUDA (Triton dependency).
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_PROMPT = "potted plant without pot"


def segment_video(
    images_dir: Path,
    prompt: str = DEFAULT_PROMPT,
) -> int:
    """Segment all frames using SAM 3 image predictor with text prompt.

    Uses per-frame processing to fit in 8GB VRAM.

    Args:
        images_dir: Directory containing image frames
        prompt: Text prompt for semantic segmentation

    Returns:
        Number of masks generated
    """
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3 requires CUDA. Run on a GPU VM.")

    from sam3.model_builder import build_sam3_image_predictor

    # Get frames
    frames = sorted(images_dir.glob("*.jpg"))
    if not frames:
        frames = sorted(images_dir.glob("*.jpeg")) or sorted(images_dir.glob("*.png"))
    if not frames:
        raise ValueError(f"No frames found in {images_dir}")

    num_frames = len(frames)
    print(f"[segment] Loading SAM 3 image predictor on CUDA...")
    predictor = build_sam3_image_predictor()

    print(f"[segment] Processing {num_frames} frames with text prompt: '{prompt}'")

    # Process each frame individually (lower memory than video predictor)
    for i, frame_path in enumerate(frames):
        # Load image
        img = Image.open(frame_path).convert("RGB")
        img_np = np.array(img)

        # Set image and get text-prompted mask
        predictor.set_image(img_np)
        masks, scores, _ = predictor.predict(text=prompt)

        # Clear GPU memory between frames
        torch.cuda.empty_cache()

        # Combine all masks (union)
        if masks is not None and len(masks) > 0:
            combined = np.zeros(masks[0].shape, dtype=bool)
            for m in masks:
                combined = combined | (m > 0.5)
            mask_uint8 = (combined.squeeze() * 255).astype(np.uint8)
        else:
            mask_uint8 = np.zeros((img.height, img.width), dtype=np.uint8)

        # Save mask
        mask_path = frame_path.parent / f"{frame_path.stem}_mask.png"
        Image.fromarray(mask_uint8).save(mask_path)

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{num_frames}] Processed")

    print(f"[segment] ✓ Generated {num_frames} masks")
    return num_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate segmentation masks using SAM 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default prompt
  uv run src/segment.py data/colmap/mint/images/

  # Custom prompt
  uv run src/segment.py data/colmap/mint/images/ --prompt "plant leaves"
        """,
    )
    parser.add_argument("images_dir", type=Path, help="Directory containing image frames")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Text prompt for segmentation (default: '{DEFAULT_PROMPT}')",
    )

    args = parser.parse_args()

    if not args.images_dir.exists():
        print(f"Error: Directory not found: {args.images_dir}")
        exit(1)

    segment_video(args.images_dir, args.prompt)
