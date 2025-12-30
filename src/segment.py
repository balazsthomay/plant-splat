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
    batch_size: int = 10,
) -> int:
    """Segment all frames using SAM 3 video predictor with text prompt.

    Processes frames in small batches to fit in 8GB VRAM.

    Args:
        images_dir: Directory containing image frames
        prompt: Text prompt for semantic segmentation
        batch_size: Number of frames per batch (lower = less VRAM)

    Returns:
        Number of masks generated
    """
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3 requires CUDA. Run on a GPU VM.")

    from sam3.model_builder import build_sam3_video_predictor

    # Get frames
    frames = sorted(images_dir.glob("*.jpg"))
    if not frames:
        frames = sorted(images_dir.glob("*.jpeg")) or sorted(images_dir.glob("*.png"))
    if not frames:
        raise ValueError(f"No frames found in {images_dir}")

    num_frames = len(frames)
    print(f"[segment] Loading SAM 3 video predictor on CUDA...")
    predictor = build_sam3_video_predictor(gpus_to_use=[0])

    print(f"[segment] Processing {num_frames} frames in batches of {batch_size}")
    print(f"[segment] Text prompt: '{prompt}'")

    # Process in batches to manage VRAM
    for batch_start in range(0, num_frames, batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        batch_frames = frames[batch_start:batch_end]

        # Create temp directory for this batch
        tmp_dir = images_dir.parent / ".sam3_frames"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(batch_frames):
            link_path = tmp_dir / f"{i:05d}.jpg"
            os.symlink(frame.resolve(), link_path)

        # Start session for this batch
        response = predictor.handle_request(
            request=dict(type="start_session", resource_path=str(tmp_dir))
        )
        session_id = response["session_id"]

        # Add text prompt on frame 0
        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=prompt,
            )
        )

        # Propagate through batch
        batch_masks = {}
        for response in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            frame_idx = response["frame_index"]
            outputs = response.get("outputs", {})
            if outputs:
                masks = outputs.get("out_binary_masks", [])
                if masks is not None and len(masks) > 0:
                    combined = np.zeros_like(masks[0], dtype=bool)
                    for m in masks:
                        combined = combined | (m > 0.5)
                    batch_masks[frame_idx] = combined

        # Close session and free memory
        predictor.handle_request(request=dict(type="close_session", session_id=session_id))
        torch.cuda.empty_cache()

        # Save masks for this batch
        for i, frame_path in enumerate(batch_frames):
            if i in batch_masks:
                mask = batch_masks[i].squeeze()
                mask_uint8 = (mask * 255).astype(np.uint8)
            else:
                img = Image.open(frame_path)
                mask_uint8 = np.zeros((img.height, img.width), dtype=np.uint8)

            mask_path = frame_path.parent / f"{frame_path.stem}_mask.png"
            Image.fromarray(mask_uint8).save(mask_path)

        # Cleanup temp directory
        shutil.rmtree(tmp_dir)

        print(f"  [{batch_end}/{num_frames}] Processed")

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
