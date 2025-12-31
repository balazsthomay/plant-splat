"""
SAM 3 video segmentation for plant isolation.

Uses video predictor with text prompt for semantic segmentation.
Text prompts allow precise control (e.g., "potted plant without pot").

Requires:
- CUDA (Triton dependency)
- 16GB+ VRAM for full video processing (text prompts need temporal context)
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
    """Segment all frames using SAM 3 video predictor with text prompt.

    Processes entire video in ONE session to maintain temporal context.
    This is required for text prompts like "without pot" to work correctly.

    Requires 16GB+ VRAM. For smaller GPUs, use fewer frames or lower resolution.

    Args:
        images_dir: Directory containing image frames
        prompt: Text prompt for semantic segmentation

    Returns:
        Number of masks generated
    """
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3 requires CUDA. Run on a GPU VM with 16GB+ VRAM.")

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

    print(f"[segment] Processing {num_frames} frames with text prompt: '{prompt}'")

    # Create temp directory with ALL frames (SAM 3 expects sequential naming)
    tmp_dir = images_dir.parent / ".sam3_frames"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        link_path = tmp_dir / f"{i:05d}.jpg"
        os.symlink(frame.resolve(), link_path)

    # Start ONE session for entire video
    print("[segment] Starting video session...")
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(tmp_dir))
    )
    session_id = response["session_id"]

    # Add text prompt on frame 0 (propagates to all frames)
    print(f"[segment] Adding text prompt on frame 0: '{prompt}'")
    predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt,
        )
    )

    # Propagate through ALL frames in single pass
    print("[segment] Propagating masks through video...")
    video_segments = {}
    for response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        frame_idx = response["frame_index"]

        # SAM 3 returns masks for all detected instances
        # Merge all masks (union of all detected objects matching prompt)
        outputs = response.get("outputs", {})
        if outputs:
            masks = outputs.get("out_binary_masks", [])
            if masks is not None and len(masks) > 0:
                combined = np.zeros_like(masks[0], dtype=bool)
                for m in masks:
                    combined = combined | (m > 0.5)
                video_segments[frame_idx] = combined

        if (frame_idx + 1) % 50 == 0:
            print(f"  [{frame_idx + 1}/{num_frames}] Propagated")

    # Close session
    predictor.handle_request(request=dict(type="close_session", session_id=session_id))

    # Save masks alongside original frames
    print("[segment] Saving masks...")
    for i, frame_path in enumerate(frames):
        if i in video_segments:
            mask = video_segments[i].squeeze()  # [H, W] boolean
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            # No mask for this frame, use empty
            img = Image.open(frame_path)
            mask_uint8 = np.zeros((img.height, img.width), dtype=np.uint8)

        mask_path = frame_path.parent / f"{frame_path.stem}_mask.png"
        Image.fromarray(mask_uint8).save(mask_path)

    # Cleanup temp directory
    shutil.rmtree(tmp_dir)

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
