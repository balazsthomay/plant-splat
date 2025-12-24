"""
SAM 2 video segmentation for plant isolation.

Uses video predictor with center point prompt on first frame,
then propagates mask to all frames. Much faster than per-image segmentation.
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def prepare_frames_dir(images_dir: Path, tmp_dir: Path) -> list[Path]:
    """Create temp directory with frames named for SAM 2 (00000.jpg, 00001.jpg, ...).

    Returns list of original frame paths in order.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Get original frames
    frames = sorted(images_dir.glob("*.jpg"))
    if not frames:
        frames = sorted(images_dir.glob("*.jpeg")) or sorted(images_dir.glob("*.png"))

    if not frames:
        raise ValueError(f"No frames found in {images_dir}")

    # Create symlinks with SAM 2 expected naming
    for i, frame in enumerate(frames):
        link_path = tmp_dir / f"{i:05d}.jpg"
        if link_path.exists():
            link_path.unlink()
        os.symlink(frame.resolve(), link_path)

    return frames


def segment_video(
    images_dir: Path,
    device: str = "cpu",  # SAM 2 requires CPU (MPS doesn't support float64)
    model_id: str = "facebook/sam2.1-hiera-large",
) -> int:
    """Segment all frames using video predictor with propagation.

    Args:
        images_dir: Directory containing image frames
        device: Device to run on (mps, cuda, cpu)
        model_id: Hugging Face model ID

    Returns:
        Number of masks generated
    """
    from sam2.build_sam import build_sam2_video_predictor_hf

    # Create temp dir with properly named frames
    tmp_dir = images_dir.parent / ".sam2_frames"
    original_frames = prepare_frames_dir(images_dir, tmp_dir)
    num_frames = len(original_frames)

    print(f"[segment] Loading SAM 2 video predictor on {device}...")
    predictor = build_sam2_video_predictor_hf(model_id, device=device)
    print(f"[segment] Processing {num_frames} frames with video propagation")

    # Initialize state with the frames directory
    inference_state = predictor.init_state(
        video_path=str(tmp_dir),
        offload_video_to_cpu=True,  # Save GPU memory
        async_loading_frames=True,
    )

    # Add point prompt at center of first frame
    # Assuming plant is roughly centered in the video
    first_frame = Image.open(original_frames[0])
    w, h = first_frame.size
    center_point = np.array([[w // 2, h // 2]], dtype=np.float32)
    center_label = np.array([1], dtype=np.int32)  # 1 = foreground

    print(f"[segment] Adding center point prompt on frame 0: ({w//2}, {h//2})")
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=center_point,
        labels=center_label,
    )

    # Propagate through all frames
    print("[segment] Propagating masks through video...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        if (out_frame_idx + 1) % 50 == 0:
            print(f"  [{out_frame_idx + 1}/{num_frames}] Propagated")

    # Save masks alongside original frames
    print("[segment] Saving masks...")
    for i, frame_path in enumerate(original_frames):
        if i in video_segments and 1 in video_segments[i]:
            mask = video_segments[i][1].squeeze()  # [H, W] boolean
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


# Keep old function name for compatibility with reconstruct.py
def segment_directory(images_dir: Path, pattern: str = "*.jpg", device: str = "cpu") -> int:
    """Wrapper for backward compatibility."""
    return segment_video(images_dir, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate segmentation masks using SAM 2 video predictor")
    parser.add_argument("images_dir", type=Path, help="Directory containing image frames")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda) - MPS not supported")
    parser.add_argument("--model", default="facebook/sam2.1-hiera-large", help="Model ID")

    args = parser.parse_args()

    if not args.images_dir.exists():
        print(f"Error: Directory not found: {args.images_dir}")
        exit(1)

    segment_video(args.images_dir, args.device, args.model)
