"""
COLMAP + OpenSplat reconstruction pipeline.

Converts video → frames → (optional SAM 3 masks) → COLMAP SfM → Gaussian splat.
"""

import subprocess
import sys
from pathlib import Path


# Tool paths (adjust if needed)
COLMAP_BIN = "/opt/homebrew/bin/colmap"
OPENSPLAT_BIN = Path(__file__).parent.parent / "tools/OpenSplat/build/opensplat"


def run(cmd: list[str], desc: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run subprocess with logging."""
    print(f"[reconstruct] {desc}")
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"  ✗ Failed:\n{result.stderr}")
        sys.exit(1)
    return result


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_skip: int = 10,
) -> int:
    """Extract frames from video at regular intervals.

    Args:
        video_path: Input video file
        output_dir: Directory to write frames
        frame_skip: Extract every Nth frame (default 10 = ~6fps from 60fps video)

    Returns:
        Number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"select=not(mod(n\\,{frame_skip}))",
        "-vsync", "vfr",
        "-q:v", "2",
        str(output_dir / "frame_%04d.jpg"),
    ], f"Extracting frames (every {frame_skip}th)")

    frames = list(output_dir.glob("frame_*.jpg"))
    print(f"  ✓ {len(frames)} frames extracted")
    return len(frames)


def segment_frames(images_dir: Path) -> int:
    """Run SAM 2 segmentation on all frames.

    Uses automatic mask generation, selects largest mask per frame.

    Args:
        images_dir: Directory containing images

    Returns:
        Number of masks generated
    """
    from src.segment import segment_directory
    return segment_directory(images_dir)


def run_colmap(project_dir: Path, single_camera: bool = True) -> Path:
    """Run full COLMAP SfM pipeline.

    Args:
        project_dir: Directory containing images/ subdirectory
        single_camera: Assume all images from same camera (typical for video)

    Returns:
        Path to sparse reconstruction (project_dir/sparse/0)
    """
    db_path = project_dir / "database.db"
    images_path = project_dir / "images"
    sparse_path = project_dir / "sparse"
    sparse_path.mkdir(exist_ok=True)

    # Feature extraction
    run([
        COLMAP_BIN, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_path),
        "--ImageReader.single_camera", "1" if single_camera else "0",
        "--ImageReader.camera_model", "OPENCV",
    ], "COLMAP feature extraction", timeout=1200)

    # Sequential matching (optimal for video)
    run([
        COLMAP_BIN, "sequential_matcher",
        "--database_path", str(db_path),
        "--SequentialMatching.overlap", "10",
        "--SequentialMatching.loop_detection", "1",
    ], "COLMAP sequential matching", timeout=1200)

    # Mapper (SfM)
    run([
        COLMAP_BIN, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_path),
        "--output_path", str(sparse_path),
    ], "COLMAP mapper (SfM)", timeout=1800)

    model_path = sparse_path / "0"
    if not model_path.exists():
        print("  ✗ COLMAP failed to produce reconstruction")
        sys.exit(1)

    print(f"  ✓ Reconstruction saved to {model_path}")
    return model_path


def run_opensplat(
    project_dir: Path,
    output_ply: Path,
    num_iters: int = 3000,
    downscale: int = 1,
) -> Path:
    """Train Gaussian splat using OpenSplat.

    Args:
        project_dir: COLMAP project directory
        output_ply: Output .ply file path
        num_iters: Training iterations (default 7000)
        downscale: Image downscale factor (default 2)

    Returns:
        Path to output .ply file
    """
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    run([
        str(OPENSPLAT_BIN),
        str(project_dir),
        "-o", str(output_ply),
        "--num-iters", str(num_iters),
        "-d", str(downscale),
    ], f"OpenSplat training ({num_iters} iters)", timeout=3600)

    if not output_ply.exists():
        print("  ✗ OpenSplat failed to produce output")
        sys.exit(1)

    size_mb = output_ply.stat().st_size / (1024 * 1024)
    print(f"  ✓ Splat saved: {output_ply} ({size_mb:.1f} MB)")
    return output_ply


def reconstruct(
    video_path: Path,
    output_dir: Path,
    name: str | None = None,
    frame_skip: int = 10,
    num_iters: int = 3000,
    downscale: int = 1,
    segment: bool = False,
) -> Path:
    """Full pipeline: video → Gaussian splat.

    Args:
        video_path: Input video file
        output_dir: Base output directory
        name: Project name (default: video filename stem)
        frame_skip: Extract every Nth frame
        num_iters: OpenSplat training iterations
        downscale: Image downscale factor
        segment: If True, run SAM 2 segmentation for background removal

    Returns:
        Path to output .ply file
    """
    name = name or video_path.stem
    project_dir = output_dir / "colmap" / name
    splat_path = output_dir / "splats" / f"{name}.ply"

    print(f"\n{'='*60}")
    print(f"Reconstructing: {video_path.name} → {splat_path.name}")
    if segment:
        print("  (with SAM 2 segmentation)")
    print(f"{'='*60}\n")

    # Step 1: Extract frames
    extract_frames(video_path, project_dir / "images", frame_skip)

    # Step 2: Generate masks (optional)
    if segment:
        segment_frames(project_dir / "images")

    # Step 3: COLMAP SfM (uses original images for feature matching)
    run_colmap(project_dir)

    # Step 4: OpenSplat (uses masks if present)
    run_opensplat(project_dir, splat_path, num_iters, downscale)

    print(f"\n{'='*60}")
    print(f"✓ Done! Output: {splat_path}")
    print(f"{'='*60}\n")

    return splat_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video → Gaussian Splat pipeline")
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("-o", "--output", type=Path, default=Path("data"), help="Output directory")
    parser.add_argument("-n", "--name", help="Project name (default: video filename)")
    parser.add_argument("--frame-skip", type=int, default=10, help="Extract every Nth frame")
    parser.add_argument("--iters", type=int, default=3000, help="Training iterations")
    parser.add_argument("--downscale", type=int, default=1, help="Image downscale factor")
    parser.add_argument("--segment", action="store_true", help="Enable SAM 2 segmentation for background removal")

    args = parser.parse_args()

    reconstruct(
        args.video,
        args.output,
        args.name,
        args.frame_skip,
        args.iters,
        args.downscale,
        args.segment,
    )
