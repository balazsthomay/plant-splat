"""
Synthetic dataset generator from Gaussian splats.

Generates training images with:
- Varied camera viewpoints (orbit around plant)
- Lighting variation (color temperature, intensity, contrast)
- Background variation (solid colors or image folder)
- Alpha masks (automatic from rendering)
- Bounding boxes (from alpha extent)

Backends:
- CUDA + gsplat: ~60fps (use this for production)
- MPS/CPU + PyTorch: ~2min/frame (fallback for development)

Usage:
    uv run src/generate_dataset.py data/splats/mint3_clean.ply -n 100 -o data/synthetic/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from augment import BackgroundAugmentor, LightingAugmentor, composite
from render import (
    Camera,
    Gaussians,
    create_orbit_camera,
    load_cameras,
    load_splat,
    render as render_pytorch,
)

# Try importing gsplat (CUDA only)
GSPLAT_AVAILABLE = False
try:
    from gsplat import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    pass


def get_best_device() -> str:
    """Auto-detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def camera_to_gsplat(cam: Camera) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Camera to gsplat's viewmat and K format.

    Returns:
        viewmat: [4, 4] world-to-camera transformation
        K: [3, 3] intrinsic matrix
    """
    device = cam.R.device

    # Intrinsic matrix
    K = torch.tensor([
        [cam.fx, 0, cam.cx],
        [0, cam.fy, cam.cy],
        [0, 0, 1],
    ], device=device, dtype=torch.float32)

    # View matrix: world-to-camera
    # viewmat = [R | -R @ t]
    #           [0 |   1   ]
    R = cam.R  # [3, 3]
    t = cam.t  # [3] camera position in world

    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = -R @ t  # Translation in camera space

    return viewmat, K


def render_gsplat(
    gaussians: Gaussians,
    camera: Camera,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render using gsplat (CUDA only, ~1000x faster).

    Returns:
        rgb: [H, W, 3] in [0, 1]
        alpha: [H, W] in [0, 1]
    """
    device = gaussians.means.device

    # Convert camera
    viewmat, K = camera_to_gsplat(camera)

    # Prepare Gaussian parameters for gsplat
    # gsplat expects: scales (not log), opacities in [0,1]
    means = gaussians.means  # [N, 3]
    quats = gaussians.rotations  # [N, 4]
    scales = torch.exp(gaussians.scales)  # [N, 3] - gsplat wants actual scales
    opacities = torch.sigmoid(gaussians.opacities)  # [N] - gsplat wants [0,1]

    # Colors from SH (use DC only for now)
    SH_C0 = 0.28209479177387814
    colors = torch.clamp(SH_C0 * gaussians.sh_dc + 0.5, 0, 1)  # [N, 3]

    # Render
    renders, alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        width=camera.width,
        height=camera.height,
        packed=False,
    )

    # Output: [1, H, W, 3], [1, H, W, 1]
    rgb = renders[0]  # [H, W, 3]
    alpha = alphas[0, :, :, 0]  # [H, W]

    return rgb, alpha


def compute_bounding_box(alpha: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int, int] | None:
    """Compute bounding box from alpha mask.

    Args:
        alpha: [H, W] alpha values in [0, 1]
        threshold: Minimum alpha to consider as foreground

    Returns:
        (x_min, y_min, x_max, y_max) or None if no foreground
    """
    mask = alpha > threshold
    if not mask.any():
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return int(x_min), int(y_min), int(x_max), int(y_max)




def generate_orbit_cameras(
    center: torch.Tensor,
    distance: float,
    fx: float,
    fy: float,
    width: int,
    height: int,
    n_azimuth: int = 36,
    n_elevation: int = 5,
    elevation_range: tuple[float, float] = (-15, 45),
) -> list[tuple[Camera, dict]]:
    """Generate cameras on an orbit around center.

    Args:
        center: [3] center point
        distance: Distance from center
        fx, fy: Focal lengths
        width, height: Image dimensions
        n_azimuth: Number of azimuth angles (around Y axis)
        n_elevation: Number of elevation angles
        elevation_range: (min, max) elevation in degrees

    Returns:
        List of (Camera, metadata_dict) tuples
    """
    cameras = []

    azimuths = np.linspace(0, 360, n_azimuth, endpoint=False)
    elevations = np.linspace(elevation_range[0], elevation_range[1], n_elevation)

    for az in azimuths:
        for el in elevations:
            cam = create_orbit_camera(
                center=center,
                distance=distance,
                azimuth=float(az),
                elevation=float(el),
                fx=fx,
                fy=fy,
                width=width,
                height=height,
            )
            meta = {
                "azimuth": float(az),
                "elevation": float(el),
                "distance": float(distance),
            }
            cameras.append((cam, meta))

    return cameras


def generate_dataset(
    splat_path: Path,
    output_dir: Path,
    n_views: int = 100,
    downscale: int = 2,
    device: str | None = None,
    bg_dir: Path | None = None,
    bg_mode: str = "varied",
    lighting: bool = True,
    cameras_json: Path | None = None,
) -> None:
    """Generate synthetic dataset from Gaussian splat.

    Args:
        splat_path: Path to PLY file
        output_dir: Output directory for images and annotations
        n_views: Number of views to generate
        downscale: Image downscale factor
        device: PyTorch device (auto-detect if None)
        bg_dir: Directory with background images (optional)
        bg_mode: Background mode: "varied", "white", "black", "random"
        lighting: Apply lighting augmentation
        cameras_json: Path to cameras.json for reference intrinsics
    """
    # Auto-detect device
    if device is None:
        device = get_best_device()

    # Determine backend
    use_gsplat = (device == "cuda" and GSPLAT_AVAILABLE)
    backend = "gsplat" if use_gsplat else "pytorch"

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    # Setup augmentors
    bg_augmentor = BackgroundAugmentor(image_dir=bg_dir, color_mode=bg_mode)
    light_augmentor = LightingAugmentor() if lighting else None

    print(f"[generate] Device: {device}, Backend: {backend}")
    print(f"[generate] Background: {bg_mode}" + (f" + {len(bg_augmentor.images)} images" if bg_augmentor.images else ""))
    print(f"[generate] Lighting: {'enabled' if lighting else 'disabled'}")
    print(f"[generate] Loading splat from {splat_path}...")
    gaussians = load_splat(splat_path, device=device)
    n_gaussians = gaussians.means.shape[0]
    print(f"[generate] Loaded {n_gaussians:,} Gaussians")

    # Get reference camera for intrinsics
    if cameras_json is None:
        cameras_json = splat_path.parent / "cameras.json"

    if cameras_json.exists():
        ref_cameras = load_cameras(cameras_json, device=device)
        ref = ref_cameras[0]
        fx = ref.fx / downscale
        fy = ref.fy / downscale
        width = ref.width // downscale
        height = ref.height // downscale
    else:
        # Default intrinsics
        fx = fy = 1500 / downscale
        width = 1080 // downscale
        height = 1920 // downscale

    # Compute scene center and distance
    center = gaussians.means.mean(dim=0)
    if cameras_json.exists():
        # Use average distance of training cameras
        distances = [torch.norm(c.t - center).item() for c in ref_cameras]
        base_distance = np.mean(distances)
    else:
        base_distance = 1.0

    print(f"[generate] Scene center: {center.tolist()}")
    print(f"[generate] Base distance: {base_distance:.3f}")
    print(f"[generate] Output resolution: {width}x{height}")

    # Determine orbit parameters from n_views
    # Aim for square-ish grid: n_az * n_el ≈ n_views
    n_el = max(3, int(np.sqrt(n_views / 6)))
    n_az = max(6, n_views // n_el)
    actual_views = n_az * n_el

    print(f"[generate] Generating {actual_views} views ({n_az} azimuth × {n_el} elevation)")

    # Generate cameras
    cameras = generate_orbit_cameras(
        center=center,
        distance=base_distance,
        fx=fx,
        fy=fy,
        width=width,
        height=height,
        n_azimuth=n_az,
        n_elevation=n_el,
        elevation_range=(-15, 45),
    )

    # Render all views
    annotations = []

    for i, (cam, meta) in enumerate(cameras):
        print(f"[generate] [{i+1}/{len(cameras)}] az={meta['azimuth']:.0f}° el={meta['elevation']:.0f}°", end="")

        # Render with appropriate backend
        if use_gsplat:
            rgb_tensor, alpha_tensor = render_gsplat(gaussians, cam)
        else:
            rgb_tensor, alpha_tensor, _ = render_pytorch(gaussians, cam, bg_color=(0, 0, 0))

        rgb = (rgb_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        alpha = (alpha_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # Apply lighting augmentation (before compositing to affect plant only)
        light_meta = {}
        if light_augmentor:
            rgb, light_meta = light_augmentor.apply(rgb)

        # Get background and composite
        background, bg_meta = bg_augmentor.get_background(width, height)
        final_rgb = composite(rgb, alpha, background)

        print(" ✓")

        # Compute bounding box
        bbox = compute_bounding_box(alpha / 255.0, threshold=0.5)

        # Save mask
        mask_path = masks_dir / f"{i:04d}_mask.png"
        Image.fromarray(alpha).save(mask_path)

        # Save image
        img_path = images_dir / f"{i:04d}.png"
        Image.fromarray(final_rgb).save(img_path)

        # Build annotation
        ann = {
            "id": i,
            "image": f"images/{i:04d}.png",
            "mask": f"masks/{i:04d}_mask.png",
            "width": width,
            "height": height,
            "camera": meta,
            "lighting": light_meta,
            "background": bg_meta,
        }
        if bbox:
            ann["bbox"] = {
                "x_min": bbox[0],
                "y_min": bbox[1],
                "x_max": bbox[2],
                "y_max": bbox[3],
                "width": bbox[2] - bbox[0],
                "height": bbox[3] - bbox[1],
            }
        annotations.append(ann)

    # Save annotations
    ann_path = output_dir / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump({
            "splat": str(splat_path),
            "n_gaussians": n_gaussians,
            "n_images": len(annotations),
            "downscale": downscale,
            "images": annotations,
        }, f, indent=2)

    print(f"\n[generate] Done! Generated {len(annotations)} images")
    print(f"[generate] Output: {output_dir}")
    print(f"[generate] Annotations: {ann_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset from Gaussian splat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (varied backgrounds + lighting)
  uv run src/generate_dataset.py data/splats/mint3_clean.ply -n 100

  # With background images
  uv run src/generate_dataset.py data/splats/mint3_clean.ply -n 100 --bg-dir data/backgrounds/

  # White backgrounds, no lighting variation
  uv run src/generate_dataset.py data/splats/mint3_clean.ply -n 100 --bg-mode white --no-lighting
        """,
    )
    parser.add_argument("splat", type=Path, help="Path to PLY file")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/synthetic"), help="Output directory")
    parser.add_argument("-n", "--n-views", type=int, default=100, help="Number of views to generate")
    parser.add_argument("-d", "--downscale", type=int, default=1, help="Image downscale factor (1=full, 2=half)")
    parser.add_argument("--device", default=None, help="Device (auto-detect: cuda > mps > cpu)")
    parser.add_argument("--bg-dir", type=Path, help="Directory with background images")
    parser.add_argument("--bg-mode", default="varied", choices=["varied", "white", "black", "random"],
                        help="Background color mode (default: varied greenhouse-like colors)")
    parser.add_argument("--no-lighting", action="store_true", help="Disable lighting augmentation")
    parser.add_argument("--cameras-json", type=Path, help="Path to cameras.json for reference")

    args = parser.parse_args()

    # Show backend info
    detected = get_best_device()
    gsplat_status = "available" if GSPLAT_AVAILABLE else "not installed"
    print(f"[generate] Auto-detected device: {detected}, gsplat: {gsplat_status}")

    generate_dataset(
        splat_path=args.splat,
        output_dir=args.output,
        n_views=args.n_views,
        downscale=args.downscale,
        device=args.device,
        bg_dir=args.bg_dir,
        bg_mode=args.bg_mode,
        lighting=not args.no_lighting,
        cameras_json=args.cameras_json,
    )
