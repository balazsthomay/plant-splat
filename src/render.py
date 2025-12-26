"""
Pure PyTorch Gaussian Splat Renderer.

Renders 3D Gaussian splats from arbitrary camera viewpoints.
Works on MPS/CPU (no CUDA required). Really slow but correct. I use this for testing/debugging.

Usage:
    uv run src/render.py data/splats/mint3_clean.ply --camera 0 --output test.png
    uv run src/render.py data/splats/mint3_clean.ply --orbit --output orbit.png
"""

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass
class Gaussians:
    """Container for Gaussian splat parameters."""
    means: torch.Tensor      # [N, 3] positions
    scales: torch.Tensor     # [N, 3] log-scales
    rotations: torch.Tensor  # [N, 4] quaternions (w, x, y, z)
    opacities: torch.Tensor  # [N] sigmoid-pre-activated opacities
    sh_dc: torch.Tensor      # [N, 3] DC spherical harmonic coefficients
    sh_rest: torch.Tensor    # [N, 45] higher-order SH (optional, for view-dependent color)


@dataclass
class Camera:
    """Camera with intrinsics and extrinsics."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    R: torch.Tensor  # [3, 3] world-to-camera rotation
    t: torch.Tensor  # [3] camera position in world coords

    @classmethod
    def from_json(cls, cam_dict: dict, device: str = "cpu") -> "Camera":
        """Load camera from cameras.json format."""
        R = torch.tensor(cam_dict["rotation"], dtype=torch.float32, device=device)
        t = torch.tensor(cam_dict["position"], dtype=torch.float32, device=device)
        return cls(
            fx=cam_dict["fx"],
            fy=cam_dict["fy"],
            cx=cam_dict["width"] / 2,
            cy=cam_dict["height"] / 2,
            width=cam_dict["width"],
            height=cam_dict["height"],
            R=R,
            t=t,
        )

    @classmethod
    def look_at(
        cls,
        eye: torch.Tensor,
        target: torch.Tensor,
        up: torch.Tensor,
        fx: float,
        fy: float,
        width: int,
        height: int,
    ) -> "Camera":
        """Create camera looking at target from eye position."""
        device = eye.device

        # Forward vector (camera -Z points at target)
        forward = target - eye
        forward = forward / torch.norm(forward)

        # Right vector
        right = torch.linalg.cross(forward, up)
        right = right / torch.norm(right)

        # Recompute up to ensure orthogonality
        up = torch.linalg.cross(right, forward)

        # Rotation matrix: R transforms world to camera space
        # R[2] = forward so that points in front of camera have positive depth
        R = torch.stack([right, up, forward], dim=0)

        return cls(
            fx=fx, fy=fy,
            cx=width / 2, cy=height / 2,
            width=width, height=height,
            R=R, t=eye,
        )


def load_splat(path: Path, device: str = "cpu") -> Gaussians:
    """Load Gaussian splat from PLY file.

    Handles OpenSplat/3DGS format with spherical harmonics.
    """
    with open(path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("utf-8").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Extract vertex count and properties
        n_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[2], parts[1]))  # (name, type)

        # Build format string for struct
        prop_names = [p[0] for p in properties]
        n_floats = len(properties)
        fmt = f"<{n_floats}f"
        stride = n_floats * 4

        # Read binary data
        data = np.zeros((n_vertices, n_floats), dtype=np.float32)
        for i in range(n_vertices):
            values = struct.unpack(fmt, f.read(stride))
            data[i] = values

    # Map property names to indices
    idx = {name: i for i, name in enumerate(prop_names)}

    # Extract tensors
    means = torch.tensor(
        data[:, [idx["x"], idx["y"], idx["z"]]],
        dtype=torch.float32, device=device
    )

    scales = torch.tensor(
        data[:, [idx["scale_0"], idx["scale_1"], idx["scale_2"]]],
        dtype=torch.float32, device=device
    )

    rotations = torch.tensor(
        data[:, [idx["rot_0"], idx["rot_1"], idx["rot_2"], idx["rot_3"]]],
        dtype=torch.float32, device=device
    )
    # Normalize quaternions
    rotations = rotations / torch.norm(rotations, dim=1, keepdim=True)

    opacities = torch.tensor(
        data[:, idx["opacity"]],
        dtype=torch.float32, device=device
    )

    sh_dc = torch.tensor(
        data[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]],
        dtype=torch.float32, device=device
    )

    # Higher-order SH (45 coefficients = 15 per channel)
    sh_rest_indices = [idx[f"f_rest_{i}"] for i in range(45)]
    sh_rest = torch.tensor(
        data[:, sh_rest_indices],
        dtype=torch.float32, device=device
    )

    return Gaussians(
        means=means,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
    )


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to rotation matrices.

    Args:
        q: [N, 4] quaternions (w, x, y, z)

    Returns:
        [N, 3, 3] rotation matrices
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)

    return R


def compute_3d_covariance(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Compute 3D covariance matrices from scale and rotation.

    Σ = R @ S @ S^T @ R^T where S = diag(exp(scale))

    Args:
        scales: [N, 3] log-scales
        rotations: [N, 4] quaternions

    Returns:
        [N, 3, 3] covariance matrices
    """
    N = scales.shape[0]

    # Scale matrices: S = diag(exp(scale))
    S = torch.zeros(N, 3, 3, device=scales.device, dtype=scales.dtype)
    exp_scales = torch.exp(scales)
    S[:, 0, 0] = exp_scales[:, 0]
    S[:, 1, 1] = exp_scales[:, 1]
    S[:, 2, 2] = exp_scales[:, 2]

    # Rotation matrices
    R = quat_to_rotmat(rotations)

    # Covariance: R @ S @ S^T @ R^T = R @ S @ (R @ S)^T = M @ M^T where M = R @ S
    M = torch.bmm(R, S)
    cov = torch.bmm(M, M.transpose(1, 2))

    return cov


def project_gaussians(
    gaussians: Gaussians,
    camera: Camera,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D Gaussians to 2D using EWA splatting.

    Returns:
        means_2d: [N, 2] projected centers (pixel coords)
        covs_2d: [N, 2, 2] 2D covariance matrices
        depths: [N] depth values (for sorting)
        opacities: [N] activated opacities (sigmoid)
        colors: [N, 3] RGB colors (from DC SH)
    """
    device = gaussians.means.device
    N = gaussians.means.shape[0]

    # Transform to camera space: p_cam = R @ (p_world - t)
    means_world = gaussians.means  # [N, 3]
    means_centered = means_world - camera.t.unsqueeze(0)  # [N, 3]
    means_cam = torch.mm(means_centered, camera.R.T)  # [N, 3]

    # Depth (z in camera space)
    depths = means_cam[:, 2]

    # Filter points behind camera
    valid = depths > 0.1

    # Project to image plane: x' = fx * x/z + cx, y' = fy * y/z + cy
    x_cam, y_cam, z_cam = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    means_2d = torch.zeros(N, 2, device=device)
    means_2d[:, 0] = camera.fx * x_cam / z_cam + camera.cx
    means_2d[:, 1] = camera.fy * y_cam / z_cam + camera.cy

    # Compute 3D covariance
    cov_3d = compute_3d_covariance(gaussians.scales, gaussians.rotations)  # [N, 3, 3]

    # Transform covariance to camera space: Σ_cam = R @ Σ_world @ R^T
    R = camera.R.unsqueeze(0)  # [1, 3, 3]
    cov_cam = torch.bmm(torch.bmm(R.expand(N, -1, -1), cov_3d), R.transpose(1, 2).expand(N, -1, -1))

    # Jacobian of perspective projection at each point
    # J = [[fx/z, 0, -fx*x/z²], [0, fy/z, -fy*y/z²]]
    J = torch.zeros(N, 2, 3, device=device)
    z2 = z_cam * z_cam
    J[:, 0, 0] = camera.fx / z_cam
    J[:, 0, 2] = -camera.fx * x_cam / z2
    J[:, 1, 1] = camera.fy / z_cam
    J[:, 1, 2] = -camera.fy * y_cam / z2

    # Project covariance: Σ_2d = J @ Σ_cam @ J^T
    covs_2d = torch.bmm(torch.bmm(J, cov_cam), J.transpose(1, 2))

    # Add low-pass filter to prevent aliasing (EWA antialiasing)
    covs_2d[:, 0, 0] += 0.3
    covs_2d[:, 1, 1] += 0.3

    # Activate opacities with sigmoid
    opacities = torch.sigmoid(gaussians.opacities)

    # Convert SH DC to RGB color
    # The SH DC coefficient relates to color as: color = SH_C0 * sh_dc + 0.5
    # where SH_C0 = 0.28209479177387814 (1/(2*sqrt(pi)))
    SH_C0 = 0.28209479177387814
    colors = SH_C0 * gaussians.sh_dc + 0.5
    colors = torch.clamp(colors, 0, 1)

    # Mask invalid points
    means_2d = torch.where(valid.unsqueeze(1), means_2d, torch.zeros_like(means_2d))
    depths = torch.where(valid, depths, torch.full_like(depths, float("inf")))
    opacities = torch.where(valid, opacities, torch.zeros_like(opacities))

    return means_2d, covs_2d, depths, opacities, colors


def render(
    gaussians: Gaussians,
    camera: Camera,
    bg_color: tuple[float, float, float] = (0, 0, 0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render Gaussians to an image.

    Uses per-pixel evaluation (slow but correct).

    Args:
        gaussians: Gaussian splat data
        camera: Camera parameters
        bg_color: Background color (R, G, B) in [0, 1]

    Returns:
        rgb: [H, W, 3] RGB image
        alpha: [H, W] alpha channel
        depth: [H, W] depth map
    """
    device = gaussians.means.device
    H, W = camera.height, camera.width

    # Project all Gaussians
    means_2d, covs_2d, depths, opacities, colors = project_gaussians(gaussians, camera)
    N = means_2d.shape[0]

    # Sort by depth (front to back)
    sort_idx = torch.argsort(depths)
    means_2d = means_2d[sort_idx]
    covs_2d = covs_2d[sort_idx]
    depths = depths[sort_idx]
    opacities = opacities[sort_idx]
    colors = colors[sort_idx]

    # Precompute inverse covariances for Mahalanobis distance
    # Add small regularization for numerical stability
    covs_2d_reg = covs_2d.clone()
    covs_2d_reg[:, 0, 0] += 1e-6
    covs_2d_reg[:, 1, 1] += 1e-6
    cov_inv = torch.inverse(covs_2d_reg)  # [N, 2, 2]

    # Compute determinants for normalization (optional, affects intensity)
    det = covs_2d[:, 0, 0] * covs_2d[:, 1, 1] - covs_2d[:, 0, 1] * covs_2d[:, 1, 0]

    # Initialize output
    rgb = torch.zeros(H, W, 3, device=device)
    alpha = torch.zeros(H, W, device=device)
    depth_out = torch.zeros(H, W, device=device)

    # Background color
    bg = torch.tensor(bg_color, device=device, dtype=torch.float32)

    # Precompute Gaussian bounding boxes (3-sigma radius)
    max_scales = torch.sqrt(torch.maximum(covs_2d[:, 0, 0], covs_2d[:, 1, 1]))
    radii = 3 * max_scales  # [N]

    # Gaussian bounds: [N, 4] = (x_min, y_min, x_max, y_max)
    g_bounds = torch.stack([
        means_2d[:, 0] - radii,
        means_2d[:, 1] - radii,
        means_2d[:, 0] + radii,
        means_2d[:, 1] + radii,
    ], dim=1)

    # Filter out invalid Gaussians globally
    valid_mask = (opacities >= 0.001) & (depths < float("inf"))

    # Process in tiles
    tile_size = 16  # Smaller tiles = better culling

    for ty in range(0, H, tile_size):
        for tx in range(0, W, tile_size):
            y_end = min(ty + tile_size, H)
            x_end = min(tx + tile_size, W)
            tile_h = y_end - ty
            tile_w = x_end - tx

            # Find Gaussians that overlap this tile (vectorized)
            tile_overlap = (
                valid_mask &
                (g_bounds[:, 0] < x_end) &
                (g_bounds[:, 2] > tx) &
                (g_bounds[:, 1] < y_end) &
                (g_bounds[:, 3] > ty)
            )
            tile_indices = torch.where(tile_overlap)[0]

            if len(tile_indices) == 0:
                continue

            # Pixel coordinates for this tile
            yy = torch.arange(ty, y_end, device=device, dtype=torch.float32)
            xx = torch.arange(tx, x_end, device=device, dtype=torch.float32)
            pixels_y, pixels_x = torch.meshgrid(yy, xx, indexing="ij")
            pixels = torch.stack([pixels_x, pixels_y], dim=-1)  # [tile_h, tile_w, 2]

            # Get tile-relevant data
            t_means = means_2d[tile_indices]      # [K, 2]
            t_cov_inv = cov_inv[tile_indices]     # [K, 2, 2]
            t_opacities = opacities[tile_indices] # [K]
            t_colors = colors[tile_indices]       # [K, 3]
            t_depths = depths[tile_indices]       # [K]

            # Compute all offsets: [tile_h, tile_w, K, 2]
            d = pixels.unsqueeze(2) - t_means  # [tile_h, tile_w, K, 2]

            # Mahalanobis distance for all Gaussians: [tile_h, tile_w, K]
            # d @ cov_inv @ d^T
            d_cov = torch.einsum("hwki,kij->hwkj", d, t_cov_inv)
            mahal = (d_cov * d).sum(dim=-1)  # [tile_h, tile_w, K]

            # Gaussian weights: [tile_h, tile_w, K]
            weights = torch.exp(-0.5 * mahal)
            alphas = torch.clamp(t_opacities * weights, 0, 0.99)

            # Front-to-back compositing (sequential over K, but K << N)
            tile_rgb = torch.zeros(tile_h, tile_w, 3, device=device)
            tile_T = torch.ones(tile_h, tile_w, device=device)
            tile_depth = torch.zeros(tile_h, tile_w, device=device)

            for k in range(len(tile_indices)):
                alpha_k = alphas[:, :, k]
                contribution = tile_T * alpha_k
                tile_rgb += contribution.unsqueeze(-1) * t_colors[k]
                tile_depth += contribution * t_depths[k]
                tile_T = tile_T * (1 - alpha_k)

                if tile_T.max() < 0.001:
                    break

            # Finalize tile
            tile_alpha = 1 - tile_T
            tile_rgb += tile_T.unsqueeze(-1) * bg

            rgb[ty:y_end, tx:x_end] = tile_rgb
            alpha[ty:y_end, tx:x_end] = tile_alpha
            depth_out[ty:y_end, tx:x_end] = tile_depth

    return rgb, alpha, depth_out


def render_image(
    splat_path: Path,
    camera: Camera,
    output_path: Path | None = None,
    device: str = "cpu",
    bg_color: tuple[float, float, float] = (0, 0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Load splat and render to image.

    Args:
        splat_path: Path to PLY file
        camera: Camera to render from
        output_path: Optional path to save PNG
        device: PyTorch device
        bg_color: Background color

    Returns:
        rgb: [H, W, 3] uint8 image
        alpha: [H, W] uint8 alpha
    """
    print(f"[render] Loading splat from {splat_path}...")
    gaussians = load_splat(splat_path, device=device)
    print(f"[render] Loaded {gaussians.means.shape[0]:,} Gaussians")

    print(f"[render] Rendering {camera.width}x{camera.height}...")
    rgb, alpha, depth = render(gaussians, camera, bg_color=bg_color)

    # Convert to numpy uint8
    rgb_np = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    alpha_np = (alpha.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    if output_path:
        # Save RGBA PNG
        rgba = np.dstack([rgb_np, alpha_np])
        Image.fromarray(rgba, mode="RGBA").save(output_path)
        print(f"[render] Saved to {output_path}")

    return rgb_np, alpha_np


def load_cameras(cameras_path: Path, device: str = "cpu") -> list[Camera]:
    """Load cameras from cameras.json."""
    with open(cameras_path) as f:
        cam_list = json.load(f)
    return [Camera.from_json(c, device=device) for c in cam_list]


def create_orbit_camera(
    center: torch.Tensor,
    distance: float,
    azimuth: float,
    elevation: float,
    fx: float,
    fy: float,
    width: int,
    height: int,
) -> Camera:
    """Create camera orbiting around center point.

    Uses Y-up coordinate system (COLMAP convention):
    - Cameras orbit in XZ plane
    - Y is up

    Args:
        center: [3] center point to orbit around
        distance: Distance from center
        azimuth: Horizontal angle in degrees (0 = +X, 90 = +Z)
        elevation: Vertical angle in degrees (0 = horizontal, 90 = top-down)
        fx, fy: Focal lengths
        width, height: Image dimensions
    """
    device = center.device

    # Convert to radians
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)

    # Compute camera position (Y-up, orbit in XZ plane)
    x = distance * np.cos(el_rad) * np.cos(az_rad)
    z = distance * np.cos(el_rad) * np.sin(az_rad)
    y = distance * np.sin(el_rad)

    eye = center + torch.tensor([x, y, z], device=device, dtype=torch.float32)
    up = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)

    return Camera.look_at(eye, center, up, fx, fy, width, height)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Gaussian splat")
    parser.add_argument("splat", type=Path, help="Path to PLY file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index from cameras.json")
    parser.add_argument("--cameras-json", type=Path, help="Path to cameras.json")
    parser.add_argument("--output", "-o", type=Path, default=Path("render.png"), help="Output PNG path")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--orbit", action="store_true", help="Render from orbit camera instead")
    parser.add_argument("--azimuth", type=float, default=0, help="Orbit azimuth in degrees")
    parser.add_argument("--elevation", type=float, default=30, help="Orbit elevation in degrees")
    parser.add_argument("--distance", type=float, default=1.0, help="Orbit distance multiplier")
    parser.add_argument("--downscale", type=int, default=1, help="Downscale factor for faster rendering")

    args = parser.parse_args()

    # Find cameras.json
    cameras_path = args.cameras_json
    if cameras_path is None:
        cameras_path = args.splat.parent / "cameras.json"

    if not cameras_path.exists():
        print(f"Error: cameras.json not found at {cameras_path}")
        exit(1)

    # Load cameras for reference
    cameras = load_cameras(cameras_path, device=args.device)
    ref_camera = cameras[args.camera]

    if args.orbit:
        # Compute center of Gaussians for orbit
        gaussians = load_splat(args.splat, device=args.device)
        center = gaussians.means.mean(dim=0)

        # Estimate distance from reference camera
        base_distance = torch.norm(ref_camera.t - center).item()

        camera = create_orbit_camera(
            center=center,
            distance=base_distance * args.distance,
            azimuth=args.azimuth,
            elevation=args.elevation,
            fx=ref_camera.fx / args.downscale,
            fy=ref_camera.fy / args.downscale,
            width=ref_camera.width // args.downscale,
            height=ref_camera.height // args.downscale,
        )
    else:
        camera = ref_camera
        if args.downscale > 1:
            camera = Camera(
                fx=camera.fx / args.downscale,
                fy=camera.fy / args.downscale,
                cx=camera.cx / args.downscale,
                cy=camera.cy / args.downscale,
                width=camera.width // args.downscale,
                height=camera.height // args.downscale,
                R=camera.R,
                t=camera.t,
            )

    render_image(args.splat, camera, args.output, device=args.device)
