"""
Filter Gaussian splat PLY to remove background Gaussians.

Strategies:
1. Mask projection - project Gaussians to cameras, keep if in foreground masks
2. Opacity threshold - remove low-opacity Gaussians
3. Distance from centroid - remove outliers far from the plant center
4. Bounding box - keep only Gaussians within specified bounds
"""

import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image

# Import COLMAP reading/projection from filter_points
from filter_points import (
    read_cameras_binary,
    read_images_binary,
    project_point,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Convert logit to probability."""
    return 1 / (1 + np.exp(-x))


def read_ply(path: Path) -> tuple[dict, np.ndarray, list[str]]:
    """Read Gaussian splat PLY file.

    Returns:
        header_info: dict with format, element count
        data: structured numpy array with all properties
        property_names: list of property names in order
    """
    with open(path, 'rb') as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        # Extract info from header
        num_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property float'):
                properties.append(line.split()[-1])

        # Create dtype for structured array
        dtype = [(name, '<f4') for name in properties]

        # Read binary data
        data = np.frombuffer(f.read(), dtype=dtype, count=num_vertices)

    return {'num_vertices': num_vertices, 'header': header_lines}, data, properties


def write_ply(path: Path, data: np.ndarray, properties: list[str]):
    """Write Gaussian splat PLY file."""
    with open(path, 'wb') as f:
        # Write header
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {len(data)}\n'.encode())
        for prop in properties:
            f.write(f'property float {prop}\n'.encode())
        f.write(b'end_header\n')

        # Write data
        f.write(data.tobytes())


def filter_by_masks(
    xyz: np.ndarray,
    sparse_dir: Path,
    images_dir: Path,
    min_visible_ratio: float = 0.5,
) -> np.ndarray:
    """Filter Gaussians by projecting to masks.

    For each Gaussian, project its center to all training cameras.
    Keep if projection lands in foreground mask (>127) for ≥ min_visible_ratio of views.

    Args:
        xyz: [N, 3] Gaussian center positions
        sparse_dir: COLMAP sparse reconstruction directory
        images_dir: Directory containing masks (*_mask.png)
        min_visible_ratio: Min fraction of views where Gaussian must be in foreground

    Returns:
        Boolean mask [N] - True = keep
    """
    print(f"[filter_splat] Loading COLMAP data from {sparse_dir}")
    cameras = read_cameras_binary(sparse_dir / 'cameras.bin')
    images = read_images_binary(sparse_dir / 'images.bin')
    print(f"[filter_splat] Loaded {len(cameras)} cameras, {len(images)} images")

    # Load masks
    print("[filter_splat] Loading masks...")
    masks_cache = {}
    for image_id, img_data in images.items():
        mask_path = images_dir / f"{Path(img_data['name']).stem}_mask.png"
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert('L'))
            masks_cache[image_id] = mask

    print(f"[filter_splat] Loaded {len(masks_cache)} masks")
    if len(masks_cache) == 0:
        print("[filter_splat] WARNING: No masks found, skipping mask filter")
        return np.ones(len(xyz), dtype=bool)

    # Project each Gaussian to all cameras
    print(f"[filter_splat] Projecting {len(xyz)} Gaussians to {len(images)} cameras...")
    keep_mask = np.zeros(len(xyz), dtype=bool)

    for i, pos in enumerate(xyz):
        foreground_count = 0
        total_checked = 0

        for image_id, img_data in images.items():
            if image_id not in masks_cache:
                continue

            camera = cameras[img_data['camera_id']]
            mask = masks_cache[image_id]

            # Project Gaussian center
            proj = project_point(pos, img_data, camera)
            if proj is None:
                continue

            u, v = proj
            u_int, v_int = int(round(u)), int(round(v))

            # Check bounds
            h, w = mask.shape
            if 0 <= u_int < w and 0 <= v_int < h:
                total_checked += 1
                if mask[v_int, u_int] > 127:
                    foreground_count += 1

        # Keep if in foreground for enough views
        if total_checked > 0 and foreground_count / total_checked >= min_visible_ratio:
            keep_mask[i] = True

        # Progress
        if (i + 1) % 5000 == 0:
            print(f"  [{i + 1}/{len(xyz)}] processed")

    kept = keep_mask.sum()
    print(f"[filter_splat] Mask filter: {kept} / {len(xyz)} Gaussians in foreground")
    return keep_mask


def filter_splat(
    input_path: Path,
    output_path: Path,
    opacity_threshold: float = 0.1,
    distance_threshold: float | None = None,
    bbox: tuple[float, float, float, float, float, float] | None = None,
    percentile: float = 95,
    sparse_dir: Path | None = None,
    images_dir: Path | None = None,
    min_visible_ratio: float = 0.5,
) -> int:
    """Filter Gaussian splat by mask projection and geometric criteria.

    Args:
        input_path: Input PLY file
        output_path: Output PLY file
        opacity_threshold: Minimum opacity (0-1) to keep
        distance_threshold: Max distance from centroid (computed from high-opacity points)
        bbox: Bounding box (x_min, x_max, y_min, y_max, z_min, z_max)
        percentile: If distance_threshold is None, compute threshold as this percentile
                    of distances from high-opacity Gaussians
        sparse_dir: COLMAP sparse reconstruction (for mask filtering)
        images_dir: Directory with masks (for mask filtering)
        min_visible_ratio: Min foreground visibility ratio for mask filter

    Returns:
        Number of Gaussians kept
    """
    print(f"[filter_splat] Loading {input_path}")
    header_info, data, properties = read_ply(input_path)
    print(f"[filter_splat] Loaded {len(data)} Gaussians")

    # Get positions and opacity
    xyz = np.column_stack([data['x'], data['y'], data['z']])
    opacity = sigmoid(data['opacity'])

    print(f"[filter_splat] Opacity range: {opacity.min():.3f} - {opacity.max():.3f}")
    print(f"[filter_splat] Opacity mean: {opacity.mean():.3f}")

    # Start with mask filter (most important for isolation)
    if sparse_dir is not None and images_dir is not None:
        keep_mask = filter_by_masks(xyz, sparse_dir, images_dir, min_visible_ratio)
    else:
        keep_mask = np.ones(len(data), dtype=bool)

    # Filter by opacity
    opacity_mask = opacity >= opacity_threshold
    print(f"[filter_splat] Opacity >= {opacity_threshold}: {opacity_mask.sum()} / {len(data)}")
    keep_mask &= opacity_mask

    # Compute centroid from high-opacity Gaussians
    high_opacity_mask = opacity >= 0.5
    if high_opacity_mask.sum() > 0:
        centroid = xyz[high_opacity_mask].mean(axis=0)
    else:
        centroid = xyz.mean(axis=0)
    print(f"[filter_splat] Centroid: {centroid}")

    # Compute distances
    distances = np.linalg.norm(xyz - centroid, axis=1)
    print(f"[filter_splat] Distance range: {distances.min():.3f} - {distances.max():.3f}")

    # Filter by distance
    if distance_threshold is None and bbox is None:
        # Auto-compute threshold from percentile of high-opacity distances
        high_opacity_distances = distances[high_opacity_mask]
        if len(high_opacity_distances) > 0:
            distance_threshold = np.percentile(high_opacity_distances, percentile)
            print(f"[filter_splat] Auto distance threshold ({percentile}th percentile): {distance_threshold:.3f}")

    if distance_threshold is not None:
        distance_mask = distances <= distance_threshold
        print(f"[filter_splat] Distance <= {distance_threshold:.3f}: {distance_mask.sum()} / {len(data)}")
        keep_mask &= distance_mask

    # Filter by bounding box
    if bbox is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        bbox_mask = (
            (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) &
            (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max) &
            (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
        )
        print(f"[filter_splat] In bounding box: {bbox_mask.sum()} / {len(data)}")
        keep_mask &= bbox_mask

    # Apply filter
    filtered_data = data[keep_mask]
    print(f"[filter_splat] Keeping {len(filtered_data)} / {len(data)} Gaussians")

    # Write output
    write_ply(output_path, filtered_data, properties)
    print(f"[filter_splat] Wrote {output_path}")

    return len(filtered_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Gaussian splat PLY")
    parser.add_argument("input", type=Path, help="Input PLY file")
    parser.add_argument("-o", "--output", type=Path, help="Output PLY file")
    parser.add_argument("--sparse-dir", type=Path, help="COLMAP sparse dir (for mask filtering)")
    parser.add_argument("--images-dir", type=Path, help="Images dir with masks (for mask filtering)")
    parser.add_argument("--min-ratio", type=float, default=0.5, help="Min foreground visibility ratio")
    parser.add_argument("--opacity", type=float, default=0.1, help="Min opacity threshold (0-1)")
    parser.add_argument("--distance", type=float, help="Max distance from centroid")
    parser.add_argument("--percentile", type=float, default=95, help="Auto-compute distance as this percentile")
    parser.add_argument("--bbox", type=float, nargs=6, metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX', 'Z_MIN', 'Z_MAX'),
                        help="Bounding box")

    args = parser.parse_args()

    output_path = args.output or args.input.with_stem(args.input.stem + "_filtered")
    bbox = tuple(args.bbox) if args.bbox else None

    filter_splat(
        args.input,
        output_path,
        args.opacity,
        args.distance,
        bbox,
        args.percentile,
        args.sparse_dir,
        args.images_dir,
        args.min_ratio,
    )
