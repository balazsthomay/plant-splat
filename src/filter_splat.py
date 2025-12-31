"""
Filter Gaussian splat PLY to remove background Gaussians.

Strategies:
1. Opacity threshold - remove low-opacity Gaussians
2. Distance from centroid - remove outliers far from the plant center
3. Bounding box - keep only Gaussians within specified bounds
"""

import argparse
import struct
from pathlib import Path

import numpy as np


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


def filter_splat(
    input_path: Path,
    output_path: Path,
    opacity_threshold: float = 0.1,
    distance_threshold: float | None = None,
    bbox: tuple[float, float, float, float, float, float] | None = None,
    percentile: float = 95,
) -> int:
    """Filter Gaussian splat by opacity and spatial criteria.

    Args:
        input_path: Input PLY file
        output_path: Output PLY file
        opacity_threshold: Minimum opacity (0-1) to keep
        distance_threshold: Max distance from centroid (computed from high-opacity points)
        bbox: Bounding box (x_min, x_max, y_min, y_max, z_min, z_max)
        percentile: If distance_threshold is None, compute threshold as this percentile
                    of distances from high-opacity Gaussians

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

    # Start with all True mask
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
    parser.add_argument("--opacity", type=float, default=0.1, help="Min opacity threshold (0-1)")
    parser.add_argument("--distance", type=float, help="Max distance from centroid")
    parser.add_argument("--percentile", type=float, default=95, help="Auto-compute distance as this percentile")
    parser.add_argument("--bbox", type=float, nargs=6, metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX', 'Z_MIN', 'Z_MAX'),
                        help="Bounding box")

    args = parser.parse_args()

    output_path = args.output or args.input.with_stem(args.input.stem + "_filtered")
    bbox = tuple(args.bbox) if args.bbox else None

    filter_splat(args.input, output_path, args.opacity, args.distance, bbox, args.percentile)
