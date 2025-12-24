"""
Filter COLMAP point cloud to keep only points visible in foreground masks.

For each 3D point, projects to all cameras that observe it.
Keeps points where majority of projections land in foreground (mask > 0).
"""

import argparse
import struct
from pathlib import Path
from collections import namedtuple

import numpy as np
from PIL import Image


# COLMAP binary format structures
CameraModel = namedtuple('CameraModel', ['model_id', 'model_name', 'num_params'])
CAMERA_MODELS = {
    0: CameraModel(0, 'SIMPLE_PINHOLE', 3),
    1: CameraModel(1, 'PINHOLE', 4),
    2: CameraModel(2, 'SIMPLE_RADIAL', 4),
    3: CameraModel(3, 'RADIAL', 5),
    4: CameraModel(4, 'OPENCV', 8),
    5: CameraModel(5, 'OPENCV_FISHEYE', 8),
}


def read_cameras_binary(path: Path) -> dict:
    """Read cameras.bin file."""
    cameras = {}
    with open(path, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack('<I', f.read(4))[0]
            model_id = struct.unpack('<i', f.read(4))[0]
            width = struct.unpack('<Q', f.read(8))[0]
            height = struct.unpack('<Q', f.read(8))[0]
            num_params = CAMERA_MODELS[model_id].num_params
            params = struct.unpack(f'<{num_params}d', f.read(8 * num_params))
            cameras[camera_id] = {
                'model_id': model_id,
                'width': width,
                'height': height,
                'params': params,
            }
    return cameras


def read_images_binary(path: Path) -> dict:
    """Read images.bin file."""
    images = {}
    with open(path, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack('<I', f.read(4))[0]
            qw, qx, qy, qz = struct.unpack('<4d', f.read(32))
            tx, ty, tz = struct.unpack('<3d', f.read(24))
            camera_id = struct.unpack('<I', f.read(4))[0]

            # Read image name (null-terminated)
            name_chars = []
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name_chars.append(c.decode('utf-8'))
            name = ''.join(name_chars)

            # Read 2D points
            num_points2D = struct.unpack('<Q', f.read(8))[0]
            points2D = []
            point3D_ids = []
            for _ in range(num_points2D):
                x, y = struct.unpack('<2d', f.read(16))
                point3D_id = struct.unpack('<q', f.read(8))[0]  # signed, -1 = invalid
                points2D.append((x, y))
                point3D_ids.append(point3D_id)

            images[image_id] = {
                'qvec': np.array([qw, qx, qy, qz]),
                'tvec': np.array([tx, ty, tz]),
                'camera_id': camera_id,
                'name': name,
                'points2D': np.array(points2D),
                'point3D_ids': np.array(point3D_ids),
            }
    return images


def read_points3D_binary(path: Path) -> dict:
    """Read points3D.bin file."""
    points3D = {}
    with open(path, 'rb') as f:
        num_points = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack('<Q', f.read(8))[0]
            x, y, z = struct.unpack('<3d', f.read(24))
            r, g, b = struct.unpack('<3B', f.read(3))
            error = struct.unpack('<d', f.read(8))[0]

            track_length = struct.unpack('<Q', f.read(8))[0]
            track = []
            for _ in range(track_length):
                image_id = struct.unpack('<I', f.read(4))[0]
                point2D_idx = struct.unpack('<I', f.read(4))[0]
                track.append((image_id, point2D_idx))

            points3D[point3D_id] = {
                'xyz': np.array([x, y, z]),
                'rgb': np.array([r, g, b]),
                'error': error,
                'track': track,
            }
    return points3D


def write_points3D_binary(path: Path, points3D: dict):
    """Write points3D.bin file."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(points3D)))
        for point3D_id, point in points3D.items():
            f.write(struct.pack('<Q', point3D_id))
            f.write(struct.pack('<3d', *point['xyz']))
            f.write(struct.pack('<3B', *point['rgb']))
            f.write(struct.pack('<d', point['error']))
            f.write(struct.pack('<Q', len(point['track'])))
            for image_id, point2D_idx in point['track']:
                f.write(struct.pack('<I', image_id))
                f.write(struct.pack('<I', point2D_idx))


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
    ])


def project_point(xyz: np.ndarray, image: dict, camera: dict) -> tuple[float, float] | None:
    """Project 3D point to 2D image coordinates. Returns None if behind camera."""
    # World to camera transform
    R = qvec_to_rotmat(image['qvec'])
    t = image['tvec']

    # Transform to camera coordinates
    xyz_cam = R @ xyz + t

    # Check if point is in front of camera
    if xyz_cam[2] <= 0:
        return None

    # Project using camera model
    model_id = camera['model_id']
    params = camera['params']

    x = xyz_cam[0] / xyz_cam[2]
    y = xyz_cam[1] / xyz_cam[2]

    if model_id == 4:  # OPENCV
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        r2 = x*x + y*y
        radial = 1 + k1*r2 + k2*r2*r2
        x_dist = x * radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        y_dist = y * radial + p1*(r2 + 2*y*y) + 2*p2*x*y
        u = fx * x_dist + cx
        v = fy * y_dist + cy
    elif model_id == 1:  # PINHOLE
        fx, fy, cx, cy = params
        u = fx * x + cx
        v = fy * y + cy
    else:
        # Fallback for other models - simple pinhole
        fx = params[0]
        fy = params[1] if len(params) > 1 else fx
        cx = params[2] if len(params) > 2 else camera['width'] / 2
        cy = params[3] if len(params) > 3 else camera['height'] / 2
        u = fx * x + cx
        v = fy * y + cy

    return u, v


def filter_points(
    sparse_dir: Path,
    images_dir: Path,
    output_dir: Path,
    min_visible_ratio: float = 0.5,
) -> int:
    """Filter points to keep only those visible in foreground masks.

    Args:
        sparse_dir: COLMAP sparse reconstruction directory
        images_dir: Directory containing images and masks
        output_dir: Output directory for filtered reconstruction
        min_visible_ratio: Minimum ratio of views where point must be in foreground

    Returns:
        Number of points kept
    """
    print(f"[filter] Loading COLMAP reconstruction from {sparse_dir}")
    cameras = read_cameras_binary(sparse_dir / 'cameras.bin')
    images = read_images_binary(sparse_dir / 'images.bin')
    points3D = read_points3D_binary(sparse_dir / 'points3D.bin')

    print(f"[filter] Loaded {len(cameras)} cameras, {len(images)} images, {len(points3D)} points")

    # Build image_id -> image data mapping with loaded masks
    print("[filter] Loading masks...")
    masks_cache = {}
    for image_id, img_data in images.items():
        mask_path = images_dir / f"{Path(img_data['name']).stem}_mask.png"
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert('L'))
            masks_cache[image_id] = mask

    print(f"[filter] Loaded {len(masks_cache)} masks")

    if len(masks_cache) == 0:
        print("[filter] ERROR: No masks found!")
        return 0

    # Filter points
    print("[filter] Filtering points by mask projection...")
    filtered_points = {}
    kept = 0
    discarded = 0

    for point3D_id, point in points3D.items():
        xyz = point['xyz']
        track = point['track']

        if len(track) == 0:
            discarded += 1
            continue

        # Check each observation
        foreground_count = 0
        total_checked = 0

        for image_id, point2D_idx in track:
            if image_id not in masks_cache:
                continue
            if image_id not in images:
                continue

            img_data = images[image_id]
            camera = cameras[img_data['camera_id']]
            mask = masks_cache[image_id]

            # Project point to this camera
            proj = project_point(xyz, img_data, camera)
            if proj is None:
                continue

            u, v = proj
            u_int, v_int = int(round(u)), int(round(v))

            # Check if in bounds
            h, w = mask.shape
            if 0 <= u_int < w and 0 <= v_int < h:
                total_checked += 1
                if mask[v_int, u_int] > 127:  # Foreground
                    foreground_count += 1

        # Keep point if visible in foreground in enough views
        if total_checked > 0 and foreground_count / total_checked >= min_visible_ratio:
            filtered_points[point3D_id] = point
            kept += 1
        else:
            discarded += 1

    print(f"[filter] Kept {kept} points, discarded {discarded}")

    # Write filtered reconstruction
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy cameras.bin and images.bin unchanged
    import shutil
    shutil.copy(sparse_dir / 'cameras.bin', output_dir / 'cameras.bin')
    shutil.copy(sparse_dir / 'images.bin', output_dir / 'images.bin')
    if (sparse_dir / 'project.ini').exists():
        shutil.copy(sparse_dir / 'project.ini', output_dir / 'project.ini')

    # Write filtered points
    write_points3D_binary(output_dir / 'points3D.bin', filtered_points)

    print(f"[filter] ✓ Wrote filtered reconstruction to {output_dir}")
    return kept


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter COLMAP points by mask projection")
    parser.add_argument("sparse_dir", type=Path, help="COLMAP sparse reconstruction directory")
    parser.add_argument("images_dir", type=Path, help="Directory with images and masks")
    parser.add_argument("-o", "--output", type=Path, help="Output directory (default: sparse_dir/../sparse_filtered/0)")
    parser.add_argument("--min-ratio", type=float, default=0.5, help="Min foreground visibility ratio (default: 0.5)")

    args = parser.parse_args()

    output_dir = args.output or args.sparse_dir.parent.parent / 'sparse_filtered' / '0'

    filter_points(args.sparse_dir, args.images_dir, output_dir, args.min_ratio)
