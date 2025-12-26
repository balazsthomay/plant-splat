"""
Image augmentations for synthetic data generation.

Provides lighting and background variations to increase dataset diversity.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image


# Color temperature presets (Kelvin -> RGB multipliers)
# Based on Tanner Helland's algorithm
def kelvin_to_rgb(kelvin: int) -> tuple[float, float, float]:
    """Convert color temperature to RGB multipliers.

    Args:
        kelvin: Color temperature in Kelvin (1000-40000)

    Returns:
        (r, g, b) multipliers normalized around 1.0
    """
    temp = kelvin / 100.0

    # Red
    if temp <= 66:
        r = 255
    else:
        r = temp - 60
        r = 329.698727446 * (r ** -0.1332047592)
        r = max(0, min(255, r))

    # Green
    if temp <= 66:
        g = temp
        g = 99.4708025861 * np.log(g) - 161.1195681661
    else:
        g = temp - 60
        g = 288.1221695283 * (g ** -0.0755148492)
    g = max(0, min(255, g))

    # Blue
    if temp >= 66:
        b = 255
    elif temp <= 19:
        b = 0
    else:
        b = temp - 10
        b = 138.5177312231 * np.log(b) - 305.0447927307
        b = max(0, min(255, b))

    # Normalize around 1.0 (daylight ~6500K as reference)
    ref = kelvin_to_rgb_raw(6500)
    return (r / ref[0], g / ref[1], b / ref[2])


def kelvin_to_rgb_raw(kelvin: int) -> tuple[float, float, float]:
    """Raw kelvin to RGB without normalization."""
    temp = kelvin / 100.0

    if temp <= 66:
        r = 255
    else:
        r = temp - 60
        r = 329.698727446 * (r ** -0.1332047592)
        r = max(0, min(255, r))

    if temp <= 66:
        g = temp
        g = 99.4708025861 * np.log(g) - 161.1195681661
    else:
        g = temp - 60
        g = 288.1221695283 * (g ** -0.0755148492)
    g = max(0, min(255, g))

    if temp >= 66:
        b = 255
    elif temp <= 19:
        b = 0
    else:
        b = temp - 10
        b = 138.5177312231 * np.log(b) - 305.0447927307
        b = max(0, min(255, b))

    return (r, g, b)


class LightingAugmentor:
    """Applies lighting variations to rendered images."""

    # Common lighting conditions
    PRESETS = {
        "warm_indoor": {"temp_range": (2700, 3500), "intensity_range": (0.7, 1.0)},
        "cool_indoor": {"temp_range": (4000, 5000), "intensity_range": (0.8, 1.1)},
        "daylight": {"temp_range": (5500, 6500), "intensity_range": (0.9, 1.2)},
        "overcast": {"temp_range": (6500, 7500), "intensity_range": (0.6, 0.9)},
        "golden_hour": {"temp_range": (2500, 3500), "intensity_range": (0.8, 1.1)},
        "shade": {"temp_range": (7000, 9000), "intensity_range": (0.5, 0.8)},
    }

    def __init__(
        self,
        temp_range: tuple[int, int] = (2700, 8000),
        intensity_range: tuple[float, float] = (0.6, 1.3),
        contrast_range: tuple[float, float] = (0.9, 1.1),
    ):
        """
        Args:
            temp_range: Color temperature range in Kelvin (min, max)
            intensity_range: Brightness multiplier range
            contrast_range: Contrast adjustment range
        """
        self.temp_range = temp_range
        self.intensity_range = intensity_range
        self.contrast_range = contrast_range

    @classmethod
    def from_preset(cls, preset: str) -> "LightingAugmentor":
        """Create augmentor from a preset."""
        if preset not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(cls.PRESETS.keys())}")
        cfg = cls.PRESETS[preset]
        return cls(
            temp_range=cfg["temp_range"],
            intensity_range=cfg["intensity_range"],
        )

    def random_params(self) -> dict:
        """Generate random lighting parameters."""
        return {
            "temperature": random.randint(*self.temp_range),
            "intensity": random.uniform(*self.intensity_range),
            "contrast": random.uniform(*self.contrast_range),
        }

    def apply(
        self,
        rgb: np.ndarray,
        temperature: int | None = None,
        intensity: float | None = None,
        contrast: float | None = None,
    ) -> np.ndarray:
        """Apply lighting augmentation to RGB image.

        Args:
            rgb: [H, W, 3] uint8 image
            temperature: Color temperature in Kelvin (random if None)
            intensity: Brightness multiplier (random if None)
            contrast: Contrast multiplier (random if None)

        Returns:
            Augmented [H, W, 3] uint8 image
        """
        params = self.random_params()
        temperature = temperature or params["temperature"]
        intensity = intensity or params["intensity"]
        contrast = contrast or params["contrast"]

        # Convert to float
        img = rgb.astype(np.float32) / 255.0

        # Apply color temperature
        r_mult, g_mult, b_mult = kelvin_to_rgb(temperature)
        img[:, :, 0] *= r_mult
        img[:, :, 1] *= g_mult
        img[:, :, 2] *= b_mult

        # Apply intensity (brightness)
        img *= intensity

        # Apply contrast (around 0.5 midpoint)
        img = (img - 0.5) * contrast + 0.5

        # Clip and convert back
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        return img, {"temperature": temperature, "intensity": intensity, "contrast": contrast}


class BackgroundAugmentor:
    """Generates varied backgrounds for compositing."""

    def __init__(
        self,
        image_dir: Path | None = None,
        color_mode: str = "varied",  # "varied", "white", "black", "random"
    ):
        """
        Args:
            image_dir: Directory containing background images (optional)
            color_mode: How to generate solid colors when no images
        """
        self.image_dir = image_dir
        self.color_mode = color_mode
        self.images: list[Path] = []

        if image_dir and image_dir.exists():
            self.images = list(image_dir.glob("*.jpg")) + \
                          list(image_dir.glob("*.jpeg")) + \
                          list(image_dir.glob("*.png"))

    def _random_color(self) -> tuple[int, int, int]:
        """Generate a random background color."""
        if self.color_mode == "white":
            return (255, 255, 255)
        elif self.color_mode == "black":
            return (0, 0, 0)
        elif self.color_mode == "varied":
            # Greenhouse-like colors: greens, browns, grays, whites
            palette = [
                (255, 255, 255),  # White
                (240, 240, 240),  # Light gray
                (200, 200, 200),  # Gray
                (180, 200, 180),  # Light green-gray
                (150, 170, 150),  # Muted green
                (120, 100, 80),   # Brown
                (200, 180, 160),  # Tan
                (220, 220, 210),  # Off-white
                (100, 120, 100),  # Dark green
                (80, 80, 80),     # Dark gray
            ]
            return random.choice(palette)
        else:  # random
            return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def get_background(self, width: int, height: int) -> tuple[np.ndarray, dict]:
        """Get a background for compositing.

        Args:
            width: Target width
            height: Target height

        Returns:
            [H, W, 3] uint8 background image, metadata dict
        """
        if self.images:
            # Random image from directory
            img_path = random.choice(self.images)
            img = Image.open(img_path).convert("RGB")

            # Random crop to target aspect ratio, then resize
            img_w, img_h = img.size
            target_ratio = width / height
            img_ratio = img_w / img_h

            if img_ratio > target_ratio:
                # Image is wider, crop width
                new_w = int(img_h * target_ratio)
                left = random.randint(0, img_w - new_w)
                img = img.crop((left, 0, left + new_w, img_h))
            else:
                # Image is taller, crop height
                new_h = int(img_w / target_ratio)
                top = random.randint(0, img_h - new_h)
                img = img.crop((0, top, img_w, top + new_h))

            img = img.resize((width, height), Image.Resampling.LANCZOS)
            return np.array(img), {"type": "image", "source": str(img_path.name)}
        else:
            # Solid color
            color = self._random_color()
            bg = np.full((height, width, 3), color, dtype=np.uint8)
            return bg, {"type": "solid", "color": color}


def composite(
    rgb: np.ndarray,
    alpha: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    """Composite foreground onto background using alpha.

    Args:
        rgb: [H, W, 3] foreground RGB (uint8)
        alpha: [H, W] alpha values (uint8, 0-255)
        background: [H, W, 3] background RGB (uint8)

    Returns:
        [H, W, 3] composited image (uint8)
    """
    alpha_f = alpha.astype(np.float32) / 255.0

    result = (rgb.astype(np.float32) * alpha_f[:, :, None] +
              background.astype(np.float32) * (1 - alpha_f[:, :, None]))

    return result.clip(0, 255).astype(np.uint8)
