"""
Disease synthesis using Stable Diffusion 1.5 inpainting.

Applies simulated plant diseases to healthy rendered images with controllable:
- Disease type (powdery_mildew, leaf_spot, rust, chlorosis, blight)
- Severity (0.0-1.0 continuous scale)
- Spatial distribution (auto-generated via Perlin noise + plant mask)

Supports both CUDA and MPS backends.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter


class DiseaseType(Enum):
    """Supported plant disease types."""
    POWDERY_MILDEW = "powdery_mildew"
    LEAF_SPOT = "leaf_spot"
    RUST = "rust"
    CHLOROSIS = "chlorosis"
    BLIGHT = "blight"


@dataclass
class DiseaseConfig:
    """Configuration for a specific disease type."""
    name: str
    prompt: str
    negative_prompt: str
    # Typical severity range for this disease (for random sampling)
    typical_severity: tuple[float, float]


# Disease prompts tuned for SD 1.5 inpainting
DISEASE_CONFIGS: dict[DiseaseType, DiseaseConfig] = {
    DiseaseType.POWDERY_MILDEW: DiseaseConfig(
        name="powdery_mildew",
        prompt="powdery mildew fungal infection on plant leaf, white fuzzy powder coating, plant disease, detailed texture",
        negative_prompt="healthy green leaf, clean surface, no disease",
        typical_severity=(0.2, 0.7),
    ),
    DiseaseType.LEAF_SPOT: DiseaseConfig(
        name="leaf_spot",
        prompt="bacterial leaf spot disease, brown circular necrotic lesions with yellow chlorotic halos, plant pathology, detailed",
        negative_prompt="healthy leaf, uniform green color, no spots",
        typical_severity=(0.2, 0.6),
    ),
    DiseaseType.RUST: DiseaseConfig(
        name="rust",
        prompt="plant rust fungal disease, orange-brown pustules and spores on leaf surface, rusty discoloration, detailed texture",
        negative_prompt="healthy green leaf, no rust, no orange",
        typical_severity=(0.2, 0.6),
    ),
    DiseaseType.CHLOROSIS: DiseaseConfig(
        name="chlorosis",
        prompt="leaf chlorosis yellowing, interveinal chlorosis with yellow leaves and green veins, nutrient deficiency, fading color",
        negative_prompt="deep green healthy leaf, vibrant color",
        typical_severity=(0.3, 0.8),
    ),
    DiseaseType.BLIGHT: DiseaseConfig(
        name="blight",
        prompt="plant blight disease, dark brown-black necrotic tissue, wilting decay, severe damage, dead tissue",
        negative_prompt="healthy vibrant leaf, green, alive",
        typical_severity=(0.4, 0.9),
    ),
}


def generate_perlin_noise(shape: tuple[int, int], scale: float = 50.0, seed: int | None = None) -> np.ndarray:
    """Generate Perlin-like noise using octave summation.

    Args:
        shape: (height, width) output shape
        scale: Base noise scale (larger = more gradual)
        seed: Random seed for reproducibility

    Returns:
        [H, W] float32 noise in [0, 1]
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # Generate noise at multiple octaves
    noise = np.zeros((h, w), dtype=np.float32)

    for octave in range(4):
        freq = 2 ** octave
        amplitude = 0.5 ** octave

        # Low-res random noise
        low_h = max(2, int(h / scale * freq))
        low_w = max(2, int(w / scale * freq))
        low_noise = rng.random((low_h, low_w)).astype(np.float32)

        # Upscale with smooth interpolation
        from PIL import Image as PILImage
        low_img = PILImage.fromarray((low_noise * 255).astype(np.uint8))
        high_img = low_img.resize((w, h), PILImage.Resampling.BILINEAR)
        high_noise = np.array(high_img).astype(np.float32) / 255.0

        noise += high_noise * amplitude

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


class ChangeMapGenerator:
    """Generates spatially-coherent change maps for disease synthesis."""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self._call_count = 0

    def generate(
        self,
        plant_mask: np.ndarray,
        severity: float,
        pattern: str = "patchy",
        feather_radius: float = 5.0,
    ) -> np.ndarray:
        """Generate change map for disease application.

        Args:
            plant_mask: [H, W] binary mask (1 = plant, 0 = background)
            severity: 0.0-1.0 disease severity (affects coverage and intensity)
            pattern: Spatial distribution ("patchy", "edge", "uniform")
            feather_radius: Gaussian blur radius for soft edges

        Returns:
            [H, W] float32 change map in [0, 1]
        """
        h, w = plant_mask.shape

        # Use incrementing seed for reproducibility across calls
        call_seed = None if self.seed is None else self.seed + self._call_count
        self._call_count += 1

        if pattern == "uniform":
            # Uniform severity across plant
            change_map = np.ones((h, w), dtype=np.float32) * severity

        elif pattern == "edge":
            # Disease concentrated at leaf edges
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(plant_mask)
            max_dist = dist.max() + 1e-8
            # Invert: high values at edges
            edge_map = 1.0 - (dist / max_dist)
            # Add some noise for variation
            noise = generate_perlin_noise((h, w), scale=30.0, seed=call_seed)
            change_map = edge_map * (0.5 + 0.5 * noise) * severity

        else:  # patchy (default)
            # Perlin noise creates natural-looking patches
            noise = generate_perlin_noise((h, w), scale=40.0, seed=call_seed)

            # Threshold to create discrete patches, scaled by severity
            # Higher severity = more area affected
            threshold = 1.0 - severity
            patchy = (noise > threshold).astype(np.float32)

            # Blend with continuous noise for gradual edges
            change_map = patchy * 0.7 + noise * severity * 0.3

        # Apply plant mask
        change_map = change_map * plant_mask.astype(np.float32)

        # Feather edges for smooth blending
        if feather_radius > 0:
            change_map = gaussian_filter(change_map, sigma=feather_radius)
            # Re-mask to prevent bleeding outside plant
            change_map = change_map * plant_mask.astype(np.float32)

        # Normalize to [0, 1]
        if change_map.max() > 0:
            change_map = change_map / change_map.max()

        # Scale by severity for final intensity
        change_map = change_map * min(1.0, severity * 1.2)

        return change_map.astype(np.float32)


def get_device() -> str:
    """Auto-detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DiseaseAugmentor:
    """Applies disease symptoms to healthy plant images using SD 1.5 inpainting."""

    def __init__(
        self,
        device: str | None = None,
        model_id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting",
        num_inference_steps: int = 30,
    ):
        """
        Args:
            device: PyTorch device (auto-detect if None)
            model_id: HuggingFace model ID for inpainting
            num_inference_steps: Default diffusion steps (more = better quality, slower)
        """
        self.device = device or get_device()
        self.model_id = model_id
        self.num_inference_steps = num_inference_steps
        self.pipeline = None  # Lazy-loaded
        self.change_map_gen = ChangeMapGenerator()

    def _load_pipeline(self):
        """Lazy-load the inpainting pipeline."""
        from diffusers import StableDiffusionInpaintPipeline

        print(f"[disease] Loading {self.model_id} on {self.device}...")

        # MPS requires float32, CUDA can use float16
        dtype = torch.float32 if self.device == "mps" else torch.float16

        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,  # Disable NSFW filter for plant images
            requires_safety_checker=False,
        )
        self.pipeline.to(self.device)

        # Memory optimizations
        self.pipeline.enable_attention_slicing()

        if self.device == "cuda":
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("[disease] xformers enabled")
            except Exception:
                pass  # xformers not available

        print(f"[disease] Pipeline loaded ({dtype})")

    def apply(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        disease_type: DiseaseType | str,
        severity: float = 0.5,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        pattern: str = "patchy",
        guidance_scale: float = 7.5,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Apply disease to healthy plant image.

        Args:
            rgb: [H, W, 3] uint8 healthy plant image
            alpha: [H, W] uint8 plant mask (255 = plant)
            disease_type: Type of disease to apply
            severity: 0.0-1.0 disease severity
            seed: Random seed for reproducibility
            num_inference_steps: Diffusion steps (uses default if None)
            pattern: Change map pattern ("patchy", "edge", "uniform")
            guidance_scale: Classifier-free guidance scale

        Returns:
            diseased_rgb: [H, W, 3] uint8 diseased image
            change_map: [H, W] uint8 change map (where disease was applied)
            metadata: dict with disease application details
        """
        if self.pipeline is None:
            self._load_pipeline()

        # Convert string to enum if needed
        if isinstance(disease_type, str):
            disease_type = DiseaseType(disease_type)

        config = DISEASE_CONFIGS[disease_type]
        steps = num_inference_steps or self.num_inference_steps

        # Store original size
        orig_h, orig_w = rgb.shape[:2]

        # Resize to 512x512 for SD (much faster)
        rgb_pil = Image.fromarray(rgb).resize((512, 512), Image.Resampling.LANCZOS)
        alpha_resized = np.array(
            Image.fromarray(alpha).resize((512, 512), Image.Resampling.LANCZOS)
        )

        # Generate change map
        plant_mask = (alpha_resized > 127).astype(np.float32)
        if seed is not None:
            self.change_map_gen.seed = seed
        change_map = self.change_map_gen.generate(plant_mask, severity, pattern)

        # Convert change map to PIL mask (0 = keep, 255 = inpaint)
        # SD inpainting expects white = inpaint region
        mask_pil = Image.fromarray((change_map * 255).astype(np.uint8))

        # Run inpainting
        generator = torch.Generator(self.device).manual_seed(seed) if seed else None

        result = self.pipeline(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            image=rgb_pil,
            mask_image=mask_pil,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        # Resize back to original
        diseased_rgb = np.array(result.resize((orig_w, orig_h), Image.Resampling.LANCZOS))

        # Upscale change map for output
        change_map_full = np.array(
            Image.fromarray((change_map * 255).astype(np.uint8)).resize(
                (orig_w, orig_h), Image.Resampling.LANCZOS
            )
        )

        # Composite: only apply disease within plant mask, preserve background
        alpha_norm = alpha.astype(np.float32) / 255.0
        diseased_rgb = (
            diseased_rgb.astype(np.float32) * alpha_norm[:, :, None] +
            rgb.astype(np.float32) * (1 - alpha_norm[:, :, None])
        ).clip(0, 255).astype(np.uint8)

        metadata = {
            "disease_type": disease_type.value,
            "severity": severity,
            "pattern": pattern,
            "seed": seed,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "coverage": float(change_map.mean()),
        }

        return diseased_rgb, change_map_full, metadata
