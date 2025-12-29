"""
Disease synthesis using SDXL img2img + LoRA.

Applies simulated plant diseases to healthy rendered images with controllable:
- Disease type (powdery_mildew, leaf_spot, rust, chlorosis, blight)
- Severity (0.0-1.0 continuous scale)
- Spatial distribution (auto-generated via Perlin noise + plant mask)

Uses SDXL base model with img2img + LoRA fine-tuned on PlantSeg.
Supports CUDA, MPS, and CPU backends.
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


# Trigger words must match prepare_disease_data.py TRIGGER_WORDS
TRIGGER_WORDS = {
    "powdery_mildew": "sks_mildew",
    "rust": "sks_rust",
    "leaf_spot": "sks_spot",
    "blight": "sks_blight",
    "chlorosis": "sks_chlorosis",
}

# Disease prompts for SDXL inpainting with LoRA trigger words
DISEASE_CONFIGS: dict[DiseaseType, DiseaseConfig] = {
    DiseaseType.POWDERY_MILDEW: DiseaseConfig(
        name="powdery_mildew",
        prompt="a sks_mildew plant disease, white powdery fungal coating on leaf surface, detailed texture",
        negative_prompt="healthy green leaf, clean surface, no disease",
        typical_severity=(0.2, 0.7),
    ),
    DiseaseType.LEAF_SPOT: DiseaseConfig(
        name="leaf_spot",
        prompt="a sks_spot plant disease, brown circular necrotic lesions with yellow halos, detailed",
        negative_prompt="healthy leaf, uniform green color, no spots",
        typical_severity=(0.2, 0.6),
    ),
    DiseaseType.RUST: DiseaseConfig(
        name="rust",
        prompt="a sks_rust plant disease, orange-brown rust pustules and spores on leaf, detailed texture",
        negative_prompt="healthy green leaf, no rust, no orange",
        typical_severity=(0.2, 0.6),
    ),
    DiseaseType.CHLOROSIS: DiseaseConfig(
        name="chlorosis",
        prompt="a sks_chlorosis plant disease, yellow discoloration and chlorotic leaves, fading color",
        negative_prompt="deep green healthy leaf, vibrant color",
        typical_severity=(0.3, 0.8),
    ),
    DiseaseType.BLIGHT: DiseaseConfig(
        name="blight",
        prompt="a sks_blight plant disease, dark brown-black necrotic tissue decay, dead tissue",
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
    """Applies disease symptoms to healthy plant images using SDXL img2img + LoRA."""

    # Model variant - base model for img2img (LoRA trained on this)
    SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(
        self,
        device: str | None = None,
        model_id: str = SDXL_BASE,
        lora_path: str | Path | None = None,
        lora_dir: str | Path | None = None,
        lora_scale: float = 1.0,
        num_inference_steps: int = 30,
    ):
        """
        Args:
            device: PyTorch device (auto-detect if None)
            model_id: HuggingFace model ID for SDXL base
            lora_path: Path to single LoRA weights (.safetensors)
            lora_dir: Directory with per-disease LoRAs ({disease}.safetensors)
            lora_scale: LoRA influence strength (0.0-1.0)
            num_inference_steps: Default diffusion steps (more = better quality, slower)
        """
        self.device = device or get_device()
        self.model_id = model_id
        self.lora_path = Path(lora_path) if lora_path else None
        self.lora_dir = Path(lora_dir) if lora_dir else None
        self.lora_scale = lora_scale
        self.num_inference_steps = num_inference_steps
        self.pipeline = None  # Lazy-loaded
        self.change_map_gen = ChangeMapGenerator()
        self._loaded_lora: str | None = None  # Track which LoRA is loaded

    def _load_pipeline(self):
        """Lazy-load the img2img pipeline."""
        from diffusers import StableDiffusionXLImg2ImgPipeline

        print(f"[disease] Loading {self.model_id} on {self.device}...")

        # MPS requires float32, CUDA can use float16
        dtype = torch.float32 if self.device == "mps" else torch.float16

        self.pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
        self.pipeline.to(self.device)

        # Load single LoRA if specified (not per-disease)
        if self.lora_path:
            print(f"[disease] Loading LoRA from {self.lora_path}...")
            self.pipeline.load_lora_weights(str(self.lora_path))
            self._loaded_lora = str(self.lora_path)
            print(f"[disease] LoRA loaded (scale={self.lora_scale})")

        # Memory optimizations
        self.pipeline.enable_attention_slicing()

        if self.device == "cuda":
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("[disease] xformers enabled")
            except Exception:
                pass  # xformers not available

        print(f"[disease] Pipeline loaded ({dtype})")

    def _load_disease_lora(self, disease_type: DiseaseType) -> None:
        """Load disease-specific LoRA from lora_dir."""
        if not self.lora_dir:
            return

        lora_file = self.lora_dir / f"{disease_type.value}.safetensors"
        if not lora_file.exists():
            print(f"[disease] Warning: LoRA not found for {disease_type.value}: {lora_file}")
            return

        # Skip if already loaded
        if self._loaded_lora == str(lora_file):
            return

        # Unload previous LoRA
        if self._loaded_lora:
            self.pipeline.unload_lora_weights()

        # Load new LoRA
        print(f"[disease] Loading LoRA for {disease_type.value}...")
        self.pipeline.load_lora_weights(str(lora_file))
        self._loaded_lora = str(lora_file)
        print(f"[disease] LoRA loaded (scale={self.lora_scale})")

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
        lora_scale: float | None = None,
        strength: float = 0.6,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Apply disease to healthy plant image using img2img.

        Args:
            rgb: [H, W, 3] uint8 healthy plant image
            alpha: [H, W] uint8 plant mask (255 = plant)
            disease_type: Type of disease to apply
            severity: 0.0-1.0 disease severity (controls affected area)
            seed: Random seed for reproducibility
            num_inference_steps: Diffusion steps (uses default if None)
            pattern: Change map pattern ("patchy", "edge", "uniform")
            guidance_scale: Classifier-free guidance scale
            lora_scale: Override LoRA influence (uses self.lora_scale if None)
            strength: img2img strength (0=no change, 1=full regeneration)

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

        # Load disease-specific LoRA if using lora_dir
        self._load_disease_lora(disease_type)

        config = DISEASE_CONFIGS[disease_type]
        steps = num_inference_steps or self.num_inference_steps

        # Store original size
        orig_h, orig_w = rgb.shape[:2]

        # Resize to model's native resolution (1024 for SDXL)
        target_size = 1024
        rgb_pil = Image.fromarray(rgb).resize((target_size, target_size), Image.Resampling.LANCZOS)
        alpha_resized = np.array(
            Image.fromarray(alpha).resize((target_size, target_size), Image.Resampling.LANCZOS)
        )

        # Generate change map (determines where disease appears)
        plant_mask = (alpha_resized > 127).astype(np.float32)
        if seed is not None:
            self.change_map_gen.seed = seed
        change_map = self.change_map_gen.generate(plant_mask, severity, pattern)

        # Run img2img
        generator = torch.Generator(self.device).manual_seed(seed) if seed else None

        # Build kwargs
        pipeline_kwargs = {
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "image": rgb_pil,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "generator": generator,
        }

        # Add LoRA scale if LoRA is loaded
        if self.lora_path:
            scale = lora_scale if lora_scale is not None else self.lora_scale
            pipeline_kwargs["cross_attention_kwargs"] = {"scale": scale}

        result = self.pipeline(**pipeline_kwargs).images[0]
        result_np = np.array(result)

        # Resize back to original
        result_resized = np.array(result.resize((orig_w, orig_h), Image.Resampling.LANCZOS))

        # Upscale change map for blending
        change_map_full = np.array(
            Image.fromarray((change_map * 255).astype(np.uint8)).resize(
                (orig_w, orig_h), Image.Resampling.LANCZOS
            )
        )
        blend_mask = change_map_full.astype(np.float32) / 255.0

        # Blend: apply disease only where change_map indicates
        # change_map = 1 -> use generated result, change_map = 0 -> use original
        diseased_rgb = (
            result_resized.astype(np.float32) * blend_mask[:, :, None] +
            rgb.astype(np.float32) * (1 - blend_mask[:, :, None])
        ).clip(0, 255).astype(np.uint8)

        # Also mask to plant area (preserve background completely)
        alpha_norm = alpha.astype(np.float32) / 255.0
        diseased_rgb = (
            diseased_rgb.astype(np.float32) * alpha_norm[:, :, None] +
            rgb.astype(np.float32) * (1 - alpha_norm[:, :, None])
        ).clip(0, 255).astype(np.uint8)

        metadata = {
            "disease_type": disease_type.value,
            "severity": severity,
            "strength": strength,
            "pattern": pattern,
            "seed": seed,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "coverage": float(change_map.mean()),
            "model": "sdxl_img2img",
            "lora": str(self.lora_path) if self.lora_path else None,
            "lora_scale": lora_scale if lora_scale is not None else self.lora_scale,
        }

        return diseased_rgb, change_map_full, metadata
