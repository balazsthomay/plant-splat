"""
Train LoRA on disease images using kohya-ss/sd-scripts.

Generates TOML configs and runs kohya training for SDXL.
Trains separate LoRA per disease type for better quality.

Setup (on training machine):
    git clone https://github.com/kohya-ss/sd-scripts tools/kohya
    cd tools/kohya
    pip install -r requirements.txt
    accelerate config  # answer: This machine, No, NO, NO, NO, all, fp16

Usage:
    # Generate configs for all 5 diseases
    uv run src/train_disease_lora.py --all --config-only

    # Single disease
    uv run src/train_disease_lora.py --data-dir data/disease_training/rust --config-only

    # Then on GPU machine (run each sequentially):
    for cfg in models/lora/*_train_config.toml; do
        accelerate launch sdxl_train_network.py --config_file="$cfg"
    done
"""

import argparse
import subprocess
import sys
from pathlib import Path


def generate_dataset_config(data_dir: Path, output_path: Path, num_repeats: int = 10) -> None:
    """Generate kohya dataset config TOML."""
    config = f'''[general]
resolution = 1024
caption_extension = ".txt"
shuffle_caption = false
enable_bucket = true
min_bucket_reso = 512
max_bucket_reso = 2048
bucket_reso_steps = 32
flip_aug = true
color_aug = false

[[datasets]]
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = "{data_dir.absolute()}"
  num_repeats = {num_repeats}
  caption_extension = ".txt"
'''
    output_path.write_text(config)
    print(f"[train] Dataset config: {output_path}")


def generate_train_config(
    output_dir: Path,
    output_name: str,
    dataset_config_path: Path,
    base_model: str,
    steps: int = 1000,
    rank: int = 32,
    learning_rate: float = 1e-4,
) -> Path:
    """Generate kohya training config TOML."""
    config_path = output_dir / f"{output_name}_train_config.toml"

    config = f'''# Model
pretrained_model_name_or_path = "{base_model}"
dataset_config = "{dataset_config_path.absolute()}"

# Output
output_dir = "{output_dir.absolute()}"
output_name = "{output_name}"
save_model_as = "safetensors"
save_every_n_steps = 500

# LoRA Network
network_module = "networks.lora"
network_dim = {rank}
network_alpha = {rank // 2}
network_train_unet_only = true

# Training
max_train_steps = {steps}
train_batch_size = 1
seed = 42

# Precision & Memory
mixed_precision = "fp16"
gradient_checkpointing = true
cache_latents = true
cache_latents_to_disk = true
cache_text_encoder_outputs = true
cache_text_encoder_outputs_to_disk = true
no_half_vae = true

# Optimizer
optimizer_type = "AdamW8bit"
learning_rate = {learning_rate}
lr_scheduler = "cosine"
lr_warmup_steps = 100

# xformers (comment out if not available)
xformers = true
'''
    config_path.write_text(config)
    print(f"[train] Train config: {config_path}")
    return config_path


def check_kohya() -> Path:
    """Check if kohya-ss is available."""
    kohya_path = Path("tools/kohya")
    train_script = kohya_path / "sdxl_train_network.py"

    if not train_script.exists():
        print("[train] kohya-ss not found. Install with:")
        print("    git clone https://github.com/kohya-ss/sd-scripts tools/kohya")
        print("    cd tools/kohya && pip install -r requirements.txt")
        print("    accelerate config")
        return None

    return kohya_path


def train_lora(
    data_dir: Path,
    output_dir: Path,
    steps: int = 500,
    rank: int = 32,
    learning_rate: float = 1e-4,
    num_repeats: int = 10,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    config_only: bool = False,
) -> None:
    """
    Train LoRA using kohya-ss.

    Args:
        data_dir: Directory with training images and .txt captions
        output_dir: Output directory for LoRA and configs
        steps: Training steps (500 per disease, ~15-20 min each on RTX 4090)
        rank: LoRA rank (network_dim), 32 for single concept
        learning_rate: Learning rate
        num_repeats: How many times to repeat each image per epoch
        base_model: Base SDXL model
        config_only: Just generate configs, don't run training
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output name = disease folder name
    output_name = data_dir.name

    # Count images
    n_images = len(list(data_dir.glob("*.jpg"))) + len(list(data_dir.glob("*.png")))
    n_captions = len(list(data_dir.glob("*.txt")))

    print(f"[train] Output: {output_name}")
    print(f"[train] Images: {n_images}, Captions: {n_captions}")
    print(f"[train] Steps: {steps}, Rank: {rank}, LR: {learning_rate}")
    print(f"[train] Repeats: {num_repeats} (effective: {n_images * num_repeats} samples)")

    # Generate configs
    dataset_config_path = output_dir / f"{output_name}_dataset.toml"
    generate_dataset_config(data_dir, dataset_config_path, num_repeats)

    train_config_path = generate_train_config(
        output_dir=output_dir,
        output_name=output_name,
        dataset_config_path=dataset_config_path,
        base_model=base_model,
        steps=steps,
        rank=rank,
        learning_rate=learning_rate,
    )

    if config_only:
        print(f"\n[train] Configs generated. Run training manually:")
        print(f"    cd tools/kohya")
        print(f"    accelerate launch sdxl_train_network.py --config_file=\"{train_config_path.absolute()}\"")
        return

    # Check kohya installation
    kohya_path = check_kohya()
    if kohya_path is None:
        print("\n[train] Generate configs with --config-only and train manually.")
        sys.exit(1)

    # Run training
    train_script = kohya_path / "sdxl_train_network.py"
    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process=2",
        str(train_script),
        f"--config_file={train_config_path}",
    ]

    print(f"\n[train] Starting training...")
    print(f"[train] Command: {' '.join(cmd[:4])}...")

    result = subprocess.run(cmd, cwd=kohya_path)

    if result.returncode != 0:
        print(f"[train] Training failed with code {result.returncode}")
        sys.exit(1)

    # Find output LoRA
    lora_files = list(output_dir.glob(f"{output_name}*.safetensors"))
    if lora_files:
        print(f"\n[train] Done! LoRA saved to:")
        for f in sorted(lora_files):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"    {f} ({size_mb:.1f} MB)")


DISEASE_TYPES = ["powdery_mildew", "rust", "leaf_spot", "blight", "chlorosis"]


def main():
    parser = argparse.ArgumentParser(
        description="Train LoRA on disease images using kohya-ss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate configs for all 5 diseases (recommended)
  uv run src/train_disease_lora.py --all --config-only

  # Single disease
  uv run src/train_disease_lora.py --data-dir data/disease_training/rust --config-only

  # Custom parameters
  uv run src/train_disease_lora.py \\
      --data-dir data/disease_training/powdery_mildew \\
      --steps 500 --rank 32 --lr 5e-5 --config-only
        """,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory with training images and .txt captions"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate configs for all 5 disease types"
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=Path("models/lora"),
        help="Output directory for LoRA (default: models/lora)"
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Training steps (default: 500, ~15-20 min per disease)"
    )
    parser.add_argument(
        "--rank", type=int, default=32,
        help="LoRA rank/dim (default: 32 for single concept)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--repeats", type=int, default=10,
        help="Image repeats per epoch (default: 10)"
    )
    parser.add_argument(
        "--base-model", type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model"
    )
    parser.add_argument(
        "--config-only", action="store_true",
        help="Only generate config files, don't run training"
    )

    args = parser.parse_args()

    # Handle --all flag
    if args.all:
        base_dir = Path("data/disease_training")
        if not base_dir.exists():
            print(f"[train] Error: Base directory not found: {base_dir}")
            sys.exit(1)

        print(f"[train] Generating configs for {len(DISEASE_TYPES)} diseases...\n")
        for disease in DISEASE_TYPES:
            data_dir = base_dir / disease
            if not data_dir.exists():
                print(f"[train] Warning: Skipping {disease} (not found)")
                continue

            train_lora(
                data_dir=data_dir,
                output_dir=args.output_dir,
                steps=args.steps,
                rank=args.rank,
                learning_rate=args.lr,
                num_repeats=args.repeats,
                base_model=args.base_model,
                config_only=args.config_only,
            )
            print()

        print(f"[train] All configs generated in {args.output_dir}")
        print(f"[train] Train on GPU with:")
        print(f"    for cfg in {args.output_dir}/*_train_config.toml; do")
        print(f"        accelerate launch sdxl_train_network.py --config_file=\"$cfg\"")
        print(f"    done")
        return

    # Single disease mode
    if args.data_dir is None:
        print("[train] Error: --data-dir required (or use --all)")
        sys.exit(1)

    if not args.data_dir.exists():
        print(f"[train] Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    train_lora(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        steps=args.steps,
        rank=args.rank,
        learning_rate=args.lr,
        num_repeats=args.repeats,
        base_model=args.base_model,
        config_only=args.config_only,
    )


if __name__ == "__main__":
    main()
