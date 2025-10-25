import os
import warnings
import torch
from pathlib import Path
from dotenv import load_dotenv
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import login
from pruna import optimize

warnings.filterwarnings('ignore')

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)

device = "cpu"

print("Loading Stable Diffusion XL (baseline model)...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32
)

print("Starting optimization with Pruna AI...")
print("This may take 10-30 minutes...")

optimized_pipe = optimize(
    pipe,
    optimization_config={
        "quantization": "int8",
        "pruning": True,
        "optimization_level": 2
    }
)

output_path = Path("./models/sdxl-optimized")
output_path.mkdir(parents=True, exist_ok=True)

print(f"Saving optimized model to {output_path}...")
optimized_pipe.save_pretrained(output_path)

original_size = sum(f.stat().st_size for f in Path.home().joinpath(".cache/huggingface/hub").glob("models--stabilityai--stable-diffusion-xl-base-1.0/**/*") if f.is_file())
optimized_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())

print(f"\n=== Optimization Complete ===")
print(f"Original model size: {original_size / (1024**3):.2f} GB")
print(f"Optimized model size: {optimized_size / (1024**3):.2f} GB")
print(f"Size reduction: {((original_size - optimized_size) / original_size * 100):.1f}%")
print(f"\nOptimized model saved to: {output_path}")
