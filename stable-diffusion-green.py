import os
import warnings
import torch
import time
import psutil
from pathlib import Path
from dotenv import load_dotenv
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import login

warnings.filterwarnings('ignore')

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)

device = "cpu"

CPU_TDP_WATTS = 65
CO2_INTENSITY_G_PER_KWH = 350

print("Loading Stable Diffusion XL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32
)
pipe.enable_model_cpu_offload()
pipe = pipe.to(device)

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
model_dirs = list(cache_dir.glob("models--stabilityai--stable-diffusion-xl-base-1.0"))
model_size_gb = 0
if model_dirs:
    model_path = model_dirs[0]
    total_bytes = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
    model_size_gb = total_bytes / (1024 ** 3)

prompt = """Professional LinkedIn banner image, 1584x396 pixels, 
corporate banking theme, deep orange and navy blue geometric patterns, 
modern minimalist Dutch design aesthetic, technology and sustainability symbols, 
clean professional layout, abstract network connections, wind turbine silhouettes, 
high quality, detailed"""

process = psutil.Process()
initial_memory = process.memory_info().rss / (1024 ** 3)

print("Generating LinkedIn banner...")
start_time = time.time()

result = pipe(
    prompt=prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    height=512,
    width=2048
).images[0]

end_time = time.time()
peak_memory = process.memory_info().rss / (1024 ** 3)

generation_time = end_time - start_time
memory_used = peak_memory - initial_memory

energy_kwh = (CPU_TDP_WATTS * generation_time / 3600) / 1000
co2_grams = energy_kwh * CO2_INTENSITY_G_PER_KWH

result.save("linkedin_banner_sdxl_baseline.png")

print(f"\n=== Sustainability Metrics ===")
print(f"Generation time: {generation_time:.2f}s")
print(f"Peak memory usage: {peak_memory:.2f} GB")
print(f"Memory increase: {memory_used:.2f} GB")
print(f"Model size on disk: {model_size_gb:.2f} GB")
print(f"Estimated energy: {energy_kwh * 1000:.4f} Wh (based on {CPU_TDP_WATTS}W TDP)")
print(f"Estimated CO2: {co2_grams:.2f}g (based on {CO2_INTENSITY_G_PER_KWH}g/kWh grid)")
print(f"\nNote: Energy/CO2 are estimates")
print("\nBanner saved as linkedin_banner_sdxl_baseline.png")
