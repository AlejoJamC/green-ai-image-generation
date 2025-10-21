import os
import torch
from dotenv import load_dotenv
from diffusers import FluxPipeline
from huggingface_hub import login

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)

device = "cpu"
dtype = torch.float32

print("Loading FLUX.1-schnell (fastest FLUX variant)...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=dtype
)
pipe.enable_model_cpu_offload()
pipe = pipe.to(device)

prompt = """Professional LinkedIn banner image, 1584x396 pixels, 
corporate banking theme, deep orange and navy blue geometric patterns, 
modern minimalist Dutch design aesthetic, technology and sustainability symbols, 
clean professional layout, abstract network connections, wind turbine silhouettes, 
high quality, detailed"""

print("Generating LinkedIn banner...")
result = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=0.0,
    height=512,
    width=2048
).images[0]

result.save("linkedin_banner_baseline.png")
print("Banner saved as linkedin_banner_baseline.png")

