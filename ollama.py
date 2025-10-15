import os
import torch
import time
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline
from huggingface_hub import login

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32  # CPU solo soporta float32
print(f"Using device: {device}")

print("Loading Stable Diffusion 1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
)
pipe = pipe.to(device)

prompt = """Professional LinkedIn banner, corporate banking, 
orange and navy blue colors, geometric pattern, modern design"""

print("Generating image...")
start = time.time()

image = pipe(
    prompt=prompt,
    height=512,
    width=512,
    num_inference_steps=25
).images[0]

generation_time = time.time() - start

image.save("linkedin_banner_baseline.png")
print(f"✓ Generated in {generation_time:.1f}s")
