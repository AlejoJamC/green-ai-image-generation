import os
import torch
import numpy as np
from dotenv import load_dotenv
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from huggingface_hub import login

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)

device = "cpu"

print("Loading FLUX model...")
pipe = FluxKontextPipeline.from_pretrained(
    #"black-forest-labs/FLUX.1-Kontext-dev",
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.enable_model_cpu_offload()
pipe = pipe.to(device)

# Disable problematic attention optimization
pipe.transformer.enable_xformers_memory_efficient_attention = False

input_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
)

print("Generating image...")
result = pipe(
    prompt="Add a small red hat on the cat's head",
    image=input_image,
    num_inference_steps=20,
    guidance_scale=3.5
).images[0]

# Convert to numpy and check values
result_np = np.array(result)
print(f"Min: {result_np.min()}, Max: {result_np.max()}, Mean: {result_np.mean()}")

# If values are NaN/inf, the generation failed
if np.isnan(result_np).any() or np.isinf(result_np).any():
    print("ERROR: Generated image contains invalid values")
else:
    result.save("cat_with_hat.png")
    print("Image saved successfully")
