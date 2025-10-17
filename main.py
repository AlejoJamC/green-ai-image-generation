import os
import torch
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
    "black-forest-labs/FLUX.1-Kontext-dev",
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

result.save("cat_with_hat.png")
print("Image saved")
