import os
import torch
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image
from huggingface_hub import login

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)
else:
    print("Warning: HUGGINGFACE_TOKEN not found in .env file.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print("Loading Stable Diffusion 1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    ## "black-forest-labs/FLUX.1-Kontext-dev", TODO at this moment not supported by my local CPU
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype # Using float32 for better compatibility with CPU
)
pipe = pipe.to(device)

input_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipe(
    image=input_image,
    prompt="Add a hat to the cat",
    guidance_scale=2.5
).images[0]

image.save("cat_with_hat.png")
print("Image saved as cat_with_hat.png")
