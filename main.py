import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipe(
    image=input_image,
    prompt="Add a hat to the cat",
    guidance_scale=2.5
).images[0]

image.save("cat_with_hat.png")
print("Image saved as cat_with_hat.png")
