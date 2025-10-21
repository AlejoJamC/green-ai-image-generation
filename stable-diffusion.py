import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
from PIL import Image, ImageDraw

device = "cpu"
dtype = torch.float32

print("Loading SDXL Inpaint...")
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=dtype
)
pipe = pipe.to(device)

input_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
).resize((1024, 1024))

mask = Image.new("L", (1024, 1024), 0)
draw = ImageDraw.Draw(mask)
draw.rectangle([300, 50, 700, 350], fill=255)

print("Generating...")
result = pipe(
    prompt="red hat",
    image=input_image,
    mask_image=mask,
    num_inference_steps=25,
    guidance_scale=8.0
).images[0]

result.save("cat_with_hat.png")
print("Image saved")
