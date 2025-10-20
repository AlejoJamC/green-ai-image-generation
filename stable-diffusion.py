import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

device = "cpu"
dtype = torch.float32

print("Loading Stable Diffusion XL...")
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=dtype
)
pipe = pipe.to(device)

input_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
)

print("Generating image...")
result = pipe(
    prompt="a cat wearing a red hat on top of its head, detailed, photorealistic",
    negative_prompt="no hat, hatless, without hat",
    image=input_image,
    strength=0.75,  # Greater strength = more alteration
    num_inference_steps=30,
    guidance_scale=7.5 # Higher guidance scale = closer to prompt
).images[0]


result.save("cat_with_hat.png")
print("Image saved")
