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
    prompt="Add a small red hat on the cat's head",
    image=input_image,
    strength=0.5,
    num_inference_steps=20
).images[0]

result.save("cat_with_hat.png")
print("Image saved")
