import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

device = "cpu"
dtype = torch.float32

print("Loading Qwen Image Edit model...")
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=dtype
)
pipe = pipe.to(device)

prompt = "Add a small red hat on the cat's head"
input_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
)

print("Generating image...")
image = pipe(image=input_image, prompt=prompt).images[0]
image.save("cat_with_hat.png")
print("Image saved as cat_with_hat.png")
