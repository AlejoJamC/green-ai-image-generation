import torch
from diffusers import FluxPipeline
from diffusers.utils import load_image

MODEL_FILE = ".lmstudio/models/bullerwins/FLUX.1-Kontext-dev-GGUF/flux1-kontext-dev-Q8_0.gguf"

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device != "cpu" else torch.float32
print(f"Using device: {device}")
print(f"Loading: {MODEL_FILE}")

pipe = FluxPipeline.from_single_file(MODEL_FILE, torch_dtype=dtype).to(device)

# Loading the cat image
input_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
)

print("Adding hat to cat...")
image = pipe(
    image=input_image,
    prompt="Add a hat to the cat",
    guidance_scale=2.5
).images[0]

image.save("cat_with_hat.png")
print("Image saved as cat_with_hat.png")
