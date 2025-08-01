import torch
from diffusers import StableDiffusionPipeline
from paths.paths import prompts_file, images_dir

def generate_images():
    
    num_images = 50

    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)

    with prompts_file.open("r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    for prompt in prompts:
        safe_name = prompt.replace(" ", "_")
        out_dir = images_dir / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)   
        for i in range(num_images):
            image = pipeline(prompt).images[0]
            image.save(str(out_dir / f"{i:04d}.png"))
    

