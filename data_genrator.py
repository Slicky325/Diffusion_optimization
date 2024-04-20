from PIL import Image
import os
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
import torch
import gc
import numpy as np

sample_prompts = [
    "cute white mouse pilot wearing aviator glasses holding a flight helmet",
    "A dignified beaver wearing glasses, a vest, and colorful necktie. He stands next to a tall stack of books in a library",
    "Horses pulling a carriage on the moon's surface, with the Statue of Liberty and Great Pyramid in the background. Planet Earth can be seen in the sky.",
    "A raccoon wearing formal clothes, wearing a tophat and holding a cane. The raccoon is holding a garbage bag. Oil painting in the style of Madhubani art.",
    "A smiling sloth wearing a leather jacket, a cowboy hat, a kilt and a bowtie. The sloth is holding a quarterstaff and a big book. The sloth stands a few feet in front of a shiny VW van. The van has a cityscape painted on it and parked on grass.",
    "A photo of a hamburger fighting a hot dog in a boxing ring. The hot dog is tired and up against the ropes.",
    "A teddy bear wearing a motorcycle helmet and cape is car surfing on a taxi cab in New York City.",
    "a photograph of Mona lisa drinking coffee as she has her breakfast. her plate has an omelet and croissant.",
    "A bowl of soup that looks like a monster with tofu says deep learning.",
    "A large yellow triangle above a green square and red rectangle"
    ]

seed = 0
generator = torch.manual_seed(seed)

model_ckpt = "CompVis/stable-diffusion-v1-4"
device = "cuda"
weight_dtype = torch.float16
sd_pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt, torch_dtype=weight_dtype).to(device)

for i in sample_prompts:
    images = sd_pipeline([i], num_images_per_prompt=10, generator=generator, output_type="numpy").images
    os.mkdir(f"Dataset/{i}")
    for j in range(len(images)):
      img = images[j]
      img = Image.fromarray((img * 255).astype(np.uint8))
      img.save(f"Dataset/{i}/{j}.png")
    torch.cuda.empty_cache()
    gc.collect()