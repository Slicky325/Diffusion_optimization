from PIL import Image
import os
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
import torch
import gc
import numpy as np

with open('prompts.txt') as f:
    sample_prompts = f.readlines()
for i in range(len(sample_prompts)):
    sample_prompts[i]=sample_prompts[i].rstrip("\n")

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