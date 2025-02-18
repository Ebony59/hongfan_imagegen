import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    caption = model.generate(**inputs)
    return processor.decode(caption[0], skip_special_tokens=True)

if __name__ == "__main__":
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

    image_folder = "/workspace/hongfan_imagegen/datasets/designs/"
    caption_file = "/workspace/hongfan_imagegen/datasets/caption.txt"

    if Path(caption_file).is_file():
        os.remove(caption_file)

    for style in os.listdir(image_folder):
        if ".ipynb_checkpoints" in style:
            continue

        style_folder = os.path.join(image_folder, style)
        for img in os.listdir(style_folder):
            if ".ipynb_checkpoints" in img:
                continue
            caption = generate_caption(os.path.join(style_folder, img))
            print(f"{style_folder}/{img}: {caption}")
            caption = caption + f". In the style of HF{style}"
            with open(caption_file, "a") as f:
                f.write(f'{style_folder}/{img}\t{caption}\n')