import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import json
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
    metadata = []
    metadata_path = os.path.join(image_folder,'metadata.jsonl')

    if Path(metadata_path).is_file():
        os.remove(metadata_path)

    for style in os.listdir(image_folder):
        if ".ipynb_checkpoints" in style:
            continue

        style_folder = os.path.join(image_folder, style)
        for img in os.listdir(style_folder):
            if ".ipynb_checkpoints" in img:
                continue
            caption = generate_caption(os.path.join(style_folder, img))
            relative_path = os.path.join(style, img)
            print(f"{relative_path}/{img}: {caption}")
            caption = caption + f". In the style of HF{style}"
            metadata.append({"file_name": relative_path, "text": caption})

    with open(metadata_path, 'w', encoding='utf-8') as f:
        for entry in metadata:
            f.write(json.dumps(entry)+"\n")