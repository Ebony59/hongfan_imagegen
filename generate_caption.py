import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from utils.retailer_utils import load_product_data

import pathlib
REPO_PATH = pathlib.Path(__file__).resolve().parent
RETAILER = 'athome'

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    caption = model.generate(**inputs)
    return processor.decode(caption[0], skip_special_tokens=True)

if __name__ == "__main__":
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

    image_folder = os.path.join(REPO_PATH, 'datasets', 'retailers', RETAILER)
    data_dir = os.path.join(REPO_PATH, 'data', RETAILER)

    for category in os.listdir(image_folder):
        csv_file = os.path.join(data_dir, f'{category}.csv')
        df = pd.read_csv(csv_file)
        
        if ".ipynb_checkpoints" in category:
            continue

        category_folder = os.path.join(image_folder, category)
        for img in os.listdir(category_folder):
            if ".ipynb_checkpoints" in img or '.txt' in img:
                continue
                
            try:
                caption = generate_caption(os.path.join(category_folder, img))
            except Exception as e:
                print(f'Error when processing {category}/{img}: {e}')
                continue
                
            print(f"{category}/{img}: {caption}")

            img_name = img.split('.')[0]
            row = int(img_name.split('_')[1])
            title = df.loc[row, 'title']
            
            caption = f'{title}, {caption}, under category {category}'
            caption_file = os.path.join(category_folder, f'{img_name}.txt')

            with open(caption_file, "w") as f:
                f.write(caption)