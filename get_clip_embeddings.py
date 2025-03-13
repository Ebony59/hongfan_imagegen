import os
import pandas as pd
import torch
import clip
import requests
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from io import BytesIO

# Load CLIP model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)

DATA_DIR = "/workspace/hongfan_imagegen/data/athome"

def load_product_data(data_dir):
    all_data = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            df['category'] = file.split('.')[0]
            all_data.append(df)
    
    # Combine all category data into one DataFrame
    return pd.concat(all_data, ignore_index=True)

def get_clip_embedding(image_path, category, title):
    response = requests.get(image_path, timeout=10)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    
    # Preprocess image
    image = PREPROCESS(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_feature = MODEL.encode_image(image)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

    text_tokenized = clip.tokenize([category, title]).to(DEVICE)
    with torch.no_grad():
        text_feature = MODEL.encode_text(text_tokenized)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        return image_feature.cpu().numpy(), text_feature.cpu().numpy()


if __name__ == "__main__":
    df = load_product_data(DATA_DIR)
    print(f"Loaded {len(df)} products from At Home.")

    image_embeddings = []
    text_embeddings = []
    
    for i in tqdm(range(len(df))):
        try:
            image_embedding, text_embedding = get_clip_embedding(df.loc[i, 'img_url'], df.loc[i, 'category'], df.loc[i, 'title'])
        except Exception as e:
            print(f"Error processing {df.loc[i, 'img_url']}: {e}")
            continue
        image_embeddings.append(image_embedding)
        text_embeddings.append(text_embedding) 

    np.save("/workspace/hongfan_imagegen/output/embeddings/athome_clip_image_embeddings.npy", image_embeddings)
    np.save("/workspace/hongfan_imagegen/output/embeddings/athome_clip_text_embeddings.npy", text_embeddings)
    
    print("CLIP embeddings saved successfully!")
    