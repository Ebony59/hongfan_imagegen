import os
import requests
import pandas as pd

from PIL import Image
from io import BytesIO

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

def request_image(image_path):
    response = requests.get(image_path, timeout=10)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")

    return image