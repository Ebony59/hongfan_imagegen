import os
import requests
import pandas as pd
import time

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

def fetch_url(url, headers=None, max_retries=5):
    for attempt in range(max_retries):
        try:
            if headers:
                response = requests.get(url, headers=headers)
            else:
                response = requests.get(url)
            if response.status_code == 200:
                return response
            else:
                print(f"Attempt {attempt + 1}/{max_retries} with status {response.status_code}")
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} encountered an error: {e}")

        time.sleep(2 ** attempt)

    print(f"Failed to retrieve page after {max_retries} attempts: {url}")
    return None

def request_image(image_path):
    response = fetch_url(image_path)

    if not response:
        return False

    image = Image.open(BytesIO(response.content)).convert("RGB")

    return image