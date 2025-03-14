import os
import pandas as pd
import requests
from tqdm import tqdm

import pathlib

from utils.retailer_utils import fetch_url

REPO_PATH = pathlib.Path(__file__).resolve().parent
RETAILER = 'athome'

def download_image(url, save_path):
    response = fetch_url(url)
    
    if not response:
        return False

    with open(save_path, 'wb') as f:
        f.write(response.content)
    return True


if __name__ == "__main__":
    # Define paths
    data_dir = os.path.join(REPO_PATH, 'data', RETAILER)
    output_dir = os.path.join(REPO_PATH, 'datasets', 'retailers', RETAILER)
    
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    for csv_file in csv_files:
        category = csv_file.replace('.csv', '')
        category_dir = os.path.join(output_dir, category)

        os.makedirs(category_dir, exist_ok=True)

        df = pd.read_csv(os.path.join(data_dir, csv_file))

        for i, url in tqdm(enumerate(df["img_url"]), desc=f"Processing {category}", total=len(df)):
            save_path = os.path.join(category_dir, f"image_{i}.jpg")
            download_image(url, save_path)