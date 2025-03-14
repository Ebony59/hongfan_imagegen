import requests
import pandas as pd
import os
import pathlib

from bs4 import BeautifulSoup
from tqdm import tqdm

from utils.config import HEADERS
from utils.retailer_utils import fetch_url

BASE_URL = 'https://www.athome.com/'
CATEGORIES = ['decorative-plates-bowls-trays', 'sculptures-figurines', 'vases', 'decorative-glass-balls']

def scrape_large_image_url(url):
    response = fetch_url(url, headers=HEADERS)

    if not response:
        return "No Image Found"

    soup = BeautifulSoup(response.text, "html.parser")
    img_url = soup.find("a", class_="MagicZoom")['href']
    return img_url

def scrape_product_page(url):
    response = fetch_url(url, headers=HEADERS)
    if not response:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    products = []

    for product in soup.find_all("div", class_='plp-tile-container'):
        try:
            title_element = product.find("h3", class_='pdp-link pt-name')
            if title_element is None:
                continue
            title = title_element.text.strip()
            link = product.find("a", class_="product-image")["href"]
            full_link = BASE_URL + link if link.startswith("/") else link

            img_url = scrape_large_image_url(full_link)
            
            products.append({
                "title": title,
                "img_url": img_url,
                "product_link": full_link
            })
        except Exception as e:
            print('Error:', e)
            continue
    
    return products

def get_scrape_url(category_url, start_index=0, size=100):
    return f'{category_url}?nav=top_nav&start={start_index}&sz={size}'

if __name__ == '__main__':
    repopath = pathlib.Path(__file__).resolve().parent
    for category in CATEGORIES:
        print(f"Scraping category: {category}")
        category_url = f'{BASE_URL}{category}/'

        start_index = 0
        size = 48
        max_batch = 50
        
        data = []
        for i in tqdm(range(max_batch)):
            url = get_scrape_url(category_url, start_index, size)
            products = scrape_product_page(url)
            data += products
            if len(products) < size:
                break
            start_index += size
        
        df = pd.DataFrame(data)
        out_csv = os.path.join(repopath, 'data', 'athome', f'{category}.csv')
        df.to_csv(out_csv, index=False)

    print("scraping completed!")
