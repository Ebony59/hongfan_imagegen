import requests
import pandas as pd
import os
import pathlib

from bs4 import BeautifulSoup
from tqdm import tqdm

# Define headers to mimic a real browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

BASE_URL = 'https://www.athome.com/'
CATEGORIES = ['statues-sculptures', 'outdoor-wall-decor', 'yard-art', 'outdoor-fountains', 'wind-chimes', 
              'vases', 'sculptures-figurines', 'candle-holders', 'decorative-plates-bowls-trays']

def scrape_large_image_url(url):
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to retrieve page: {url}")
        return "No Image Found"

    soup = BeautifulSoup(response.text, "html.parser")
    img_url = soup.find("a", class_="MagicZoom")['href']
    return img_url

def scrape_product_page(url):
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to retrieve page: {url}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    products = []

    for product in soup.find_all("div", class_='plp-tile-container'):
        try:
            title = product.find("h3", class_='pdp-link pt-name').text.strip()
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
        size = 24
        max_batch = 500
        
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