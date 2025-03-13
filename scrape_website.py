import requests
import pandas as pd
import os

from bs4 import BeautifulSoup
from tqdm import tqdm

# Define headers to mimic a real browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

BASE_URL = 'https://www.athome.com/'
CATEGORIES = ['statues-sculptures', 'outdoor-wall-decor', 'yard-art', 'outdoor-fountains', 'wind-chimes', 
              'vases', 'sculptures-figurines', 'candle-holders', 'decorative-plates-bowls-trays', 'candle-holders']

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
            img_tag = product.find("img", class_="tile-image")
            img_url = img_tag["src"] if img_tag else "No Image Found"
            link = product.find("a", class_="product-image")["href"]
            full_link = BASE_URL + link if link.startswith("/") else link
            
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
    for category in CATEGORIES:
        print(f"Scraping category: {category}")
        category_url = f'{BASE_URL}{category}/'
        url = get_scrape_url(category_url, size=100)
        data = scrape_product_page(url)
    
        df = pd.DataFrame(data)
        df.to_csv(f'/workspace/hongfan_imagegen/data/athome/{category}.csv', index=False)

    print("scraping completed!")