import os
import requests
from bs4 import BeautifulSoup
import html2text
from urllib.parse import urljoin
import pandas as pd
from tqdm import tqdm

from app.modules.scraper.utils import download_image, zip_selected_folders
from app.utils.func import try_or_default

def get_urls_from_sitemap(url, scrap_url) -> list:
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'xml')

    return [item.text for item in soup.find_all('loc') if item.text.startswith(scrap_url)]

def scrape_articles(sitemap_url, scrap_url, dataset_storage_url):

    ROOT_DIR = os.environ['ROOT_DIR']
    df_path = os.path.join(ROOT_DIR, f"{dataset_storage_url}/doc_dataset/articles_html.csv")
    articles_df = pd.read_csv(df_path)
    articles_df.set_index('Url', inplace=True)

    # print(articles_df.columns)
    existing_links = articles_df.index
    links = get_urls_from_sitemap(sitemap_url, scrap_url)

    links_to_pull = list(set(links) - set(existing_links))
    print(f"Identified {len(links_to_pull)} articles to pull")

    #TODO: remove
    links_to_pull = links_to_pull[:10]

    data = []
    for link in tqdm(links_to_pull):
        data.append((link, *extract_content_from_link(link, scrap_url, dataset_storage_url)))
    df = pd.DataFrame(data, columns =['Url', 'Title', 'Publication_Date', 'Content'])
    df.set_index('Url', inplace=True)

    df.drop_duplicates(inplace=True)
    combined_df = pd.concat([articles_df, df], axis=0)
    # print(combined_df.columns, len(articles_df), len(df), len(combined_df))

    combined_df.to_csv(df_path)
    print(f"Saved dataset to: {df_path}")
    ###################
    # new_df = deduped_df[(deduped_df['Title'] != '')]

    # folders_clean = [url.replace('https://www.deeplearning.ai/the-batch/', '') for url in new_df.index]

    # Usage
    # output_filename = 'images_clean.zip'
    # base_directory = 'article_images'  # Replace with your base directory path

    # zip_selected_folders(output_filename, folders_clean, base_directory)

    # return base_directory

def extract_and_download_images(soup: BeautifulSoup, base_url, scrap_url, dataset_storage_url):
    images = soup.find_all("img", src=True)

    ROOT_DIR = os.environ['ROOT_DIR']
    folder_path = os.path.join(ROOT_DIR, f"{dataset_storage_url}/doc_dataset/images_clean", base_url.replace(scrap_url, ''))

    # folder_path = os.path.join('article_images', base_url.replace(scrap_url, ''))

    for img_tag in images:
        img_url = urljoin(base_url, img_tag["src"])
        download_image(img_url, folder_path)


def extract_content_from_link(url, scrap_url, dataset_storage_url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    article = soup.select_one('article')

    title = try_or_default(lambda: article.select_one("h1.leading-tight").get_text(), '')

    publication_date = try_or_default(lambda: article.select_one('.container--boxed > div > div:first-child > div:last-child > div:first-child > div:last-child').get_text(), '')

    for data in article(['style', 'script', 'aside', 'footer', 'nav']) + soup.find_all("section", id="subscribe"):
        # Remove tags
        data.decompose()

    extract_and_download_images(article, url, scrap_url, dataset_storage_url)

    for data in article(['header']):
        # Remove tags
        data.decompose()

    return title, publication_date, html2text.html2text(' '.join(soup.stripped_strings))

