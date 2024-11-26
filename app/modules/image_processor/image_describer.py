from tqdm import tqdm
import os
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

from app.modules.image_processor.utils import generate_img_summaries

def describe_images(dataset_storage_url, descriptions_dataset_storage_url):
    ROOT_DIR = os.environ['ROOT_DIR']
    articles_df_path = os.path.join(ROOT_DIR, f"{dataset_storage_url}/doc_dataset/articles_html.csv")
    articles_df = pd.read_csv(articles_df_path)

    descriptions_df_path = os.path.join(ROOT_DIR, f"{descriptions_dataset_storage_url}/image_descriptions_dataset/image_captions.csv")
    descriptions_df = pd.read_csv(descriptions_df_path)

    present_descriptions = set(descriptions_df['URL'].to_numpy())

    images_clean_path = os.path.join(ROOT_DIR, f"{dataset_storage_url}/doc_dataset/images_clean/")
    images_in_dataset = set()
    for dirname, _, filenames in os.walk(images_clean_path):
        for filename in filenames:
            images_in_dataset.add(os.path.join(dirname, filename).replace(images_clean_path, ''))

    images_to_describe = images_in_dataset - present_descriptions

    print(f"Found {len(images_to_describe)} to describe.")

    if len(images_to_describe) == 0:
        return

    summaries_new = []
    for url in tqdm(list(images_to_describe)):
        summaries_new.append([url, generate_summary_with_context(url, articles_df, images_clean_path)])

    summaries_new_np = np.array(summaries_new)

    df_new = pd.DataFrame(data=summaries_new_np,    
                columns=['URL', 'Description'])

    df_combined = pd.concat([descriptions_df, df_new], ignore_index=True)
    df_combined = df_combined[['URL', 'Description']]

    df_combined.to_csv(descriptions_df_path)

def generate_summary_with_context(url, articles_df, images_path):
    context = ""
    
    try:
        row = articles_df.loc[f'https://www.deeplearning.ai/the-batch/{url.split("/")[0]}/']
        soup = BeautifulSoup(row['Content'], "html.parser")
        context = soup.get_text()
    except:
        context = ""

    return generate_img_summaries(os.path.join(images_path, url), context=context)