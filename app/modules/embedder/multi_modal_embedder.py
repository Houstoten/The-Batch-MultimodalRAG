import os
import pandas as pd
import nltk
from tqdm import tqdm
from bs4 import BeautifulSoup
from app.modules.embedder.utils import sliding_window

from langchain_core.documents import Document


def embed_content(dataset_storage_url, descriptions_dataset_storage_url, vector_store):
    doc_list = content_to_documents(dataset_storage_url, descriptions_dataset_storage_url)
    vector_store.add_documents(doc_list)


def content_to_documents(dataset_storage_url, descriptions_dataset_storage_url):
    ROOT_DIR = os.environ['ROOT_DIR']
    articles_df_path = os.path.join(ROOT_DIR, f"{dataset_storage_url}/doc_dataset/articles_html.csv")
    articles_df = pd.read_csv(articles_df_path)

    descriptions_df_path = os.path.join(ROOT_DIR, f"{descriptions_dataset_storage_url}/image_descriptions_dataset/image_captions.csv")
    descriptions_df = pd.read_csv(descriptions_df_path)
    descriptions_df.set_index('URL', inplace=True)

    doc_list = create_chunks(articles_df, descriptions_df)
    return doc_list

def create_chunks(articles_df, descriptions_df):
    nltk.download('punkt_tab')

    processed_combined = []
    for index, row in tqdm(articles_df.iterrows(), total=articles_df.shape[0]):    
        soup = BeautifulSoup(row['Content'], "html.parser")
        text_content = soup.get_text()
        processed_combined.append(text_content)

    document_chunk_list = []

    window_size = 2 
    overlap = 1     

    print('\nChunking articles')
    for index, row in tqdm(articles_df.iterrows(), total=articles_df.shape[0]):    
        soup = BeautifulSoup(row['Content'], "html.parser")
        text_content = soup.get_text()

        sentences = nltk.tokenize.sent_tokenize(text_content)
        chunks = sliding_window(sentences, window_size, overlap)

        for i, chunk in enumerate(chunks):
            document_chunk_list.append(
                Document(
                    page_content=''.join(chunk), 
                    metadata={
                        "split_id": i, 
                        "doc_id": index, 
                        "type": "text", 
                        "image_url": ''
                        }))

    print('\nChunking image descriptions')
    for index, row in tqdm(descriptions_df.iterrows(), total=descriptions_df.shape[0]): 

        sentences = nltk.tokenize.sent_tokenize(row['Description'])
        chunks = sliding_window(sentences, window_size, overlap)

        for i, chunk in enumerate(chunks):
            document_chunk_list.append(
                Document(
                    page_content=''.join(chunk), 
                    metadata={
                        "split_id": i, 
                        "doc_id": f'https://www.deeplearning.ai/the-batch/{index.split("/")[0]}/', 
                        "type": "image", 
                        "image_url": index
                        }))
    
    return document_chunk_list

