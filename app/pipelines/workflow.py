from app.modules.scraper import article_scraper
from app.modules.image_processor import image_describer
from app.modules.embedder import multi_modal_embedder
from app.modules.qa_chain_composer import articles_qa_runner

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
import chromadb
import os

sitemap_url = 'https://www.deeplearning.ai/sitemap-0.xml'
scrap_url = 'https://www.deeplearning.ai/the-batch/'

def setup_embedding_fn():
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'mps'}
    encode_kwargs = {"normalize_embeddings": True}

    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def setup_chroma():
    embedding_function = setup_embedding_fn()

    ROOT_DIR = os.environ['ROOT_DIR']
    vector_store = Chroma(
        collection_name="all_docs_collection",
        embedding_function=embedding_function,
        persist_directory=os.path.join(ROOT_DIR, "/chroma/")
    )
    return vector_store

def setup_preloaded_chroma():
    embedding_function = setup_embedding_fn()

    ROOT_DIR = os.environ['ROOT_DIR']
    persistent_client = chromadb.PersistentClient(path=os.path.join(ROOT_DIR, "chroma_preloaded/all_docs_collection"))

    vector_store = Chroma(
        client=persistent_client,
        persist_directory=os.path.join(ROOT_DIR, "/chroma_preloaded/"),  
        collection_name="all_docs_collection",
        embedding_function=embedding_function,
    )

    return vector_store

def setup_preloaded_RAG():
    vector_store = setup_preloaded_chroma()

    return articles_qa_runner.compose_qa(vector_store)

def setup_RAG(dataset_storage_url):
    article_scraper.scrape_articles(sitemap_url, scrap_url, dataset_storage_url)
    image_describer.describe_images(dataset_storage_url, dataset_storage_url)

    multi_modal_embedder.content_to_documents(dataset_storage_url, dataset_storage_url)

    vector_store = setup_chroma()
    multi_modal_embedder.embed_content(
        dataset_storage_url, dataset_storage_url, vector_store)

    return articles_qa_runner.compose_qa(vector_store)

