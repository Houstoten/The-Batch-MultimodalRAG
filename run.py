from app.pipelines.workflow import setup_RAG, setup_preloaded_RAG
from app.modules.qa_chain_composer.utils import create_multi_modal_query_from_description
from app.modules.image_processor.utils import generate_img_summaries, encode_image_from_bytes
import os

from kaggle import api 
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

if __name__ == "__main__":

    load_dotenv()

    os.environ['ROOT_DIR'] = os.path.dirname(os.path.abspath(__file__))

    dataset_storage_url = 'data'

    if not os.path.exists("./data/doc_dataset"):
        api.dataset_download_files('ivanhusarov/the-batch-articles-initial', path='./data/doc_dataset', unzip=True)
    if not os.path.exists("./data/image_descriptions_dataset"):
        api.dataset_download_files('ivanhusarov/the-batch-articles-image-descriptions', path='./data/image_descriptions_dataset', unzip=True)

    qa_chain = None
    if os.environ['USE_PRELOADED_VECTOR_STORE']:
        print("Setting up preloaded chroma")
        if not os.path.exists("./chroma_preloaded"):
            api.dataset_download_files('ivanhusarov/the-batch-rag-chroma', path='chroma_preloaded', unzip=True)
        qa_chain, retriever = setup_preloaded_RAG()
    else:
        qa_chain, retriever = setup_RAG(dataset_storage_url)

    df_path = f"{dataset_storage_url}/doc_dataset/articles_html.csv"
    df = pd.read_csv(df_path)
    df.set_index('Url', inplace=True)

    st.title("📄 The Batch articles RAG system")
    st.write(
        "Ask anything about tech and we will try to help you!",
    )
    st.write(    "At least image or question should be filled"
    )
    question = st.text_area(
        "Now ask your question!",
        placeholder="Explain the misclassification in AI, mentioning the bias in ML models?",
    )

    image_uploaded = st.file_uploader("Choose an image")
    if image_uploaded is not None:
        st.image(image_uploaded)


    if st.button('Process   ->'):
        image_description = None
        if image_uploaded:
            image_description = generate_img_summaries(image_uploaded, image_encoder=encode_image_from_bytes)

        query = create_multi_modal_query_from_description(question=question, image_description=image_description)
        
        retrieved_docs = retriever.get_relevant_documents(query)

        # Convert retrieved documents to context string
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Prepare the chain inputs
        chain_inputs = {
            "context": context,
            "text_input": question,
            "image_description": image_description
        }

        result = qa_chain.invoke(chain_inputs)

        answer = result['answer']
        retrieved_doc = retrieved_docs[0]

        doc_id = retrieved_doc.metadata['doc_id']
        relevance_score = retrieved_doc.metadata['relevance_score']
        image_found = retrieved_doc.metadata["image_url"]

        st.title("Retrieval query:")
        st.write(query)

        st.title("Answer:")
        st.write(answer)
        if image_found is not "":
            st.image(f"./data/doc_dataset/images_clean/{image_found}")

        if doc_id is not None:
            st.title("\nOriginal document:")
            st.write(f"Relevance score: {round(relevance_score, 2)}")
            st.html((df.loc[doc_id])['Content'])

            img_folder_path = f"./data/doc_dataset/images_clean/{os.path.basename(os.path.normpath(doc_id))}/"
            for img_path in os.listdir(img_folder_path):
                st.image(img_folder_path + img_path)
        
            st.write('\n\nCheck the source link:')
            st.write(retrieved_doc.metadata['doc_id'])
