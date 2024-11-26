from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import base64

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def encode_image_from_bytes(image):
    """Getting the base64 string"""
    return base64.b64encode(image.getvalue()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """Make image summary"""
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(img_file, context="", image_encoder=encode_image):
    
    # Prompt
    prompt = f"""You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval. \
    Consider given context: {context}"""

    # Apply to images
    base64_image = image_encoder(img_file)

    return image_summarize(base64_image, prompt)
