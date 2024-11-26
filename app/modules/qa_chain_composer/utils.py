from app.modules.image_processor.utils import generate_img_summaries, encode_image_from_bytes

def create_multi_modal_query(question=None, image=None):
    query_arr = []
    if question:
        query_arr.append(question)

    if image:
        query_arr.append(generate_img_summaries(image, image_encoder=encode_image_from_bytes))
        
    return " . ".join(str(element) for element in query_arr)

def create_multi_modal_query_from_description(question=None, image_description=None):
    query_arr = []
    if question:
        query_arr.append(question)

    if image_description:
        query_arr.append(image_description)
        
    return " . ".join(str(element) for element in query_arr)