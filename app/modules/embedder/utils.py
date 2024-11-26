def sliding_window(sentences, window_size, overlap):
    chunks = []
    for i in range(0, len(sentences), window_size - overlap):
        chunk = sentences[i:i + window_size]
        chunks.append(chunk)
    return chunks