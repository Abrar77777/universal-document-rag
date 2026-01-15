def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks
