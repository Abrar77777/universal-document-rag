def split_text(text: str, chunk_size: int = 900, overlap: int = 150):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= overlap < chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + chunk_size * 0.55:
                end = boundary + 1
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        if end == text_length:
            start = end
        else:
            next_start = max(end - overlap, 0)
            space = text.find(" ", next_start, min(text_length, next_start + overlap + 80))
            start = max(start + 1, min(end, space + 1 if space != -1 else next_start))

    return chunks
