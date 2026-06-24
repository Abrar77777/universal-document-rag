def split_text(text: str, chunk_size: int = 900, overlap: int = 150):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

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
            start = space + 1 if space != -1 else next_start

    return chunks
