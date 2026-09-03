def create_chunks(text, chunk_size=500, overlap=100):
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks