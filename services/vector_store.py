import faiss
import numpy as np

def create_index(embeddings):
    embeddings = np.asarray(embeddings, dtype="float32")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Cannot create a FAISS index without embeddings.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_index(query_embedding, index, chunks, k=3):
    query_vector = np.asarray([query_embedding], dtype="float32")
    _, indices = index.search(query_vector, min(k, index.ntotal))
    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return retrieved_chunks