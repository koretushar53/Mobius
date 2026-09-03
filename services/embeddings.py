from sentence_transformers import SentenceTransformer


model = None


def _get_model():
    global model

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


def create_embeddings(chunks):
    return _get_model().encode(chunks)