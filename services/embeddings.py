from sentence_transformers import SentenceTransformer

model = None


def _get_model():
    global model

    if model is None:
        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

    return model


def create_embeddings(chunks):
    if not chunks:
        raise ValueError("No text was extracted from the PDF.")

    return _get_model().encode(
        chunks,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        device="cpu"
    )