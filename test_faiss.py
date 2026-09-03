from services.embeddings import create_embeddings
from services.vector_store import create_index
import numpy as np


chunks = [
    "Flask is a Python web framework.",
    "Python is a programming language.",
    "Machine learning is a branch of artificial intelligence."
]


# Create embeddings
embeddings = create_embeddings(chunks)


# Create FAISS index
index = create_index(embeddings)


# Ask a question
question = "What is Flask?"


# Convert question into embedding
question_embedding = create_embeddings([question])


# Search FAISS
question_embedding = np.array(question_embedding).astype("float32")

distances, indices = index.search(question_embedding, 2)


print("Question:", question)

print("\nRelevant chunks:")

for i in indices[0]:
    print(chunks[i])