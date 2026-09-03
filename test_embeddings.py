from services.embeddings import create_embeddings


chunks = [
    "Flask is a Python web framework.",
    "Python is a programming language.",
    "Machine learning is a branch of artificial intelligence."
]


embeddings = create_embeddings(chunks)


print("Number of chunks:", len(chunks))

print("Embedding shape:", embeddings.shape)

print("First embedding:")
print(embeddings[0])