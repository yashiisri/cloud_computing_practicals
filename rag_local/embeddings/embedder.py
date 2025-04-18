from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_faiss_index(file_path='data/documents.txt'):
    model = SentenceTransformer('BAAI/bge-small-en')

    with open(file_path, 'r') as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]

    embeddings = model.encode(docs)
    embedding_array = np.array(embeddings).astype('float32')

    index = faiss.IndexFlatL2(embedding_array.shape[1])
    index.add(embedding_array)

    return index, docs, model
