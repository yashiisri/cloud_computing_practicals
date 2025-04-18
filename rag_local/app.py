from embeddings.embedder import build_faiss_index
from retriever.search import retrieve_top_k
from rag.generator import generate_answer

def main():
    query = input("Enter your question: ")

    index, docs, embedder = build_faiss_index()
    top_results = retrieve_top_k(query, embedder, index, docs, k=2)

    print("\nTop Retrieved Docs:")
    for i, doc in enumerate(top_results):
        print(f"{i+1}. {doc}")

    context = ' '.join(top_results)
    answer = generate_answer(context, query)

    print("\nAnswer:")
    print(answer)

if __name__ == "__main__":
    main()
