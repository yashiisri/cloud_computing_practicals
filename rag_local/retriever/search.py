def retrieve_top_k(query, model, index, docs, k=2):
    query_vec = model.encode([query]).astype('float32')
    D, I = index.search(query_vec, k)
    results = [docs[i] for i in I[0]]
    return results
