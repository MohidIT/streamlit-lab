import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
embeddings = np.load("embeddings.npy")
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

def retrieve_top_k(query_embedding, embeddings, k=10):
    sims = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_idx = sims.argsort()[-k:][::-1]
    return [(documents[i].strip(), float(sims[i])) for i in top_idx]

def get_query_embedding(query: str):
    # placeholder embedding
    return np.random.rand(embeddings.shape[1])

st.title("Information Retrieval using Document Embeddings")
query = st.text_input("Enter your query:")

if st.button("Search"):
    q_emb = get_query_embedding(query)
    results = retrieve_top_k(q_emb, embeddings, k=10)

    st.subheader("Top 10 Relevant Documents")
    for doc, score in results:
        st.write(f"- **{doc}** (Score: {score:.4f})")
