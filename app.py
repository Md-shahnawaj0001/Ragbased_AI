import streamlit as st
import json
import os
import numpy as np
import requests

st.set_page_config(page_title="RAG AI Chatbot", layout="wide")

st.title("🤖 RAG AI Chatbot")
st.write("Ask anything from your data 📚")

# 🔥 embedding
def create_embedding(text):
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "bge-m3",
            "prompt": text
        }
    )
    return r.json()["embedding"]


# 🔥 LLM answer
def generate_answer(query, context):
    prompt = f"""
    You are an AI assistant.

    ONLY answer using the given context.
    If the answer is not present in the context, say:
    "I don't know based on the provided data."

    Context:
    {context}

    Question:
    {query}
    """

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    res = r.json()

    if "response" in res:
        return res["response"]
    else:
        return "❌ Error: " + str(res)


# 🔥 load chunks (once)
@st.cache_data
def load_data():
    all_chunks = []
    for file in os.listdir("jsons"):
        with open(f"jsons/{file}", encoding="utf-8") as f:
            content = json.load(f)
            all_chunks.extend(content["chunks"])
    return all_chunks

all_chunks = load_data()

st.success(f"Loaded {len(all_chunks)} chunks ✅")


# 🔥 cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 🔥 input box
query = st.text_input("💬 Ask your question:")

if st.button("Ask"):

    if query:
        with st.spinner("⏳ Thinking..."):

            query_embedding = create_embedding(query)

            scores = []

            for chunk in all_chunks:
                if "embedding" not in chunk:
                    continue

                score = cosine_similarity(
                    np.array(query_embedding),
                    np.array(chunk["embedding"])
                )

                scores.append((score, chunk))

            scores.sort(reverse=True, key=lambda x: x[0])
            top_chunks = [c[1] for c in scores[:3]]

            context = "\n".join([c["text"] for c in top_chunks])

            answer = generate_answer(query, context)

        st.subheader("🤖 Answer:")
        st.write(answer)

        st.subheader("📚 Retrieved Context:")
        for c in top_chunks:
            st.write("•", c["text"])