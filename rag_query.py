import json
import os
import numpy as np
import requests

# 🔥 create embedding
def create_embedding(text):
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "bge-m3",
            "prompt": text
        }
    )
    return r.json()["embedding"]


# 🔥 generate LLM answer (FIXED + IMPROVED)
def generate_answer(query, context):
    prompt = f"""
You are an expert React teacher.

Explain clearly in simple words with examples.

Context:
{context}

Question:
{query}

Answer:
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

    # 🔥 SAFE HANDLING (NO ERROR NOW)
    if "response" in res:
        return res["response"]
    else:
        return "❌ Error from LLM: " + str(res)


# 🔥 load all chunks
all_chunks = []

for file in os.listdir("jsons"):
    with open(f"jsons/{file}", encoding="utf-8") as f:
        content = json.load(f)
        all_chunks.extend(content["chunks"])

print(f"✅ Loaded {len(all_chunks)} chunks")


# 🔥 cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 🔥 query loop
while True:
    query = input("\n💬 Ask something: ")

    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting...")
        break

    print("⏳ Processing...")

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

    # 🔥 top 5 chunks (IMPROVED)
    scores.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [c[1] for c in scores[:5]]

    # 🔥 combine context
    context = "\n".join([c["text"] for c in top_chunks])

    # 🔥 generate final answer
    answer = generate_answer(query, context)

    print("\n🤖 Answer:\n")
    print(answer)
    print("\n" + "="*60)