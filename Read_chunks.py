import requests
import os
import json

# 🔥 Create embedding function
def create_embedding(text):
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "bge-m3",
            "prompt": text
        }
    )

    res = r.json()

    # ⚠️ error check
    if 'embedding' not in res:
        print("❌ Error in embedding:", res)
        return None

    return res['embedding']


# 📂 Read json files
jsons = os.listdir("jsons")

# 🔥 ONLY FIRST FILE (for testing)
jsons = jsons[:1]

chunk_id = 0

for json_file in jsons:
    print(f"\n📂 Processing file: {json_file}")

    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)

    # 🔥 process each chunk
    for chunk in content["chunks"]:
        print(f"⚡ Processing chunk {chunk_id}")

        embedding = create_embedding(chunk["text"])

        if embedding is None:
            continue

        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embedding

        chunk_id += 1

    # 💾 save updated file
    with open(f"jsons/{json_file}", "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)

    print(f"✅ Saved: {json_file}")

print("\n🎉 DONE!")