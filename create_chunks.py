import whisper
import json
import os

model = whisper.load_model("small")
audios = os.listdir("audios")

chunk_id = 1
for audion in audios:
    if "_" in audion:   
        number = audion.split("_")[0]
        title = audion.split("_", 1)[1]

        result = model.transcribe(
            audio=f"audios/{audion}",   
            language="hi",
            task="translate",
            word_timestamps=False
        )

        # ----- build chunks-------#
        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
            chunk_id += 1

        # -------save JSON ------#
        chunks_with_metadata = {
            "chunks": chunks,
            "text": result["text"]
        }

        # save with same name but .json
        with open(f"jsons/{audion}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)

        print(f" Processed {audion} -> jsons/{audion}.json")
